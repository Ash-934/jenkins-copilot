"""
Proactive Build Failure Monitor

Background service that polls Jenkins for new build failures,
classifies errors via regex pattern matching.
"""

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from jenkins_client import JenkinsClient
from config import settings

logger = logging.getLogger("chatbot.monitor")


# -- Error Pattern Engine --

@dataclass
class ErrorPattern:
    name: str
    category: str
    regex: str
    fix: str
    severity: str = "error"  # error, warning


ERROR_PATTERNS = [
    ErrorPattern(
        name="fatal",
        category="Fatal Error",
        regex=r"(FATAL|CRITICAL|SEVERE)",
        fix="A critical error occurred. Click Diagnose for AI analysis.",
    ),
    ErrorPattern(
        name="failure",
        category="❌ Build Failure",
        regex=r"(FAILURE|FAILED|BUILD FAILED|COMPILATION ERROR)",
        fix="Build failed. Click Diagnose to identify the root cause.",
    ),
    ErrorPattern(
        name="error",
        category="Error",
        regex=r"(ERROR|Exception|Traceback)",
        fix="Errors detected in the build log. Click Diagnose for details.",
        severity="warning",
    ),
    ErrorPattern(
        name="timeout",
        category="Timeout",
        regex=r"(timed?\s*out|deadline exceeded|timeout)",
        fix="Build timed out. Check for hanging processes.",
    ),
    ErrorPattern(
        name="permission",
        category="Access Issue",
        regex=r"(Permission denied|Access Denied|403 Forbidden|Unauthorized)",
        fix="Access denied. Check credentials and permissions.",
    ),
]


def classify_error(log_text: str) -> list[dict]:
    """Match log text against error patterns, return all matches."""
    matches = []
    for pattern in ERROR_PATTERNS:
        m = re.search(pattern.regex, log_text, re.IGNORECASE | re.MULTILINE)
        if m:
            # Extract a few lines of context around the match
            start = max(0, m.start() - 200)
            end = min(len(log_text), m.end() + 200)
            snippet = log_text[start:end].strip()
            # Trim to nearest newlines
            lines = snippet.split("\n")
            if len(lines) > 6:
                lines = lines[:6]
            snippet = "\n".join(lines)

            matches.append({
                "pattern": pattern.name,
                "category": pattern.category,
                "severity": pattern.severity,
                "fix": pattern.fix,
                "snippet": snippet,
            })
    return matches


# -- Alert Model --

@dataclass
class Alert:
    id: str
    job_name: str
    build_number: int
    result: str
    category: str
    severity: str
    fix: str
    snippet: str
    timestamp: float
    dismissed: bool = False
    auto_diagnosed: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_name": self.job_name,
            "build_number": self.build_number,
            "result": self.result,
            "category": self.category,
            "severity": self.severity,
            "fix": self.fix,
            "snippet": self.snippet,
            "timestamp": self.timestamp,
            "dismissed": self.dismissed,
            "auto_diagnosed": self.auto_diagnosed,
        }


# -- Monitor Service --

class BuildMonitor:
    """Background service that watches Jenkins for build failures and build time anomalies."""

    DURATION_HISTORY_SIZE = 10
    SLOW_BUILD_THRESHOLD = 2.0

    def __init__(self):
        self._seen_builds: dict[str, int] = {}
        self._build_durations: dict[str, list[int]] = {}
        self._alerts: list[Alert] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.auto_diagnose = False
        self._auto_diagnose_callback = None

    @property
    def alerts(self) -> list[Alert]:
        return [a for a in self._alerts if not a.dismissed]

    def get_all_alerts(self) -> list[dict]:
        return [a.to_dict() for a in self.alerts]

    def dismiss_alert(self, alert_id: str) -> bool:
        for a in self._alerts:
            if a.id == alert_id:
                a.dismissed = True
                return True
        return False

    def clear_alerts(self):
        self._alerts = []

    def set_auto_diagnose_callback(self, callback):
        """Set callback: async fn(job_name, build_number, alert) for auto-diagnosis."""
        self._auto_diagnose_callback = callback

    async def start(self):
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Build monitor started (interval={settings.MONITOR_INTERVAL}s)")

    async def stop(self):
        """Stop the monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Build monitor stopped")

    async def _poll_loop(self):
        """Main polling loop."""
        await asyncio.sleep(3)
        while self._running:
            try:
                await self._check_builds()
            except Exception as e:
                logger.error(f"Monitor poll error: {e}")
            await asyncio.sleep(settings.MONITOR_INTERVAL)

    async def _check_builds(self):
        """Check all jobs for failures and build time anomalies."""
        jenkins = JenkinsClient()
        try:
            jobs = await jenkins.list_jobs()
            for job in jobs:
                job_name = job.get("name", "")
                try:
                    await self._check_job(jenkins, job_name)
                except Exception as e:
                    logger.debug(f"Error checking job {job_name}: {e}")
        finally:
            await jenkins.close()

    async def _check_job(self, jenkins: JenkinsClient, job_name: str):
        """Check a job for new failures AND build time anomalies."""
        try:
            info = await jenkins.get_build_info(job_name, "lastBuild")
        except Exception:
            return

        build_num = info.get("number", 0)
        result = info.get("result", "")
        duration = info.get("duration", 0)

        # Skip if we've already seen this build
        if self._seen_builds.get(job_name, 0) >= build_num:
            return

        # If this job has no duration history, backfill from recent builds
        if job_name not in self._build_durations:
            await self._backfill_durations(jenkins, job_name, build_num)

        self._seen_builds[job_name] = build_num

        # -- Build time anomaly detection --
        if duration > 0:
            await self._check_build_time(job_name, build_num, duration, result)

        # -- Failure detection --
        if result in ("FAILURE", "UNSTABLE", "ABORTED"):
            await self._handle_failure(jenkins, job_name, build_num, result)

        self._cap_alerts()

    async def _backfill_durations(self, jenkins: JenkinsClient, job_name: str, current_build_num: int):
        """Seed duration history from recent builds so spike detection works on first run."""
        try:
            recent = await jenkins.get_recent_builds(job_name, count=self.DURATION_HISTORY_SIZE)
            durations = []
            for build in reversed(recent):  # oldest first
                d = build.get("duration", 0)
                num = build.get("number", 0)
                # Skip the current build (it will be added by _check_build_time)
                if d > 0 and num != current_build_num:
                    durations.append(d)
            if durations:
                self._build_durations[job_name] = durations[-self.DURATION_HISTORY_SIZE:]
                logger.debug(f"Backfilled {len(durations)} build durations for {job_name}")
        except Exception as e:
            logger.debug(f"Could not backfill durations for {job_name}: {e}")

    async def _check_build_time(self, job_name: str, build_num: int, duration: int, result: str):
        """Track build duration and alert on sharp spikes."""
        history = self._build_durations.get(job_name, [])

        if len(history) >= 3:
            avg = sum(history) / len(history)
            if avg > 0 and duration > avg * self.SLOW_BUILD_THRESHOLD:
                ratio = duration / avg
                duration_s = duration // 1000
                avg_s = int(avg) // 1000

                alert = Alert(
                    id=str(uuid.uuid4())[:8],
                    job_name=job_name,
                    build_number=build_num,
                    result=result or "SUCCESS",
                    category="Slow Build",
                    severity="warning",
                    fix=f"Build took {duration_s}s, which is {ratio:.1f}x the average ({avg_s}s). Check for new dependencies, larger assets, or infrastructure issues.",
                    snippet=f"Duration: {duration_s}s | Average: {avg_s}s | Ratio: {ratio:.1f}x",
                    timestamp=time.time(),
                )
                self._alerts.append(alert)
                logger.info(f"Slow build: {job_name} #{build_num} took {duration_s}s ({ratio:.1f}x avg)")

                if self.auto_diagnose and self._auto_diagnose_callback:
                    try:
                        await self._auto_diagnose_callback(job_name, build_num, alert)
                        alert.auto_diagnosed = True
                    except Exception as e:
                        logger.error(f"Auto-diagnose failed for slow build: {e}")

        # Update history (rolling window)
        history.append(duration)
        if len(history) > self.DURATION_HISTORY_SIZE:
            history = history[-self.DURATION_HISTORY_SIZE:]
        self._build_durations[job_name] = history

    async def _handle_failure(self, jenkins: JenkinsClient, job_name: str, build_num: int, result: str):
        """Handle a failed build: classify error and create alerts."""
        logger.info(f"Detected failed build: {job_name} #{build_num} ({result})")

        try:
            log_text = await jenkins.get_build_log_tail(job_name, "lastBuild", lines=120)
        except Exception:
            log_text = ""

        error_matches = classify_error(log_text) if log_text else []

        if error_matches:
            for match in error_matches:
                alert = Alert(
                    id=str(uuid.uuid4())[:8],
                    job_name=job_name,
                    build_number=build_num,
                    result=result,
                    category=match["category"],
                    severity=match["severity"],
                    fix=match["fix"],
                    snippet=match["snippet"],
                    timestamp=time.time(),
                )
                self._alerts.append(alert)
                logger.info(f"  Alert: {match['category']} in {job_name} #{build_num}")

                if self.auto_diagnose and self._auto_diagnose_callback:
                    try:
                        await self._auto_diagnose_callback(job_name, build_num, alert)
                        alert.auto_diagnosed = True
                    except Exception as e:
                        logger.error(f"Auto-diagnose failed: {e}")
        else:
            alert = Alert(
                id=str(uuid.uuid4())[:8],
                job_name=job_name,
                build_number=build_num,
                result=result,
                category="Build Failed",
                severity="error",
                fix="Check the build log for details. Click 'Diagnose' for AI analysis.",
                snippet=log_text[-500:] if log_text else "No log available",
                timestamp=time.time(),
            )
            self._alerts.append(alert)

            if self.auto_diagnose and self._auto_diagnose_callback:
                try:
                    await self._auto_diagnose_callback(job_name, build_num, alert)
                    alert.auto_diagnosed = True
                except Exception as e:
                    logger.error(f"Auto-diagnose failed: {e}")

    def _cap_alerts(self):
        if len(self._alerts) > 50:
            self._alerts = self._alerts[-50:]


# -- Singleton --

monitor = BuildMonitor()
