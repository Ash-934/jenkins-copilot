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
    """Background service that watches Jenkins for build failures."""

    def __init__(self):
        self._seen_builds: dict[str, int] = {}
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
        logger.info(f"🔍 Build monitor started (interval={settings.MONITOR_INTERVAL}s)")

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
        """Main polling loop — runs every MONITOR_INTERVAL seconds."""
        # Initial delay to let the app start
        await asyncio.sleep(3)

        while self._running:
            try:
                await self._check_builds()
            except Exception as e:
                logger.error(f"Monitor poll error: {e}")
            await asyncio.sleep(settings.MONITOR_INTERVAL)

    async def _check_builds(self):
        """Check all jobs for new failures."""
        jenkins = JenkinsClient()
        try:
            jobs = await jenkins.list_jobs()

            for job in jobs:
                job_name = job.get("name", "")
                color = job.get("color", "")

                # Only check jobs that show a failure state
                # Jenkins colors: blue=success, red=fail, yellow=unstable, etc.
                if not any(c in color for c in ["red", "yellow", "aborted"]):
                    continue

                try:
                    await self._check_job(jenkins, job_name)
                except Exception as e:
                    logger.debug(f"Error checking job {job_name}: {e}")

        finally:
            await jenkins.close()

    async def _check_job(self, jenkins: JenkinsClient, job_name: str):
        """Check a specific job for new failures."""
        try:
            info = await jenkins.get_build_info(job_name, "lastBuild")
        except Exception:
            return

        build_num = info.get("number", 0)
        result = info.get("result", "")

        # Skip if we've already seen this build
        if self._seen_builds.get(job_name, 0) >= build_num:
            return

        self._seen_builds[job_name] = build_num

        # Only alert on failures / unstable
        if result not in ("FAILURE", "UNSTABLE", "ABORTED"):
            return

        logger.info(f"🚨 Detected failed build: {job_name} #{build_num} ({result})")

        # Fetch log and classify
        try:
            log_text = await jenkins.get_build_log_tail(job_name, "lastBuild", lines=120)
        except Exception:
            log_text = ""

        error_matches = classify_error(log_text) if log_text else []

        if error_matches:
            # Create one alert per matched pattern
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
                logger.info(f"  → Alert: {match['category']} in {job_name} #{build_num}")

                # Auto-diagnose if enabled
                if self.auto_diagnose and self._auto_diagnose_callback:
                    try:
                        await self._auto_diagnose_callback(job_name, build_num, alert)
                        alert.auto_diagnosed = True
                    except Exception as e:
                        logger.error(f"Auto-diagnose failed: {e}")
        else:
            # Generic failure alert (no pattern matched)
            alert = Alert(
                id=str(uuid.uuid4())[:8],
                job_name=job_name,
                build_number=build_num,
                result=result,
                category="❌ Build Failed",
                severity="error",
                fix="Check the build log for details. Click 'Diagnose' for AI analysis.",
                snippet=log_text[-500:] if log_text else "No log available",
                timestamp=time.time(),
            )
            self._alerts.append(alert)

        # Cap alerts at 50
        if len(self._alerts) > 50:
            self._alerts = self._alerts[-50:]


# -- Singleton --

monitor = BuildMonitor()
