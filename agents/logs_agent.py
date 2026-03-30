"""Logs Agent - fetches, parses, and analyzes build logs."""

from langchain_core.tools import tool
from agents import get_client, to_json

AGENT_NAME = "logs_agent"

SYSTEM_PROMPT = """\
You are the Logs Agent, a specialist in Jenkins build log analysis.
Your responsibility is to fetch build logs, identify errors, diagnose failures,
and suggest fixes based on log content.

Guidelines:
- If the user does NOT specify a job name, ALWAYS call list_jobs first to discover available jobs.
- When asked "why did my build fail?", fetch the job list first (if no name given), then the log and build info.
- Identify the specific error lines in the log and quote them.
- Recognize common error patterns: compilation errors, test failures, OOM, timeouts,
  permission denied, dependency resolution failures, Docker build errors.
- Always suggest concrete fixes, not just explanations.
- When analyzing performance, compare stage durations across recent builds.
- Format error snippets in code blocks.
"""


@tool
async def list_jobs() -> str:
    """List all Jenkins jobs with their name and current status.
    ALWAYS call this first if the user hasn't specified a job name."""
    result = await get_client().list_jobs()
    return to_json(result)


@tool
async def get_build_log(job_name: str, build_number: str = "lastBuild") -> str:
    """Fetch the last 80 lines of a build's console log.
    Use when diagnosing build failures or checking build output.
    Args:
        job_name: Name of the Jenkins job
        build_number: Build number, or 'lastBuild', 'lastFailedBuild'. Defaults to 'lastBuild'.
    """
    try:
        return await get_client().get_build_log_tail(job_name, build_number)
    except Exception as e:
        return f"Error fetching build log: {e}"


@tool
async def get_build_info(job_name: str, build_number: str = "lastBuild") -> str:
    """Get metadata about a build - result, duration, trigger cause, changes.
    Args:
        job_name: Name of the Jenkins job
        build_number: Build number or 'lastBuild'. Defaults to 'lastBuild'.
    """
    try:
        result = await get_client().get_build_info(job_name, build_number)
        return to_json(result)
    except Exception as e:
        return f"Error fetching build info: {e}"


@tool
async def get_pipeline_stages(job_name: str, build_number: str = "lastBuild") -> str:
    """Get stage-by-stage breakdown of a pipeline build with timings and status.
    Use to identify which stage failed or is the bottleneck.
    Args:
        job_name: Name of the Jenkins pipeline job
        build_number: Build number or 'lastBuild'. Defaults to 'lastBuild'.
    """
    try:
        result = await get_client().get_pipeline_stages(job_name, build_number)
        return to_json(result)
    except Exception as e:
        return f"Error fetching pipeline stages: {e}"


@tool
async def get_recent_builds(job_name: str, count: int = 10) -> str:
    """Get summary of the last N builds (number, result, duration) for trend analysis.
    Args:
        job_name: Name of the Jenkins job
        count: Number of recent builds to fetch. Defaults to 10.
    """
    try:
        result = await get_client().get_recent_builds(job_name, count)
        return to_json(result)
    except Exception as e:
        return f"Error fetching recent builds: {e}"


TOOLS = [list_jobs, get_build_log, get_build_info, get_pipeline_stages, get_recent_builds]
