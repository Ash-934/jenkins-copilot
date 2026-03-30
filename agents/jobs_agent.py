"""Jobs Agent - lists jobs, checks status, gets job info."""

from langchain_core.tools import tool
from agents import get_client, to_json

AGENT_NAME = "jobs_agent"

SYSTEM_PROMPT = """\
You are the Jobs Agent, a specialist in Jenkins job management.
Your responsibility is to help users understand their Jenkins jobs - listing them,
checking status, getting job details, and providing insights.

Guidelines:
- Always list jobs when the user asks vaguely about "my jobs" or "what do I have".
- Show job status using clear indicators (SUCCESS ✅, FAILURE ❌, UNSTABLE ⚠️, etc.).
- If a user asks about a specific job, get its detailed info.
- Format your responses in markdown tables when listing multiple jobs.
"""


@tool
async def list_jobs() -> str:
    """List all Jenkins jobs with their current status (name, url, color/status).
    Use when the user asks about their jobs, projects, or pipelines."""
    result = await get_client().list_jobs()
    return to_json(result)


@tool
async def get_job_info(job_name: str) -> str:
    """Get detailed info for a specific job including last build, health report.
    Args:
        job_name: Name of the Jenkins job
    """
    try:
        result = await get_client().get_job_info(job_name)
        return to_json(result)
    except Exception as e:
        return f"Error fetching job info: {e}"


@tool
async def get_job_config(job_name: str) -> str:
    """Fetch a job's XML configuration (includes Jenkinsfile for pipeline jobs).
    Args:
        job_name: Name of the Jenkins job
    """
    try:
        xml = await get_client().get_job_config_xml(job_name)
        if len(xml) > 8000:
            return xml[:8000] + "\n... [truncated]"
        return xml
    except Exception as e:
        return f"Error fetching job config: {e}"


TOOLS = [list_jobs, get_job_info, get_job_config]
