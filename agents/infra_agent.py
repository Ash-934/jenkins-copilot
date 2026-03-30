"""Infra Agent - monitors queue, agents/nodes, and system health."""

from langchain_core.tools import tool
from agents import get_client, to_json

AGENT_NAME = "infra_agent"

SYSTEM_PROMPT = """\
You are the Infrastructure Agent, a specialist in Jenkins infrastructure health.
Your responsibility is to monitor build agents/nodes, the build queue, and
system resources to help users troubleshoot infrastructure issues.

Guidelines:
- When builds are stuck, check BOTH the queue AND agent status.
- Explain why builds are waiting (no matching agents, all busy, label mismatch, etc.).
- Show agent status clearly: online/offline, idle/busy, labels.
- Suggest solutions: start agents, change labels, increase executors.
- Format output in clear markdown tables.
"""


@tool
async def get_queue_status() -> str:
    """Get the Jenkins build queue - shows pending builds and why they're waiting.
    Use when builds are stuck or slow to start."""
    try:
        result = await get_client().get_queue()
        if not result:
            return "The build queue is empty - no pending builds."
        return to_json(result)
    except Exception as e:
        return f"Error fetching queue: {e}"


@tool
async def get_agents() -> str:
    """List all Jenkins build agents/nodes with status (online/offline, idle/busy, labels).
    Use when troubleshooting agent availability or capacity."""
    try:
        result = await get_client().get_agents()
        return to_json(result)
    except Exception as e:
        return f"Error fetching agents: {e}"


TOOLS = [get_queue_status, get_agents]
