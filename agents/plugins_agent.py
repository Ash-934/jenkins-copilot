"""Plugins Agent - lists installed plugins, recommends plugins, checks compatibility."""

from langchain_core.tools import tool
from agents import get_client, to_json

AGENT_NAME = "plugins_agent"

SYSTEM_PROMPT = """\
You are the Plugins Agent, a specialist in Jenkins plugin management.
Your responsibility is to help users understand their installed plugins,
recommend plugins for specific use cases, and identify plugin issues.

Guidelines:
- When asked about plugins, always check what's installed first.
- Flag plugins that have updates available.
- If a user asks for recommendations, base them on their current setup.
- Know the popular plugins: Git, Pipeline, Docker, Kubernetes, Credentials,
  Blue Ocean, Slack Notification, JUnit, Cobertura, etc.
- Warn about known compatibility issues between plugins.
- Format plugin lists in markdown tables.
"""


@tool
async def list_plugins() -> str:
    """List all installed Jenkins plugins with name, version, and status.
    Use when checking installed plugins, recommending additions, or auditing."""
    try:
        result = await get_client().list_plugins()
        return to_json(result)
    except Exception as e:
        return f"Error fetching plugins: {e}"


TOOLS = [list_plugins]
