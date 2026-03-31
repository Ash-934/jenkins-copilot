import logging
import traceback

from langchain_core.messages import HumanMessage

from config import settings
from agents.supervisor import (
    build_supervisor_graph,
    get_available_agents,
    DEFAULT_ACTIVE_AGENTS,
)

logger = logging.getLogger("chatbot.agent")

# -- Session state --

# session: conversation history + active agents
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages": [],
            "active_agents": set(DEFAULT_ACTIVE_AGENTS),
            "graph": None,
        }
    return _sessions[session_id]


# -- Agent activation --

def set_active_agents(session_id: str, agent_names: list[str]):
    session = _get_session(session_id)
    available = set(get_available_agents().keys())
    valid = set(agent_names) & available

    if not valid:
        raise ValueError(f"No valid agents in {agent_names}. Available: {available}")

    session["active_agents"] = valid
    session["graph"] = None  # Force rebuild
    logger.info(f"[{session_id[:8]}] Active agents set to: {valid}")


def get_session_agents(session_id: str) -> dict:
    session = _get_session(session_id)
    all_agents = get_available_agents()
    return {
        name: {**info, "active": name in session["active_agents"]}
        for name, info in all_agents.items()
    }


# -- Chat --

async def chat(session_id: str, user_message: str) -> str:
    """
    Process a user message through the multi-agent supervisor.

    Args:
        session_id: Conversation session ID
        user_message: The user's message

    Returns:
        The assistant's response text
    """
    session = _get_session(session_id)
    logger.debug(f"[{session_id[:8]}] Processing: {user_message[:80]}")

    # Build/rebuild graph if needed
    if session["graph"] is None:
        try:
            session["graph"] = build_supervisor_graph(session["active_agents"])
        except Exception as e:
            logger.error(f"[{session_id[:8]}] Graph build failed:\n{traceback.format_exc()}")
            raise RuntimeError(
                f"Failed to build agent graph - is LM Studio running at "
                f"{settings.LLM_BASE_URL}? Error: {e}"
            ) from e

    graph = session["graph"]

    # Append user message to session history
    session["messages"].append(HumanMessage(content=user_message))

    # Invoke the supervisor graph
    try:
        result = await graph.ainvoke({
            "messages": session["messages"],
            "active_agents": session["active_agents"],
            "selected_agents": [],
            "pending_agents": [],
            "agent_outputs": {},
            "agent_reasoning": "",
        })
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[{session_id[:8]}] Supervisor invoke failed:\n{tb}")

        err_str = str(e).lower()
        if "connection refused" in err_str or "connect call failed" in err_str:
            raise ConnectionError(
                "Cannot connect to LM Studio. Start it and load a model."
            ) from e

        raise RuntimeError(f"Agent error: {type(e).__name__}: {e}") from e

    # Extract response
    output_messages = result.get("messages", [])
    if not output_messages:
        return "I received an empty response. Please try again."

    response_text = ""
    for msg in reversed(output_messages):
        if hasattr(msg, "content") and msg.content:
            from langchain_core.messages import AIMessage
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                response_text = msg.content
                break

    if not response_text:
        response_text = "I processed your request but couldn't generate a response."

    # Update session history with the result
    session["messages"] = list(output_messages)

    agents_used = result.get("selected_agents", ["unknown"])
    logger.info(f"[{session_id[:8]}] Agents: {agents_used} | Response: {len(response_text)} chars")

    return response_text


def clear_session(session_id: str):
    """Clear a session's history and graph."""
    _sessions.pop(session_id, None)
