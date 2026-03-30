"""
Supervisor Agent - routes user queries to the appropriate specialist agent(s).

Architecture:
    User → Supervisor (Router) → Agent(s) → Synthesizer → Response
"""

import logging
import traceback
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from config import settings
from agents import jobs_agent, logs_agent, plugins_agent, infra_agent

logger = logging.getLogger("chatbot.supervisor")

# -- All available agents --

AGENT_REGISTRY = {
    "jobs": {
        "module": jobs_agent,
        "label": "📋 Jobs",
        "description": "List jobs, check status, view job configurations",
    },
    "logs": {
        "module": logs_agent,
        "label": "📜 Logs",
        "description": "Analyze build logs, diagnose failures, view pipeline stages",
    },
    "plugins": {
        "module": plugins_agent,
        "label": "🔌 Plugins",
        "description": "List installed plugins, recommend plugins, check compatibility",
    },
    "infra": {
        "module": infra_agent,
        "label": "🖥️ Infrastructure",
        "description": "Monitor build queue, agents/nodes, system health",
    },
}

DEFAULT_ACTIVE_AGENTS = set(AGENT_REGISTRY.keys())


# -- State ----

class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    active_agents: set[str]       # Which agents the user has toggled on
    selected_agents: list[str]    # Which agents the router picked (1 or more)
    agent_outputs: dict[str, str] # Collected outputs: {agent_name: response}
    pending_agents: list[str]     # Agents still to execute
    agent_reasoning: str


# -- LLM ----

def get_llm():
    return ChatOpenAI(
        base_url=settings.LM_STUDIO_BASE_URL,
        api_key=settings.LM_STUDIO_API_KEY,
        model=settings.LM_STUDIO_MODEL,
        temperature=0.1,
    )


# -- Supervisor router node --

ROUTER_PROMPT = """\
You are the Jenkins AI Copilot Supervisor. Analyze the user's message and decide \
which specialist agent(s) should handle it.

Available agents:
{agent_descriptions}

RESPOND WITH ONLY the agent name(s) as a comma-separated list. Examples:
- "jobs" — for simple job listing
- "logs" — for build failure analysis
- "logs, infra" — when a build is stuck (need both logs and queue/agent info)
- "jobs, plugins" — when asking about job setup and required plugins

Rules:
- Build failures, logs, errors, "why did X fail" → jobs, logs
- List jobs, job status, job config → jobs
- Plugins, extensions, what's installed → plugins
- Queue, agents, nodes, stuck builds → infra
- "Why is my build stuck" → logs, infra (need both)
- "Set up a new pipeline" → jobs, plugins (need config + plugin recommendations)
- General greeting or chitchat → jobs
- If unsure, pick just one. Only select multiple when clearly needed.
"""


async def router_node(state: SupervisorState) -> dict:
    """Classify the query and decide which agent(s) to invoke."""
    active = state.get("active_agents", DEFAULT_ACTIVE_AGENTS)
    active_agents = {k: v for k, v in AGENT_REGISTRY.items() if k in active}

    # Single agent active → skip router LLM call
    if len(active_agents) == 1:
        name = list(active_agents.keys())[0]
        logger.info(f"Only one agent active, routing to: {name}")
        return {
            "selected_agents": [name],
            "pending_agents": [name],
            "agent_outputs": {},
            "agent_reasoning": "Single active agent",
        }

    # Build prompt
    desc = "\n".join(f"- {n}: {i['description']}" for n, i in active_agents.items())
    names = ", ".join(active_agents.keys())
    prompt = ROUTER_PROMPT.format(agent_descriptions=desc, agent_names=names)

    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    llm = get_llm()
    try:
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Route this query: {user_text}"),
        ])
        raw = response.content.strip().lower()

        candidates = [s.strip() for s in raw.replace("\n", ",").split(",")]
        selected = [c for c in candidates if c in active_agents]

        logger.info(f"Router selected: {selected} (query: {user_text[:60]})")
        return {
            "selected_agents": selected,
            "pending_agents": list(selected),
            "agent_outputs": {},
            "agent_reasoning": f"Routed to {', '.join(selected)}",
        }

    except Exception as e:
        logger.error(f"Router error: {e}\n{traceback.format_exc()}")
        fallback = list(active_agents.keys())[0]
        return {
            "selected_agents": [fallback],
            "pending_agents": [fallback],
            "agent_outputs": {},
            "agent_reasoning": f"Router failed, defaulting to {fallback}",
        }


# -- Agent Executor Node --

async def agent_executor_node(state: SupervisorState) -> dict:
    pending = list(state.get("pending_agents", []))
    outputs = dict(state.get("agent_outputs", {}))

    if not pending:
        return {"pending_agents": [], "agent_outputs": outputs}

    # Pop the next agent to execute
    current_agent = pending.pop(0)
    agent_info = AGENT_REGISTRY.get(current_agent)

    if not agent_info:
        logger.warning(f"Unknown agent: {current_agent}, skipping")
        return {"pending_agents": pending, "agent_outputs": outputs}

    module = agent_info["module"]
    logger.info(f"▶ Executing agent: {current_agent} ({module.AGENT_NAME})")

    try:
        llm = get_llm()
        react_agent = create_react_agent(
            model=llm,
            tools=module.TOOLS,
            prompt=module.SYSTEM_PROMPT,
        )

        # Build messages — start with conversation history
        agent_messages = list(state["messages"])

        # If previous agents have already produced output, inject it as context
        # so this agent can use their findings (e.g., job names from jobs_agent)
        if outputs:
            context_parts = []
            for prev_agent, prev_output in outputs.items():
                label = AGENT_REGISTRY.get(prev_agent, {}).get("label", prev_agent)
                context_parts.append(f"[{label} findings]:\n{prev_output}")

            context_msg = HumanMessage(content=(
                "The following information was already gathered by other agents. "
                "Use it as context for your analysis:\n\n"
                + "\n\n---\n\n".join(context_parts)
            ))
            agent_messages.append(context_msg)

        result = await react_agent.ainvoke({"messages": agent_messages})
        output_messages = result.get("messages", [])

        # Log tool calls
        for msg in output_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    logger.info(f"  🔧 [{current_agent}] {tc['name']}({tc.get('args', {})})")

        # Extract the final response
        response = ""
        for msg in reversed(output_messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                response = msg.content
                break

        outputs[current_agent] = response or "No response generated."

    except Exception as e:
        logger.error(f"Agent {current_agent} failed: {e}\n{traceback.format_exc()}")
        outputs[current_agent] = f"Error from {current_agent}: {e}"

    return {"pending_agents": pending, "agent_outputs": outputs}


# -- Routing Logic: Loop or Synthesize --

def should_continue(state: SupervisorState) -> str:
    pending = state.get("pending_agents", [])
    if pending:
        return "agent_executor"
    return "synthesizer"


# -- Synthesizer Node --

SYNTH_PROMPT = """\
You are the Jenkins AI Copilot. Multiple specialist agents have analyzed the user's \
query and provided their findings. Synthesize their outputs into a single, coherent, \
well-structured response.

Do NOT just concatenate the outputs. Merge them intelligently:
- Remove duplicate information
- Organize by relevance
- Highlight the most actionable insights first
- Use markdown formatting
- If agents produced conflicting info, note it

Agent outputs:
{agent_outputs}
"""


async def synthesizer_node(state: SupervisorState) -> dict:
    outputs = state.get("agent_outputs", {})
    selected = state.get("selected_agents", [])

    # Single agent → just return its output directly (no synthesis needed)
    if len(outputs) <= 1:
        content = list(outputs.values())[0] if outputs else "No response generated."
        return {"messages": [AIMessage(content=content)]}

    # Multiple agents → synthesize
    logger.info(f"Synthesizing outputs from: {list(outputs.keys())}")

    formatted = "\n\n".join(
        f"### {AGENT_REGISTRY.get(name, {}).get('label', name)} Agent:\n{text}"
        for name, text in outputs.items()
    )

    llm = get_llm()
    try:
        last_msg = state["messages"][-1]
        user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        response = await llm.ainvoke([
            SystemMessage(content=SYNTH_PROMPT.format(agent_outputs=formatted)),
            HumanMessage(content=f"Original user query: {user_text}"),
        ])
        return {"messages": [AIMessage(content=response.content)]}

    except Exception as e:
        logger.error(f"Synthesizer failed: {e}\n{traceback.format_exc()}")
        # Fallback: just concatenate with headers
        fallback = "\n\n---\n\n".join(
            f"**{AGENT_REGISTRY.get(n, {}).get('label', n)}:**\n{t}"
            for n, t in outputs.items()
        )
        return {"messages": [AIMessage(content=fallback)]}


# -- Build the Graph --

def build_supervisor_graph(active_agents: set[str] | None = None):
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("agent_executor", agent_executor_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Edges
    graph.set_entry_point("router")
    graph.add_edge("router", "agent_executor")

    # After agent_executor: loop back if more pending, else synthesize
    graph.add_conditional_edges(
        "agent_executor",
        should_continue,
        {"agent_executor": "agent_executor", "synthesizer": "synthesizer"},
    )

    graph.add_edge("synthesizer", END)

    compiled = graph.compile()
    active = active_agents or DEFAULT_ACTIVE_AGENTS
    logger.info(f"✅ Supervisor graph built (multi-agent capable). Active: {active}")
    return compiled


# -- Convenience ----

def get_available_agents() -> dict:
    return {
        name: {"label": info["label"], "description": info["description"]}
        for name, info in AGENT_REGISTRY.items()
    }
