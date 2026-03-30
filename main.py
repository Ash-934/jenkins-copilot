"""FastAPI application - multi-agent chat"""

import logging
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import (
    chat as agent_chat,
    clear_session,
    set_active_agents,
    get_session_agents,
)
from jenkins_client import JenkinsClient

# -- Logging setup --

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chatbot")


# -- Lifespan --

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    logger.info("🚀 Jenkins AI Copilot starting (multi-agent supervisor)")

    try:
        jenkins = JenkinsClient()
        ok = await jenkins.health_check()
        if ok:
            info = await jenkins.get_system_info()
            logger.info(f"✅ Connected to Jenkins {info['version']} at {info['url']}")
        else:
            logger.warning("⚠️  Could not connect to Jenkins")
        await jenkins.close()
    except Exception:
        logger.error(f"❌ Jenkins connection error:\n{traceback.format_exc()}")

    yield
    logger.info("👋 Shutting down...")


# -- App --

app = FastAPI(
    title="Jenkins AI Copilot",
    description="Multi-agent AI assistant for Jenkins",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Global exception handler --

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.method} {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": type(exc).__name__},
    )


# -- Models --

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    error: str | None = None


class AgentToggleRequest(BaseModel):
    agents: list[str]


# -- Chat route --

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(f"💬 [{session_id[:8]}] User: {req.message[:100]}")

    try:
        response_text = await agent_chat(session_id, req.message)
        logger.info(f"✅ [{session_id[:8]}] Response: {response_text[:100]}...")
        return ChatResponse(response=response_text, session_id=session_id)

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ [{session_id[:8]}] Chat error:\n{tb}")
        return ChatResponse(
            response=f"**Error:** {type(e).__name__}: {e}",
            session_id=session_id,
            error=str(e),
        )


# -- Agent activation routes --

@app.get("/api/agents")
async def list_agents(session_id: str | None = None):
    sid = session_id or "default"
    return get_session_agents(sid)


@app.put("/api/agents")
async def toggle_agents(req: AgentToggleRequest, session_id: str | None = None):
    sid = session_id or "default"
    try:
        set_active_agents(sid, req.agents)
        return {"status": "ok", "active_agents": req.agents}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -- Session routes --

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/sessions")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared"}


@app.get("/api/health")
async def health():
    result = {"api": "ok", "architecture": "multi-agent-supervisor", "lm_studio": "unknown", "jenkins": "unknown"}
    try:
        import httpx
        from config import settings
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.LM_STUDIO_BASE_URL}/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                result["lm_studio"] = "connected"
                result["lm_studio_models"] = [m.get("id") for m in models]
            else:
                result["lm_studio"] = "error"
    except Exception as e:
        result["lm_studio"] = f"unreachable: {e}"

    try:
        jenkins = JenkinsClient()
        jenkins_ok = await jenkins.health_check()
        info = await jenkins.get_system_info() if jenkins_ok else {}
        await jenkins.close()
        result["jenkins"] = "connected" if jenkins_ok else "unreachable"
        result["jenkins_version"] = info.get("version")
    except Exception as e:
        result["jenkins"] = f"unreachable: {e}"

    from agents.supervisor import get_available_agents
    result["available_agents"] = get_available_agents()
    return result


# -- Static files --

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    from config import settings
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
