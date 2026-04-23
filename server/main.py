"""
server/main.py
FastAPI backend for the Kaizen OS dashboard.

Endpoints
---------
WS  /ws               — real-time state broadcast to all dashboard clients
POST /start_episode   — trigger a new agent episode (non-blocking)
GET  /status          — current episode number, agent status, WS stats
GET  /health          — liveness probe for HF Spaces / Docker

Architecture
------------
The agent loop runs as an asyncio background task so it never blocks the
WebSocket handler or HTTP endpoints.  A single asyncio.Lock prevents two
episodes from running simultaneously if /start_episode is called rapidly.

CORS is enabled for all origins so the Vite dev server (localhost:5173)
and the HuggingFace Space frontend can both connect without proxy config.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from server.broadcast import manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
# The LLMAgent is expensive to initialise (model download + GPU load).
# We load it once on first /start_episode call, not at import time, so the
# server starts instantly and health checks pass immediately.

_agent = None
_agent_lock = asyncio.Lock()
_episode_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Episode state (shared across endpoints)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "status": "idle",           # idle | loading | running | done | error
    "episode": 0,
    "current_step": 0,
    "max_steps": 10,
    "last_reward": 0.0,
    "cumulative_reward": 0.0,
    "chaos_active": None,
    "started_at": None,
    "error": None,
}


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background ping task on startup; cancel on shutdown."""
    ping_task = asyncio.create_task(_ping_loop())
    logger.info("[Server] Kaizen OS backend started.")
    yield
    ping_task.cancel()
    try:
        await ping_task
    except asyncio.CancelledError:
        pass
    logger.info("[Server] Kaizen OS backend shutting down.")


async def _ping_loop() -> None:
    """Send a keep-alive ping to all WS clients every 30 seconds."""
    while True:
        await asyncio.sleep(30)
        if manager.connection_count > 0:
            await manager.ping_all()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kaizen OS — Agentic Kernel",
    description="Real-time LLM OS management agent dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Accept a dashboard client.  Sends current state immediately on connect
    so the UI is never blank on first load, then stays open for broadcasts.
    """
    await manager.connect(ws)
    try:
        # Send current state immediately so the UI isn't blank
        await ws.send_text(
            _build_hello_payload()
        )
        # Keep the connection alive — the client drives disconnection
        while True:
            # We don't expect messages from the client, but we must await
            # something to yield to the event loop and detect disconnects.
            data = await ws.receive_text()
            # Echo commands back (allows client to send "ping" for latency test)
            if data == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as exc:
        logger.warning(f"[WS] Unexpected error: {exc}")
        manager.disconnect(ws)


def _build_hello_payload() -> str:
    import json
    return json.dumps({
        "type": "hello",
        "status": _state["status"],
        "episode": _state["episode"],
        "message": "Connected to Kaizen OS backend. Waiting for episode start.",
        "ws_clients": manager.connection_count,
    })


# ---------------------------------------------------------------------------
# POST /start_episode
# ---------------------------------------------------------------------------

@app.post("/start_episode")
async def start_episode() -> dict[str, Any]:
    """
    Trigger a new agent episode.

    If an episode is already running, returns 409.
    Model loading happens lazily on the first call.
    The episode runs as a non-blocking background task.
    """
    if _state["status"] == "running":
        return {
            "status": "error",
            "message": "An episode is already running. Wait for it to finish.",
        }

    if _episode_lock.locked():
        return {
            "status": "error",
            "message": "Episode start already in progress.",
        }

    asyncio.create_task(_run_episode_task())

    return {
        "status": "accepted",
        "message": "Episode starting. Watch the WebSocket for live updates.",
        "episode": _state["episode"] + 1,
    }


async def _run_episode_task() -> None:
    """
    Background task that loads the agent (if needed) and runs one episode.
    Updates _state throughout so /status always reflects reality.
    """
    global _agent

    async with _episode_lock:
        _state["status"] = "loading"
        _state["error"] = None
        _state["started_at"] = time.time()

        # Broadcast loading state
        await manager.broadcast({
            "type": "status",
            "status": "loading",
            "message": "Loading LLM agent...",
        })

        # Lazy-load the agent in a thread so we don't block the event loop
        # during the multi-second model load.
        try:
            async with _agent_lock:
                if _agent is None:
                    _agent = await asyncio.get_event_loop().run_in_executor(
                        None, _load_agent
                    )
        except Exception as exc:
            logger.error(f"[Server] Agent load failed: {exc}")
            _state["status"] = "error"
            _state["error"] = str(exc)
            await manager.broadcast({
                "type": "error",
                "message": f"Failed to load agent: {exc}",
            })
            return

        # Import here to avoid circular import at module level
        from environment.kaizen_env import KaizenEnv

        env = KaizenEnv(broadcast=True, broadcaster=manager)
        obs, info = env.reset()

        _state["status"] = "running"
        _state["episode"] = env.episode
        _state["current_step"] = 0

        logger.info(f"[Server] Episode {env.episode} started.")

        await manager.broadcast({
            "type": "status",
            "status": "running",
            "episode": env.episode,
            "message": f"Episode {env.episode} started.",
        })

        # Run the episode step by step — offload LLM inference to a thread
        while not env.is_done:
            try:
                current_obs = env.current_obs

                # Run LLM inference without blocking the event loop
                raw_output, action, error = await asyncio.get_event_loop().run_in_executor(
                    None, _agent.act, current_obs
                )

                # Step the environment (sync call, fast — no GPU work)
                obs, reward, terminated, truncated, step_info = env.step(raw_output)

                _state["current_step"] = obs.get("step", 0)
                _state["last_reward"] = reward
                _state["cumulative_reward"] = step_info.get("cumulative_reward", 0.0)
                _state["chaos_active"] = obs.get("active_chaos")

                # The env already broadcasts via _schedule_broadcast(),
                # but we yield here so the event loop can flush messages.
                await asyncio.sleep(0)

            except Exception as exc:
                logger.error(f"[Server] Step error: {exc}", exc_info=True)
                _state["status"] = "error"
                _state["error"] = str(exc)
                await manager.broadcast({
                    "type": "error",
                    "message": f"Step error: {exc}",
                })
                return

        # Episode complete
        _state["status"] = "done"
        logger.info(
            f"[Server] Episode {env.episode} done. "
            f"Cumulative reward: {_state['cumulative_reward']:.3f}"
        )

        await manager.broadcast({
            "type": "episode_done",
            "episode": env.episode,
            "cumulative_reward": _state["cumulative_reward"],
            "steps": _state["current_step"],
            "message": "Episode complete. Click Start Episode to run another.",
        })


def _load_agent():
    """
    Synchronous agent loader — runs in a thread executor.
    Reads MODEL_NAME from env so it can be overridden at HF eval time.

    Set KAIZEN_DEMO_MODE=true to skip LLM loading entirely and use a
    rule-based dummy agent for fast dashboard testing without a GPU.
    """
    # Demo mode — instant start, no model download needed
    if os.environ.get("KAIZEN_DEMO_MODE", "false").lower() == "true":
        from agent.demo_agent import DemoAgent
        return DemoAgent()

    from agent.llm_agent import LLMAgent
    model_name = os.environ.get("KAIZEN_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    load_in_4bit = os.environ.get("KAIZEN_4BIT", "true").lower() == "true"
    return LLMAgent(model_name=model_name, load_in_4bit=load_in_4bit)


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/status")
async def get_status() -> dict[str, Any]:
    """
    Return current agent and server status.

    Used by the HF Space UI and the TopBar component to show live state.
    """
    uptime = None
    if _state["started_at"] is not None:
        uptime = round(time.time() - _state["started_at"], 1)

    return {
        "status": _state["status"],
        "episode": _state["episode"],
        "current_step": _state["current_step"],
        "max_steps": _state["max_steps"],
        "last_reward": _state["last_reward"],
        "cumulative_reward": _state["cumulative_reward"],
        "chaos_active": _state["chaos_active"],
        "error": _state["error"],
        "uptime_seconds": uptime,
        "ws_connections": manager.connection_count,
        "total_broadcasts": manager.total_broadcasts,
        "model": os.environ.get("KAIZEN_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for HF Spaces, Docker, and load balancers."""
    return {"status": "ok", "service": "kaizen-os-backend"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
