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

Fix (v2)
--------
- KaizenEnv is now a module-level singleton so _episode increments correctly
  across multiple /start_episode calls (was resetting to 1 every call).
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
# Lazy model + env loading
# ---------------------------------------------------------------------------

_agent = None
_agent_lock = asyncio.Lock()
_episode_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Module-level KaizenEnv singleton
# ---------------------------------------------------------------------------
# CRITICAL FIX: The env must be created ONCE at module level, not inside
# _run_episode_task(). Creating it fresh each call resets _episode to 0,
# so env.reset() always returns episode=1.
#
# We defer actual construction until first use (lazy) so the import doesn't
# fail if the server is started before the environment package is ready.

_env = None

def _get_env():
    """Return the module-level KaizenEnv, creating it on first call."""
    global _env
    if _env is None:
        from environment.kaizen_env import KaizenEnv
        _env = KaizenEnv(broadcast=True, broadcaster=manager)
        logger.info("[Server] KaizenEnv singleton created.")
    return _env

# ---------------------------------------------------------------------------
# Episode state (shared across endpoints)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "status":            "idle",
    "episode":           0,
    "current_step":      0,
    "max_steps":         10,
    "last_reward":       0.0,
    "cumulative_reward": 0.0,
    "chaos_active":      None,
    "started_at":        None,
    "error":             None,
}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    await manager.connect(ws)
    try:
        await ws.send_text(_build_hello_payload())
        while True:
            data = await ws.receive_text()
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
        "type":       "hello",
        "status":     _state["status"],
        "episode":    _state["episode"],
        "message":    "Connected to Kaizen OS backend. Waiting for episode start.",
        "ws_clients": manager.connection_count,
    })


# ---------------------------------------------------------------------------
# POST /start_episode
# ---------------------------------------------------------------------------

@app.post("/start_episode")
async def start_episode() -> dict[str, Any]:
    """
    Trigger a new agent episode.
    Returns 409-equivalent JSON if an episode is already running.
    """
    if _state["status"] == "running":
        return {
            "status":  "error",
            "message": "An episode is already running. Wait for it to finish.",
        }

    if _episode_lock.locked():
        return {
            "status":  "error",
            "message": "Episode start already in progress.",
        }

    asyncio.create_task(_run_episode_task())

    # Return the NEXT episode number for the UI to show immediately
    next_ep = _state["episode"] + 1
    return {
        "status":  "accepted",
        "message": "Episode starting. Watch the WebSocket for live updates.",
        "episode": next_ep,
    }


async def _run_episode_task() -> None:
    """
    Background task: loads agent if needed, then runs one full episode.
    Uses the module-level env singleton so episode numbers increment correctly.
    """
    global _agent

    async with _episode_lock:
        _state["status"]     = "loading"
        _state["error"]      = None
        _state["started_at"] = time.time()

        await manager.broadcast({
            "type":    "status",
            "status":  "loading",
            "message": "Loading LLM agent...",
        })

        # Lazy-load the agent in a thread so we don't block the event loop
        try:
            async with _agent_lock:
                if _agent is None:
                    _agent = await asyncio.get_event_loop().run_in_executor(
                        None, _load_agent
                    )
        except Exception as exc:
            logger.error(f"[Server] Agent load failed: {exc}")
            _state["status"] = "error"
            _state["error"]  = str(exc)
            await manager.broadcast({
                "type":    "error",
                "message": f"Failed to load agent: {exc}",
            })
            return

        # ---- Use the singleton env — DO NOT construct a new one here ----
        env = _get_env()
        obs, info = env.reset()          # increments env._episode by 1

        _state["status"]       = "running"
        _state["episode"]      = env.episode   # now 2, 3, 4… on repeat calls
        _state["current_step"] = 0

        logger.info(f"[Server] Episode {env.episode} started.")

        await manager.broadcast({
            "type":    "status",
            "status":  "running",
            "episode": env.episode,
            "message": f"Episode {env.episode} started.",
        })

        # Episode loop
        while not env.is_done:
            try:
                current_obs = env.current_obs

                # Run LLM inference in thread — never blocks the event loop
                raw_output, action, error = await asyncio.get_event_loop().run_in_executor(
                    None, _agent.act, current_obs
                )

                obs, reward, terminated, truncated, step_info = env.step(raw_output)

                _state["current_step"]      = obs.get("step", 0)
                _state["last_reward"]       = reward
                _state["cumulative_reward"] = step_info.get("cumulative_reward", 0.0)
                _state["chaos_active"]      = obs.get("active_chaos")

                # Yield to event loop so WS messages flush
                await asyncio.sleep(0)

            except Exception as exc:
                logger.error(f"[Server] Step error: {exc}", exc_info=True)
                _state["status"] = "error"
                _state["error"]  = str(exc)
                await manager.broadcast({
                    "type":    "error",
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
            "type":             "episode_done",
            "episode":          env.episode,
            "cumulative_reward": _state["cumulative_reward"],
            "steps":            _state["current_step"],
            "message":          "Episode complete. Click Start Episode to run another.",
        })


def _load_agent():
    """
    Synchronous agent loader — runs in a thread executor.

    Priority order:
      1. KAIZEN_DEMO_MODE=true       → DemoAgent (no model needed)
      2. KAIZEN_MODEL_PATH exists    → load from local path (downloaded at build)
      3. KAIZEN_MODEL_NAME set       → load from HF Hub identifier
      4. Fallback                    → DemoAgent with a warning
    """
    demo_mode = os.environ.get("KAIZEN_DEMO_MODE", "false").lower() == "true"
    if demo_mode:
        from agent.demo_agent import DemoAgent
        return DemoAgent()

    # Check if model was downloaded at build time
    local_path  = os.environ.get("KAIZEN_MODEL_PATH", "/app/kaizen_grpo_model")
    model_name  = os.environ.get("KAIZEN_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    use_unsloth = os.environ.get("KAIZEN_USE_UNSLOTH", "false").lower() == "true"

    # Determine which path to use
    def model_valid(path):
        if not os.path.isdir(path):
            return False
        indicators = ["adapter_config.json", "config.json",
                      "model.safetensors", "pytorch_model.bin"]
        return any(os.path.exists(os.path.join(path, f)) for f in indicators)

    if model_valid(local_path):
        source = local_path
        logger.info(f"[Server] Loading model from local path: {source}")
    elif model_name and model_name != "Qwen/Qwen2.5-3B-Instruct":
        source = model_name
        logger.info(f"[Server] Loading model from HF Hub: {source}")
    else:
        logger.warning("[Server] No trained model found — falling back to DemoAgent")
        from agent.demo_agent import DemoAgent
        return DemoAgent()

    try:
        from agent.llm_agent import LLMAgent
        return LLMAgent(model_name=source, use_unsloth=use_unsloth)
    except Exception as e:
        logger.error(f"[Server] LLMAgent load failed: {e} — falling back to DemoAgent")
        from agent.demo_agent import DemoAgent
        return DemoAgent()

# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/status")
async def get_status() -> dict[str, Any]:
    uptime = None
    if _state["started_at"] is not None:
        uptime = round(time.time() - _state["started_at"], 1)

    return {
        "status":            _state["status"],
        "episode":           _state["episode"],
        "current_step":      _state["current_step"],
        "max_steps":         _state["max_steps"],
        "last_reward":       _state["last_reward"],
        "cumulative_reward": _state["cumulative_reward"],
        "chaos_active":      _state["chaos_active"],
        "error":             _state["error"],
        "uptime_seconds":    uptime,
        "ws_connections":    manager.connection_count,
        "total_broadcasts":  manager.total_broadcasts,
        "model":             os.environ.get("KAIZEN_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "kaizen-os-backend"}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Serve React frontend — must be AFTER all API/WebSocket routes
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(os.path.join(_dist, "index.html"))
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