"""
app.py
HuggingFace Space entry point.

HF Spaces with Docker SDK expect the app to listen on $PORT (default 7860).
This file starts the FastAPI backend via uvicorn on that port.

The frontend is served as static files from frontend/dist/ so the Space
is fully self-contained — one URL serves both the dashboard and the API.

Build the frontend first:
    cd frontend && npm install && npm run build

Then push the whole repo to HF Spaces.
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Build frontend if dist/ doesn't exist
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent / "frontend"
DIST_DIR     = FRONTEND_DIR / "dist"

if not DIST_DIR.exists():
    print("[Space] Building frontend...")
    result = subprocess.run(
        ["npm", "install"],
        cwd=FRONTEND_DIR,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"[Space] npm install failed: {result.stderr.decode()[:500]}")
    else:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=FRONTEND_DIR,
            capture_output=True,
        )
        if result.returncode == 0:
            print("[Space] Frontend built successfully.")
        else:
            print(f"[Space] Frontend build failed: {result.stderr.decode()[:500]}")

# ---------------------------------------------------------------------------
# Patch the FastAPI app to serve the frontend static files
# ---------------------------------------------------------------------------
# Import the app AFTER the frontend build so StaticFiles can mount dist/
from server.main import app  # noqa: E402

if DIST_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    # Mount at root — must be last so API routes take priority
    app.mount(
        "/",
        StaticFiles(directory=str(DIST_DIR), html=True),
        name="frontend",
    )
    print(f"[Space] Serving frontend from {DIST_DIR}")
else:
    print("[Space] WARNING: frontend/dist not found. Dashboard will not be available.")
    print("[Space] API endpoints (/ws, /start_episode, /status, /health) are still active.")

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    print(f"[Space] Starting Kaizen OS on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
