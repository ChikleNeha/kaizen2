# HuggingFace Space Dockerfile
# Builds frontend, serves FastAPI backend on port 7860
# Uses DemoAgent by default (no GPU needed on Space)
# For live LLM demo: run server locally on A100 Colab + ngrok

FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
# Point WebSocket at the Space's own backend
RUN echo 'VITE_WS_URL=wss://nehachikle-kaizen-os.hf.space/ws' > .env
RUN npm run build

# ── Python backend ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CPU-only — Space runs demo mode, no model load)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn>=0.29.0 \
    websockets>=12.0 \
    psutil>=5.9.0 \
    pydantic>=2.0.0 \
    gymnasium>=0.29.0

# Copy project
COPY . .

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Serve frontend via FastAPI static files
RUN pip install --no-cache-dir aiofiles

# Environment
ENV KAIZEN_DEMO_MODE=true
ENV PORT=7860
EXPOSE 7860

RUN useradd -m -u 1000 user
USER user

# Mount static frontend
CMD ["python", "-m", "uvicorn", "server.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info"]