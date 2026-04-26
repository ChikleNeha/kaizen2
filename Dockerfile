# HuggingFace Space Dockerfile
# Builds frontend, downloads GRPO model from HF Hub, serves FastAPI on port 7860

FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Python backend ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
# NOTE: CPU-only torch because HF free Spaces have no GPU.
# The model runs in 4-bit on CPU — slow but functional for demo.
# For fast inference use ngrok on Colab A100 instead (see README).
RUN pip install --no-cache-dir \
    "numpy<2" \
    fastapi>=0.110.0 \
    uvicorn>=0.29.0 \
    websockets>=12.0 \
    psutil>=5.9.0 \
    "pydantic>=2.0.0" \
    "gymnasium>=0.29.0" \
    aiofiles \
    "huggingface_hub>=0.23.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.28.0" \
    "torch>=2.4.0" --extra-index-url https://download.pytorch.org/whl/cpu \
    bitsandbytes \
    peft \

# Copy project code
COPY . .

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# ── Model download script ─────────────────────────────────────────────
# Downloads GRPO model at build time so it's baked into the Space image.
# Uses HF_TOKEN build arg (set in Space secrets as a build secret).
ARG HF_TOKEN=""
ARG KAIZEN_MODEL_NAME=""

RUN if [ -n "$KAIZEN_MODEL_NAME" ] && [ -n "$HF_TOKEN" ]; then \
        echo "Downloading model: $KAIZEN_MODEL_NAME" && \
        python -c " \
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='$KAIZEN_MODEL_NAME', \
    repo_type='model', \
    local_dir='/app/kaizen_grpo_model', \
    token='$HF_TOKEN' \
)" && echo "✅ Model downloaded"; \
    else \
        echo "⚠️ No model configured — Space will use DemoAgent"; \
    fi

ENV KAIZEN_DEMO_MODE=false
ENV KAIZEN_MODEL_PATH=/app/kaizen_grpo_model
ENV PORT=7860
EXPOSE 7860

RUN useradd -m -u 1000 user
USER user

CMD ["python", "-m", "uvicorn", "server.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info"]