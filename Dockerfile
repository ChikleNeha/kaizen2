# =============================================================================
# Kaizen OS — HuggingFace Space Dockerfile
# =============================================================================
# This is the TOP-LEVEL Dockerfile used by HF Spaces (Docker SDK).
# It is different from docker/Dockerfile.sandbox (which is for action isolation).
#
# Build order:
#   1. Install Node + npm (for frontend build)
#   2. Install Python dependencies
#   3. Copy project files
#   4. Build the React frontend
#   5. Start the FastAPI backend via app.py
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# System dependencies — Node.js for frontend build
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .

# Install torch CPU-only first to save space on CPU Space instances
# At hackathon eval with GPU Space, remove the --index-url override
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy project
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Build frontend
# ---------------------------------------------------------------------------
RUN cd frontend \
    && npm install \
    && npm run build \
    && echo "Frontend build complete"

# ---------------------------------------------------------------------------
# Expose port
# ---------------------------------------------------------------------------
EXPOSE 7860

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------
ENV PORT=7860
ENV KAIZEN_MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
ENV KAIZEN_4BIT="true"

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
CMD ["python", "app.py"]
