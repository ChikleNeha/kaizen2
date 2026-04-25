#!/bin/bash
# =============================================================================
# run_training.sh
# Runs INSIDE the HF Job container (pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel)
# Called by launch_job.py via: bash run_training.sh
#
# What this script does:
#   1. Verify GPU
#   2. Install all Python dependencies
#   3. Clone the kaizen2 GitHub repo (contains training/ and environment/)
#   4. Run SFT  → saves adapter to /output/kaizen_sft_model
#   5. Run GRPO → saves adapter to /output/kaizen_grpo_model
#   6. Push both adapters to HF Hub (NehaChikle/kaizen-qwen2.5-3b-sft)
#
# Environment variables expected (set by launch_job.py):
#   HF_TOKEN    — your HuggingFace token (passed as secret)
#   HF_REPO_ID  — target model repo, e.g. NehaChikle/kaizen-qwen2.5-3b-sft
#   GITHUB_REPO — repo to clone, e.g. https://github.com/ChikleNeha/kaizen2
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
AMBER='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[KAIZEN]${NC} $*"; }
warn() { echo -e "${AMBER}[WARN]${NC}  $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 0. Sanity checks ─────────────────────────────────────────────────────────
log "===== Kaizen OS — HF Training Job ====="
log "Date: $(date -u)"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || die "No GPU found. Check that you launched with a GPU flavor."

python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"

# ── 1. Validate required env vars ────────────────────────────────────────────
: "${HF_TOKEN:?HF_TOKEN secret is not set. Aborting.}"
: "${HF_REPO_ID:?HF_REPO_ID env var is not set. Aborting.}"
: "${GITHUB_REPO:?GITHUB_REPO env var is not set. Aborting.}"

log "Target HF repo : $HF_REPO_ID"
log "Source GitHub  : $GITHUB_REPO"

# ── 2. Install dependencies ───────────────────────────────────────────────────
log "Installing dependencies..."

# Unsloth — GPU build matching CUDA 12.4
pip install --quiet \
  "unsloth[cu124-torch260] @ git+https://github.com/unslothai/unsloth.git" \
  || {
    warn "Unsloth GPU install failed, falling back to colab-new variant..."
    pip install --quiet \
      "unsloth @ git+https://github.com/unslothai/unsloth.git"
  }

pip install --quiet \
  "trl==0.12.2" \
  "datasets>=2.18.0" \
  "accelerate>=0.28.0" \
  "bitsandbytes>=0.43.0" \
  "peft>=0.10.0" \
  "transformers>=4.40.0" \
  "matplotlib" \
  "wandb" \
  "huggingface_hub>=1.0.0" \
  "pydantic>=2.0.0" \
  "psutil>=5.9.0" \
  "gymnasium>=0.29.0"

log "All dependencies installed."

# ── 3. Clone the kaizen2 repo ─────────────────────────────────────────────────
WORKDIR="/workspace/kaizen2"
log "Cloning $GITHUB_REPO → $WORKDIR ..."

git clone --depth=1 "$GITHUB_REPO" "$WORKDIR" \
  || die "Failed to clone $GITHUB_REPO"

cd "$WORKDIR"
log "Repo cloned. Contents:"
ls -la

# Verify critical files exist
[[ -f "training/sft_train.py" ]]     || die "training/sft_train.py not found in repo"
[[ -f "training/grpo_train.py" ]]    || die "training/grpo_train.py not found in repo"
[[ -f "training/golden_examples.json" ]] || die "training/golden_examples.json not found in repo"

EXAMPLE_COUNT=$(python3 -c "import json; d=json.load(open('training/golden_examples.json')); print(len(d))")
log "Training examples found: $EXAMPLE_COUNT"

# ── 4. HF login ───────────────────────────────────────────────────────────────
log "Logging into HuggingFace Hub..."
python3 -c "
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('HF login OK')
"

# ── 5. SFT Training ───────────────────────────────────────────────────────────
log "===== PHASE 1: Supervised Fine-Tuning (SFT) ====="

SFT_OUTPUT="/workspace/kaizen_sft_model"

HF_REPO_ID="$HF_REPO_ID" \
SFT_OUTPUT_DIR="$SFT_OUTPUT" \
python3 training/sft_train.py \
  || die "SFT training failed"

log "SFT complete. Adapter at $SFT_OUTPUT"
ls -lh "$SFT_OUTPUT" 2>/dev/null || warn "SFT output dir is empty — check sft_train.py OUTPUT_DIR"

# ── 6. GRPO Training ──────────────────────────────────────────────────────────
log "===== PHASE 2: GRPO Reinforcement Learning ====="

GRPO_OUTPUT="/workspace/kaizen_grpo_model"

SFT_MODEL_PATH="$SFT_OUTPUT" \
HF_REPO_ID="${HF_REPO_ID}-grpo" \
GRPO_OUTPUT_DIR="$GRPO_OUTPUT" \
python3 training/grpo_train.py \
  || {
    warn "GRPO training failed — SFT adapter is still saved and usable."
    warn "Check grpo_train.py logs above for the error."
    # Don't exit — SFT success is still valuable
  }

# ── 7. Push to HF Hub ────────────────────────────────────────────────────────
log "===== PHASE 3: Pushing adapters to HuggingFace Hub ====="

python3 - <<'PYEOF'
import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ["HF_TOKEN"]
sft_repo  = os.environ["HF_REPO_ID"]                  # NehaChikle/kaizen-qwen2.5-3b-sft
grpo_repo = sft_repo.replace("-sft", "-grpo")          # NehaChikle/kaizen-qwen2.5-3b-grpo

sft_dir  = "/workspace/kaizen_sft_model"
grpo_dir = "/workspace/kaizen_grpo_model"

# Push SFT adapter
if os.path.isdir(sft_dir) and os.listdir(sft_dir):
    print(f"[PUSH] Uploading SFT adapter to {sft_repo}...")
    api.create_repo(sft_repo, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=sft_dir,
        repo_id=sft_repo,
        repo_type="model",
        token=token,
        commit_message="Kaizen OS SFT adapter — LoRA Qwen2.5-3B",
    )
    print(f"[PUSH] SFT adapter pushed to https://huggingface.co/{sft_repo}")
else:
    print("[PUSH] SFT adapter directory is empty or missing — skipping SFT push.")

# Push GRPO adapter
if os.path.isdir(grpo_dir) and os.listdir(grpo_dir):
    print(f"[PUSH] Uploading GRPO adapter to {grpo_repo}...")
    api.create_repo(grpo_repo, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        folder_path=grpo_dir,
        repo_id=grpo_repo,
        repo_type="model",
        token=token,
        commit_message="Kaizen OS GRPO adapter — RL fine-tuned from SFT",
    )
    print(f"[PUSH] GRPO adapter pushed to https://huggingface.co/{grpo_repo}")
else:
    print("[PUSH] GRPO adapter directory is empty or missing — skipping GRPO push.")

print("[PUSH] Done.")
PYEOF

# ── 8. Copy artefacts (loss curves) to output dir ────────────────────────────
log "Copying artefacts..."
mkdir -p /workspace/artefacts
cp /workspace/kaizen2/loss_curve.png       /workspace/artefacts/ 2>/dev/null || true
cp /workspace/kaizen2/reward_curve.png     /workspace/artefacts/ 2>/dev/null || true
cp /workspace/kaizen2/*_loss.json          /workspace/artefacts/ 2>/dev/null || true
ls /workspace/artefacts/ 2>/dev/null || true

log "===== Training Job Complete ====="
log "SFT  adapter : https://huggingface.co/${HF_REPO_ID}"
log "GRPO adapter : https://huggingface.co/${HF_REPO_ID%-sft}-grpo"
log ""
log "Next steps:"
log "  1. In your HF Space (NehaChikle/kaizen-os), set:"
log "     KAIZEN_MODEL_NAME = ${HF_REPO_ID}"
log "     KAIZEN_DEMO_MODE  = false"
log "  2. Commit loss_curve.png and reward_curve.png to your GitHub repo"