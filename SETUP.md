# Kaizen OS — Setup Guide ---
# ========================
# Complete instructions for local dev, Colab training, and HF Space deployment.

---

## STEP 0 — Prerequisites

Install these on your machine before anything else:

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10 or 3.11 | https://python.org |
| Node.js | 18 or 20 | https://nodejs.org |
| Git | any | https://git-scm.com |

Optional but recommended:
- Google account (for Colab)
- HuggingFace account (for Space + model hub)

---

## STEP 1 — Extract the project

```bash
unzip kaizen_os.zip
cd kaizen_os
```

You should see this structure:
```
kaizen_os/
├── environment/
├── agent/
├── training/
├── server/
├── frontend/
├── docker/
├── requirements.txt
├── README.md
└── app.py
```

---

## STEP 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> On some systems use `pip3` instead of `pip`.
> If you get conflicts, use a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate      # Mac/Linux
> venv\Scripts\activate         # Windows
> pip install -r requirements.txt
> ```

---

## STEP 3 — Run the backend

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     [Server] Kaizen OS backend started.
```

The LLM model loads lazily — it only downloads on the first "Start Episode" click.
The server itself starts in under 1 second.

---

## STEP 4 — Run the frontend

Open a NEW terminal (keep the backend running):

```bash
cd frontend
npm install
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in 300ms
  ➜  Local:   http://localhost:5173/
```

Open http://localhost:5173 in your browser.

---

## STEP 5 — Run your first episode

1. Open http://localhost:5173
2. You should see the dark dashboard with all vitals at zero
3. Click **Start Episode** in the top bar
4. The model will download (~2GB for Qwen2.5-3B) on the first run — this takes a few minutes
5. Once loaded, the episode starts automatically
6. Watch chaos inject at step 3 — a node turns red in the graph
7. Watch the agent reason and act in the ReasoningPanel

> If the model download is too slow, set this env var to use a smaller model:
> ```bash
> export KAIZEN_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
> ```
> Then restart the backend.

---

## STEP 6 — Training on Google Colab (optional but recommended for demo)

### 6a — Upload to Google Drive

1. Zip the `kaizen_os` folder: `zip -r kaizen_os.zip kaizen_os/`
2. Go to https://drive.google.com
3. Upload `kaizen_os.zip` to your Drive

### 6b — Open Google Colab

Go to https://colab.research.google.com and create a new notebook.
Make sure you have a T4 GPU: Runtime → Change runtime type → T4 GPU

### 6c — Run these cells in order

**Cell 1 — Mount Drive and extract:**
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/kaizen_os.zip', 'r') as z:
    z.extractall('/content/')

%cd /content/kaizen_os
```

**Cell 2 — Install Unsloth (MUST use this exact command on Colab):**
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl datasets accelerate bitsandbytes wandb matplotlib
```

**Cell 3 — Run SFT training (~12 minutes):**
```python
!python training/sft_train.py
```

Expected final output:
```
[SFT] Final loss: 0.81
[SFT] Adapter saved to ./kaizen_sft_model
```

**Cell 4 — Run GRPO training (~25 minutes):**
```python
!python training/grpo_train.py
```

Expected final output:
```
[GRPO] Reward summary: +0.82 → +4.71 (Δ +3.89)
[GRPO] Reward curve saved to: ./reward_curve.png
```

**Cell 5 — Download the trained model and reward curve:**
```python
from google.colab import files
import shutil

# Zip the trained model
shutil.make_archive('kaizen_grpo_model', 'zip', '.', 'kaizen_grpo_model')
files.download('kaizen_grpo_model.zip')
files.download('reward_curve.png')
```

### 6d — Use the trained model locally

1. Unzip `kaizen_grpo_model.zip` into your `kaizen_os/` folder
2. Set the env var before starting the backend:
```bash
export KAIZEN_MODEL_NAME="./kaizen_grpo_model"
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## STEP 7 — Deploy to HuggingFace Spaces

### 7a — Create a HuggingFace account

Go to https://huggingface.co/join

### 7b — Create a new Space

1. Go to https://huggingface.co/new-space
2. Name it: `kaizen-os`
3. SDK: **Docker**
4. Hardware: **CPU Basic** (free) — upgrade to GPU T4 at hackathon eval
5. Click Create Space

### 7c — Set Space secrets

In your Space → Settings → Variables and secrets, add:

| Name | Value |
|------|-------|
| `KAIZEN_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` |
| `KAIZEN_4BIT` | `true` |
| `PORT` | `7860` |

### 7d — Push the code

```bash
cd kaizen_os

# Initialize git if not already done
git init
git add .
git commit -m "Initial commit — Kaizen OS Agentic Kernel"

# Add HuggingFace remote (replace YOUR_USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/kaizen-os

# Push
git push hf main
```

If prompted for credentials:
- Username: your HuggingFace username
- Password: your HuggingFace **access token** (Settings → Access Tokens → New token → Write)

### 7e — Update the Space README

Rename `README_SPACE.md` to `README.md` in the HF Space repo
(or copy its contents — it has the required YAML frontmatter for HF Spaces).

The Space will build automatically. Takes 3-5 minutes on first push.

### 7f — Access your Space

Your dashboard will be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/kaizen-os`

---

## STEP 8 — At hackathon evaluation (HF Credits)

When evaluators provide HF credits, upgrade the model in one step:

1. Go to your Space → Settings → Variables and secrets
2. Change `KAIZEN_MODEL_NAME` to: `Qwen/Qwen2.5-14B-Instruct`
3. Change hardware to: **GPU T4 medium** or better
4. Click Save — Space rebuilds automatically

No code changes needed. Everything else is identical.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'gymnasium'"**
```bash
pip install gymnasium psutil pydantic fastapi uvicorn
```

**"CUDA out of memory"**
```bash
export KAIZEN_4BIT=true
# or use a smaller model:
export KAIZEN_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
```

**"WebSocket connection failed" in browser**
- Make sure the backend is running on port 8000
- Check `frontend/.env.example` — copy to `.env.local` if needed
- Try: `VITE_WS_URL=ws://localhost:8000/ws npm run dev`

**"npm: command not found"**
- Install Node.js from https://nodejs.org (LTS version)

**Frontend shows blank page**
- Open browser console (F12) — check for errors
- Make sure you ran `npm install` before `npm run dev`

**Colab: "unsloth not found" even after install**
- Restart the Colab runtime after installing: Runtime → Restart runtime
- Then re-run the training cell (skip the install cell)

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `uvicorn server.main:app --port 8000 --reload` | Start backend |
| `cd frontend && npm run dev` | Start frontend |
| `curl -X POST http://localhost:8000/start_episode` | Trigger episode via CLI |
| `curl http://localhost:8000/status` | Check agent status |
| `curl http://localhost:8000/health` | Liveness check |
| `python training/sft_train.py` | Run SFT (Colab) |
| `python training/grpo_train.py` | Run GRPO (Colab) |

---

## File Reference

```
kaizen_os/
├── environment/
│   ├── action_space.py      ← Pydantic action models + parse_action()
│   ├── observation_space.py ← psutil telemetry + partial observability
│   ├── chaos.py             ← 6 chaos scenarios incl. semantic_decoy
│   ├── reward.py            ← 8-component reward function
│   ├── sandbox.py           ← Safe action executor
│   └── kaizen_env.py        ← Main Gymnasium environment
├── agent/
│   ├── prompts.py           ← System prompt (partial observability aware)
│   └── llm_agent.py         ← LLM inference wrapper
├── training/
│   ├── golden_examples.json ← 62 SFT training examples
│   ├── sft_train.py         ← Unsloth LoRA SFT
│   └── grpo_train.py        ← TRL GRPO + live env reward bridge
├── server/
│   ├── broadcast.py         ← WebSocket manager
│   └── main.py              ← FastAPI app
├── frontend/src/
│   ├── App.jsx              ← Root layout
│   ├── components/          ← All 7 dashboard components
│   └── hooks/useWebSocket.js
├── app.py                   ← HuggingFace Space entry point
├── Dockerfile               ← HF Space Docker image
└── requirements.txt
```

---

Good luck at the hackathon.
The semantic_decoy scenario is your winning demo — two processes at identical
CPU usage, only the log distinguishes them. No rule-based system solves that.
Your agent does.
