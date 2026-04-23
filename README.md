# Kaizen OS — The Agentic Kernel

**Project Kaizen** is a reinforcement learning environment where an LLM agent autonomously manages a simulated operating system. The agent monitors real system telemetry, detects chaos events (memory leaks, CPU hogs, thermal spikes), reasons through them using chain-of-thought, and takes corrective actions via structured tool calls.

> **Submission for the Meta × Scaler OpenEnv Hackathon**
> Primary theme: Theme 3.1 — World Modeling (Professional Tasks)
> Secondary theme: Theme 2 — Long-Horizon Planning

---

## Architecture

```
psutil (real telemetry)
        ↓
  KaizenEnv (Gymnasium)
        ↓
  ChaosInjector ──→ ObservationBuilder
        ↓
   LLMAgent (Qwen2.5-3B-Instruct)
        ↓
  parse_action() [Pydantic v2]
        ↓
  SandboxExecutor ──→ compute_reward()
        ↓
  WebSocket broadcast ──→ React Dashboard
```

```
Training pipeline:
  golden_examples.json
        ↓
  sft_train.py (Unsloth + LoRA)
        ↓
  kaizen_sft_model/
        ↓
  grpo_train.py (TRL GRPOTrainer)
        ↓
  kaizen_grpo_model/  +  reward_curve.png
```

---

## Stack

| Layer | Technology |
|---|---|
| Environment | Gymnasium-style, Python, psutil |
| LLM | Qwen2.5-3B-Instruct (4-bit NF4) |
| SFT | Unsloth + LoRA |
| RL | TRL GRPOTrainer |
| Backend | FastAPI + WebSockets |
| Frontend | React 18 + Vite |
| Sandbox | Simulated (Docker upgrade path included) |
| Training | Google Colab T4 (free tier) |
| Demo | HuggingFace Spaces |

---

## Quickstart — Local

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/kaizen-os.git
cd kaizen-os
pip install -r requirements.txt
```

### 2. Start the backend

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts instantly. The LLM loads lazily on the first `/start_episode` call.

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### 4. Run an episode

Click **Start Episode** in the top bar, or:

```bash
curl -X POST http://localhost:8000/start_episode
```

Watch the dashboard — chaos injects at step 3.

---

## Training on Google Colab (Free T4)

### Step 1 — Upload the project

1. Zip the entire `kaizen_os/` folder
2. Upload to your Google Drive
3. Open a new Colab notebook

### Step 2 — Mount Drive and install dependencies

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/kaizen_os

# Install Unsloth (must match your Colab CUDA version)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl datasets accelerate bitsandbytes wandb matplotlib
```

### Step 3 — Run SFT

```python
!python training/sft_train.py
```

Expected output:
```
[SFT] Model        : unsloth/Qwen2.5-3B-Instruct
[SFT] Parameters   : 3.09B total, 39.9M trainable (1.29%)
[SFT] Peak VRAM    : 6.2 GB
[SFT] Final loss   : 0.8142
[SFT] Adapter saved to ./kaizen_sft_model
```

Runtime: ~12 minutes on T4.

### Step 4 — Run GRPO

```python
!python training/grpo_train.py
```

Expected output:
```
[GRPO] Step 010 | Batch mean reward: +1.243 | 10-step avg: +0.821
[GRPO] Step 020 | Batch mean reward: +2.108 | 10-step avg: +1.654
...
[GRPO] Step 080 | Batch mean reward: +4.712 | 10-step avg: +4.231
[GRPO] Reward summary: +0.821 → +4.712 (Δ +3.891)
[GRPO] Reward curve saved to: ./reward_curve.png
```

Runtime: ~25 minutes on T4.

### Step 5 — Push to HuggingFace Hub (optional)

```python
import os
os.environ["HF_REPO_ID"] = "your-username/kaizen-os-grpo"

!python training/grpo_train.py
# Will push automatically at end of training
```

---

## HuggingFace Space Deployment

### Step 1 — Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. SDK: **Docker**
3. Hardware: **CPU Basic** (free) — upgrade to GPU at hackathon evaluation

### Step 2 — Push the repository

```bash
# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/kaizen-os

# Push
git push hf main
```

### Step 3 — Set Space secrets

In your Space settings → Variables and secrets:

```
KAIZEN_MODEL_NAME = Qwen/Qwen2.5-3B-Instruct
KAIZEN_4BIT       = true
PORT              = 7860
```

**At hackathon evaluation with HF credits**, upgrade the model:
```
KAIZEN_MODEL_NAME = Qwen/Qwen2.5-14B-Instruct
```
No other code changes needed.

### Step 4 — Update frontend env

```bash
cd frontend
cp .env.example .env.local
```

Edit `.env.local`:
```
VITE_WS_URL=wss://YOUR_USERNAME-kaizen-os.hf.space/ws
VITE_API_URL=https://YOUR_USERNAME-kaizen-os.hf.space
```

Rebuild:
```bash
npm run build
```

The `dist/` folder can be served from the same FastAPI server by adding:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

---

## ngrok Demo (for local → public URL)

If running locally and need a public URL for judges:

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000
```

Note the `https://xxxx.ngrok.io` URL. Update frontend:
```bash
VITE_WS_URL=wss://xxxx.ngrok.io/ws
VITE_API_URL=https://xxxx.ngrok.io
npm run build
```

Serve the built frontend from FastAPI (add `StaticFiles` mount above).

---

## Docker Sandbox (optional upgrade)

The sandbox runs in simulated mode by default. To enable Docker mode:

```bash
# Build the sandbox image
docker build -f docker/Dockerfile.sandbox -t kaizen-sandbox:latest .

# Run the backend with Docker mode enabled
KAIZEN_USE_DOCKER=true uvicorn server.main:app --port 8000
```

In `server/main.py`, pass `use_docker=True` to `KaizenEnv`.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `KAIZEN_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | HuggingFace model ID or local adapter path |
| `KAIZEN_4BIT` | `true` | Enable 4-bit NF4 quantisation |
| `KAIZEN_SFT_PATH` | `./kaizen_sft_model` | Path to SFT adapter for GRPO training |
| `HF_REPO_ID` | _(empty)_ | HuggingFace repo to push trained adapter |
| `PORT` | `8000` | Backend server port |
| `VITE_WS_URL` | `ws://localhost:8000/ws` | WebSocket URL for frontend |
| `VITE_API_URL` | `http://localhost:8000` | REST API base URL for frontend |

---

## Project Structure

```
kaizen_os/
├── environment/
│   ├── action_space.py      # Pydantic v2 action models + parse_action()
│   ├── observation_space.py # psutil-based observation builder
│   ├── chaos.py             # Chaos event injector (3 scenarios)
│   ├── reward.py            # Reward function (8 components)
│   ├── sandbox.py           # Simulated + Docker sandbox executor
│   └── kaizen_env.py        # Main Gymnasium environment
├── agent/
│   ├── prompts.py           # System prompt + observation formatter
│   └── llm_agent.py         # LLM inference class
├── training/
│   ├── golden_examples.json # 62 hand-crafted SFT examples
│   ├── sft_train.py         # Unsloth LoRA SFT (T4-safe)
│   └── grpo_train.py        # TRL GRPOTrainer + live env reward bridge
├── server/
│   ├── broadcast.py         # WebSocket ConnectionManager
│   └── main.py              # FastAPI app
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Root layout
│   │   ├── main.jsx          # Vite entry
│   │   ├── hooks/
│   │   │   └── useWebSocket.js
│   │   └── components/
│   │       ├── TopBar.jsx
│   │       ├── VitalsPanel.jsx
│   │       ├── ProcessGraph.jsx   # Canvas node graph
│   │       ├── ReasoningPanel.jsx # Typewriter chain-of-thought
│   │       ├── ActionLog.jsx
│   │       ├── RewardTracker.jsx
│   │       └── StepsBar.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
├── docker/
│   └── Dockerfile.sandbox
├── requirements.txt
└── README.md
```

---

## Demo Script (for judges)

1. Open the dashboard — `http://localhost:5173`
2. Confirm idle state — all vitals green, status dot dim
3. Click **Start Episode** in the top bar
4. Watch steps 1–2 — system healthy, agent waits
5. **Step 3**: chaos injects — graph node turns red and pulses, chaos badge appears in StepsBar and TopBar
6. ReasoningPanel typewriters the agent's chain-of-thought analysis
7. Action pill shows: `kill_process  pid=XXXX`
8. Node and edges dissolve in the ProcessGraph
9. Vitals bars animate back to green — thermal drops, CPU recovers
10. RewardTracker shows cumulative improvement
11. Point to the reward curve: *"Episode 1 scored +2.1, Episode 12 scored +7.8 — this is what learning looks like"*

### Judge Q&A

**"Why not use systemd or a watchdog?"**
systemd is binary — up or down. Our agent reads logs, understands process names semantically, and makes priority judgements (Spotify is less important than Zoom). A watchdog timer cannot do that.

**"Why GRPO instead of PPO?"**
GRPO doesn't need a separate critic network. It compares trajectories within a group and uses relative performance as the advantage baseline. This makes it far more memory-efficient for LLM fine-tuning — we run it on a free T4.

**"Are these real system calls?"**
Observations are real-time telemetry from psutil — actual CPU, RAM, and thermal readings from the host machine. Actions execute in a sandboxed executor to prevent any real system modification during the demo.

**"How do you handle hallucinations?"**
Every LLM output passes through a Pydantic v2 validation layer before any action executes. Invalid tool name or malformed JSON → `-1.0` reward immediately. This trains format discipline without human intervention.

---

## Reward Function Summary

| Component | Value |
|---|---|
| Parse failure | −1.0 |
| Protected process kill | −10.0 |
| Thermal improvement | Δtemp × 0.15 |
| CPU improvement | Δcpu × 0.10 |
| Critical process alive | +0.5 per process |
| Chaos resolved | +3.0 |
| System nominal (cpu<40, thermal<70) | +1.0 |
| Unnecessary kill (cpu<5%) | −2.0 |

---

## License

MIT — built for the Meta × Scaler OpenEnv Hackathon.
