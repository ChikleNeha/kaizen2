"""
training/grpo_train.py
Group Relative Policy Optimisation (GRPO) training for the Kaizen OS agent.

Run AFTER sft_train.py has produced a LoRA adapter in ./kaizen_sft_model

    !python training/grpo_train.py

CHECKPOINT & CONTINUAL IMPROVEMENT
------------------------------------
On first run  : trains from the SFT adapter at MODEL_PATH.
On second run : detects the existing GRPO policy in OUTPUT_DIR and loads it
                instead of the SFT adapter, so each run continues improving
                the same policy rather than restarting.

Checkpoints are saved every SAVE_STEPS steps inside OUTPUT_DIR/checkpoints/.
If Colab crashes mid-run the trainer resumes from the latest checkpoint.

To force a fresh GRPO run from the SFT adapter again:
    rm -rf ./kaizen_grpo_model

Plot images
-----------
Two PNG files are saved at the end of every run:
  - grpo_reward_curve_runN.png   — reward per step for this run
  - grpo_combined_runs.png       — all past runs overlaid on one chart

Run history is persisted in OUTPUT_DIR/run_history.json so the combined
chart stays accurate across Colab sessions.

Architecture
------------
- Loads the SFT/GRPO-adapted model from MODEL_PATH (or OUTPUT_DIR if exists)
- For each training prompt (system observation), generates GROUP_SIZE
  completions and scores them against the live KaizenEnv reward function
- GRPO uses within-group relative reward as the advantage signal
  (no separate critic network required — memory-efficient for T4)

Memory budget on T4 (16 GB)
----------------------------
Qwen2.5-3B 4-bit:  ~2.5 GB model weights
GROUP_SIZE=4:       ~4 × 512 tokens × 4B = ~1 GB activation buffer
Ref model copy:     ~2.5 GB (frozen, 4-bit)
Optimiser (8-bit):  ~0.5 GB
Total estimate:     ~7 GB — safely within 16 GB
"""

import json
import os
import sys
import random
import glob
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH      = os.environ.get("KAIZEN_SFT_PATH", "./kaizen_sft_model")
GROUP_SIZE      = 4          # G — number of completions per prompt
MAX_NEW_TOKENS  = 512
KL_COEF         = 0.1
LEARNING_RATE   = 5e-6
MAX_STEPS       = 80
OUTPUT_DIR      = "./kaizen_grpo_model"
BATCH_SIZE      = 1
GRAD_ACCUM      = 4
SEED            = 42

# Checkpoint settings
SAVE_STEPS       = 20        # save a checkpoint every N steps
SAVE_TOTAL_LIMIT = 4         # keep only the last N checkpoints

CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
RUN_HISTORY_PATH = os.path.join(OUTPUT_DIR, "run_history.json")

DATASET_PATH     = os.path.join(os.path.dirname(__file__), "golden_examples.json")

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _check_imports():
    missing = []
    for pkg in ["unsloth", "trl", "datasets", "transformers", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[GRPO] Missing packages: {missing}")
        print("[GRPO] Install with: pip install unsloth trl datasets accelerate bitsandbytes")
        sys.exit(1)

_check_imports()

from unsloth import FastLanguageModel          # type: ignore
from trl import GRPOTrainer, GRPOConfig        # type: ignore
from datasets import Dataset                   # type: ignore
import torch

# Add project root to path so environment imports work from Colab
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from environment.kaizen_env import KaizenEnv
from environment.action_space import parse_action

# ---------------------------------------------------------------------------
# Run history helpers
# ---------------------------------------------------------------------------

def _load_run_history() -> list[dict]:
    if os.path.isfile(RUN_HISTORY_PATH):
        try:
            with open(RUN_HISTORY_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_run_history(history: list[dict]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RUN_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Checkpoint detection
# ---------------------------------------------------------------------------

def _find_latest_checkpoint() -> str | None:
    """
    Return path to the latest HuggingFace checkpoint directory inside
    CHECKPOINT_DIR, or None if no checkpoints exist.
    """
    pattern = os.path.join(CHECKPOINT_DIR, "checkpoint-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _step(p: str) -> int:
        try:
            return int(os.path.basename(p).split("-")[-1])
        except ValueError:
            return -1

    candidates.sort(key=_step)
    return candidates[-1]


def _grpo_policy_exists() -> bool:
    """True if a trained GRPO policy already lives in OUTPUT_DIR."""
    return os.path.isfile(os.path.join(OUTPUT_DIR, "adapter_config.json"))


# ---------------------------------------------------------------------------
# Reward tracking (module-level so reward_fn can append to it)
# ---------------------------------------------------------------------------

_reward_history: list[float] = []
_step_counter: int = 0

# One shared env instance — broadcast=False means no WebSocket during training
_env = KaizenEnv(broadcast=False)


# ---------------------------------------------------------------------------
# Reward function bridge
# ---------------------------------------------------------------------------

def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward function called by GRPOTrainer for each group of completions.

    Each completion is evaluated against a fresh environment episode.
    The environment automatically injects chaos at step 3, so completions
    that correctly identify and kill the chaos process receive the highest
    rewards (+3.0 chaos resolved + thermal/CPU improvement bonuses).
    """
    global _step_counter
    rewards: list[float] = []

    for prompt, completion in zip(prompts, completions):
        try:
            obs, _ = _env.reset()

            # Warm up to step 3 where chaos auto-injects
            dummy_action = '{"tool_name": "wait", "reason": "warm-up step"}'
            for _ in range(3):
                if not _env.is_done:
                    obs, _, terminated, truncated, _ = _env.step(dummy_action)
                    if terminated or truncated:
                        break

            if _env.is_done:
                rewards.append(0.0)
                continue

            obs_after, reward, terminated, truncated, info = _env.step(completion)
            rewards.append(float(reward))

        except Exception as exc:
            print(f"[GRPO] reward_fn error: {exc}")
            rewards.append(-1.0)

    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        _reward_history.append(mean_reward)
        _step_counter += 1

        if _step_counter % 10 == 0:
            recent = _reward_history[-10:]
            print(
                f"[GRPO] Step {_step_counter:03d} | "
                f"Batch mean: {mean_reward:+.3f} | "
                f"10-step avg: {sum(recent)/len(recent):+.3f}"
            )

    return rewards


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_prompt_dataset(path: str, seed: int) -> Dataset:
    """
    Build a dataset of prompts from golden_examples.json for GRPO.

    GRPO only needs the prompts — it generates its own completions via the
    policy and scores them with reward_fn.
    """
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Prefer chaos-active examples for higher reward variance signal
    chaos_examples = [
        ex for ex in examples
        if "Active Chaos : None" not in ex["instruction"]
    ]
    print(f"[GRPO] Using {len(chaos_examples)} chaos examples as GRPO prompts")

    prompts = [{"prompt": ex["instruction"]} for ex in chaos_examples]

    random.seed(seed)
    random.shuffle(prompts)

    return Dataset.from_list(prompts)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, fig) -> None:
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#1f1f1f")
    ax.grid(color="#1f1f1f", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="#444444", linewidth=0.5, linestyle=":")


def _save_reward_plots(
    reward_values: list[float],
    run_idx: int,
    run_history: list[dict],
) -> None:
    """
    Save two PNG files:
      1. grpo_reward_curve_run<N>.png  — current run
      2. grpo_combined_runs.png        — all past runs overlaid
    """
    try:
        import matplotlib                      # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt        # type: ignore

        colors = ["#4ade80", "#f59e0b", "#818cf8", "#ef4444",
                  "#22d3ee", "#fb923c", "#a3e635", "#e879f9"]

        steps = list(range(1, len(reward_values) + 1))

        # ---- Plot 1: current run ----------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, reward_values, color=colors[run_idx % len(colors)],
                linewidth=1.5, label=f"Run {run_idx + 1} mean reward")

        # Running average overlay
        window = min(10, len(reward_values))
        if len(reward_values) >= window:
            running_avg = [
                sum(reward_values[max(0, i - window + 1):i + 1]) / min(window, i + 1)
                for i in range(len(reward_values))
            ]
            ax.plot(steps, running_avg, color="#f59e0b", linewidth=2.0,
                    linestyle="--", label=f"{window}-step running avg")

        _style_ax(ax, fig)
        ax.set_xlabel("Training step", color="#888888")
        ax.set_ylabel("Mean group reward", color="#888888")
        ax.set_title(f"Kaizen OS GRPO — Reward Curve (Run {run_idx + 1})",
                     color="#e2e2e2", pad=12)
        ax.legend(facecolor="#161616", labelcolor="#e2e2e2", framealpha=0.8)

        single_path = os.path.join(OUTPUT_DIR, f"grpo_reward_curve_run{run_idx + 1}.png")
        plt.tight_layout()
        plt.savefig(single_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[GRPO] Reward curve saved: {single_path}")

        # ---- Plot 2: combined runs ---------------------------------------
        if len(run_history) > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 5))

            for i, record in enumerate(run_history):
                c = colors[i % len(colors)]
                r_steps = list(range(1, len(record["rewards"]) + 1))
                final_r = record["final_reward"]
                ax2.plot(r_steps, record["rewards"],
                         color=c, linewidth=1.5, alpha=0.85,
                         label=f"Run {i+1} (final {final_r:+.3f})")

            _style_ax(ax2, fig2)
            ax2.set_xlabel("Training step", color="#888888")
            ax2.set_ylabel("Mean group reward", color="#888888")
            ax2.set_title("Kaizen OS GRPO — All Runs Combined",
                          color="#e2e2e2", pad=12)
            ax2.legend(facecolor="#161616", labelcolor="#e2e2e2",
                       framealpha=0.8, fontsize=8)

            combined_path = os.path.join(OUTPUT_DIR, "grpo_combined_runs.png")
            plt.tight_layout()
            plt.savefig(combined_path, dpi=150, bbox_inches="tight",
                        facecolor=fig2.get_facecolor())
            plt.close()
            print(f"[GRPO] Combined runs plot saved: {combined_path}")

    except ImportError:
        print("[GRPO] matplotlib not available — saving reward history as JSON.")
        fallback = os.path.join(OUTPUT_DIR, f"grpo_rewards_run{run_idx + 1}.json")
        with open(fallback, "w") as f:
            json.dump({"rewards": reward_values}, f, indent=2)
        print(f"[GRPO] Reward data saved: {fallback}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    run_history = _load_run_history()
    run_idx     = len(run_history)

    latest_checkpoint = _find_latest_checkpoint()
    is_resuming       = latest_checkpoint is not None
    is_improvement    = _grpo_policy_exists()

    # Determine which model to load
    if is_improvement:
        source = OUTPUT_DIR
        mode_label = "CONTINUAL IMPROVEMENT (loading existing GRPO policy)"
    elif is_resuming:
        source = MODEL_PATH     # base model; HF trainer will load checkpoint weights
        mode_label = f"RESUME from checkpoint {latest_checkpoint}"
    else:
        source = MODEL_PATH
        mode_label = f"FRESH training from SFT adapter at {MODEL_PATH}"

    print("[GRPO] ===== Kaizen OS — GRPO Reinforcement Learning =====")
    print(f"[GRPO] Run number      : {run_idx + 1}")
    print(f"[GRPO] Mode            : {mode_label}")
    print(f"[GRPO] Source model    : {source}")
    print(f"[GRPO] Group size      : {GROUP_SIZE}")
    print(f"[GRPO] KL coefficient  : {KL_COEF}")
    print(f"[GRPO] Learning rate   : {LEARNING_RATE}")
    print(f"[GRPO] Max steps       : {MAX_STEPS}")
    print(f"[GRPO] Checkpoint every: {SAVE_STEPS} steps → {CHECKPOINT_DIR}")
    print(f"[GRPO] Output dir      : {OUTPUT_DIR}")

    # Check source model exists
    if not os.path.isdir(source):
        print(f"[GRPO] ERROR: source model not found at {source}")
        if source == MODEL_PATH:
            print("[GRPO] Run training/sft_train.py first to produce the SFT adapter.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\n[GRPO] Loading model from {source}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_training(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[GRPO] Model loaded and set to training mode.")

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    print("\n[GRPO] Building prompt dataset...")
    dataset = build_prompt_dataset(DATASET_PATH, seed=SEED + run_idx)
    print(f"[GRPO] Dataset size: {len(dataset)} prompts")

    # ------------------------------------------------------------------
    # 3. GRPO configuration
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        num_generations=GROUP_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        kl_coef=KL_COEF,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        seed=SEED + run_idx,            # vary seed so each run explores differently
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=5,
        output_dir=CHECKPOINT_DIR,      # checkpoints go here
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
    )

    # ------------------------------------------------------------------
    # 4. GRPOTrainer
    # ------------------------------------------------------------------
    print("[GRPO] Initialising GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # ------------------------------------------------------------------
    # 5. Train (resume from checkpoint if one exists)
    # ------------------------------------------------------------------
    print("\n[GRPO] Starting GRPO training...")
    print(f"[GRPO] Each step generates {GROUP_SIZE} completions and scores them.")
    print(f"[GRPO] Reward range: -10.0 (protected kill) to ~+8.0 (full resolution)")
    print()

    if torch.cuda.is_available():
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Starting VRAM: {start_mem} GB")

    t0 = time.time()

    trainer_stats = trainer.train(
        resume_from_checkpoint=latest_checkpoint   # None on fresh run
    )

    elapsed = time.time() - t0

    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Peak VRAM used: {peak_mem} GB")

    final_loss = trainer_stats.metrics.get("train_loss", 0.0)
    print(f"\n[GRPO] Training complete.")
    print(f"[GRPO] Runtime   : {elapsed:.0f}s")
    print(f"[GRPO] Final loss: {final_loss:.4f}")

    # ------------------------------------------------------------------
    # 6. Save final policy (overwrites previous run — intentional)
    # ------------------------------------------------------------------
    print(f"\n[GRPO] Saving final policy to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[GRPO] Policy saved.")

    # ------------------------------------------------------------------
    # 7. Persist run record and save plots
    # ------------------------------------------------------------------
    final_reward = _reward_history[-1] if _reward_history else 0.0

    run_record = {
        "run": run_idx + 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rewards": list(_reward_history),
        "final_reward": round(final_reward, 4),
        "final_loss": round(final_loss, 6),
        "runtime_s": round(elapsed, 1),
        "resumed_from": latest_checkpoint,
        "continual": is_improvement,
        "source_model": source,
    }
    run_history.append(run_record)
    _save_run_history(run_history)

    if _reward_history:
        _save_reward_plots(_reward_history, run_idx, run_history)

        # ASCII reward summary for judges reading Colab output
        if len(_reward_history) >= 2:
            first = _reward_history[0]
            last  = _reward_history[-1]
            delta = last - first
            print(f"\n[GRPO] Reward trajectory: {first:+.3f} → {last:+.3f} (Δ {delta:+.3f})")
    else:
        print("[GRPO] No reward history recorded — check reward_fn is being called.")

    # Cross-run improvement summary
    if len(run_history) > 1:
        prev_reward = run_history[-2]["final_reward"]
        delta       = final_reward - prev_reward
        direction   = "↑ improved" if delta > 0 else "↓ regressed"
        print(f"[GRPO] vs previous run: {prev_reward:+.4f} → {final_reward:+.4f} ({delta:+.4f}) {direction}")

    print(f"\n[GRPO] Run {run_idx + 1} complete.")
    print(f"[GRPO] Total runs so far : {len(run_history)}")
    print(f"[GRPO] To improve further: just run this script again — it loads {OUTPUT_DIR}")
    print(f"[GRPO] Load with: LLMAgent(model_name='{OUTPUT_DIR}')")

    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_path = train()
    print(f"\n[GRPO] Done. Final model at: {output_path}")