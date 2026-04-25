"""
training/grpo_train.py
Group Relative Policy Optimisation (GRPO) training for the Kaizen OS agent.

Run AFTER sft_train.py has produced a LoRA adapter in ./kaizen_sft_model

    !python training/grpo_train.py

Architecture
------------
- Loads the SFT-adapted model from MODEL_PATH
- For each training prompt (system observation), generates GROUP_SIZE
  completions and scores them against the live KaizenEnv reward function
- GRPO uses within-group relative reward as the advantage signal
  (no separate critic network required — this is what makes it memory-efficient)
- Saves the final policy to OUTPUT_DIR

HF Credits upgrade path
-----------------------
Set MODEL_PATH to a larger SFT adapter. GRPO config stays identical.

Memory budget on T4 (16 GB)
----------------------------
Qwen2.5-3B 4-bit:  ~2.5 GB model weights
GROUP_SIZE=4:       ~4 × 256 tokens × 4B = ~0.5 GB activation buffer
Ref model copy:     ~2.5 GB (frozen, 4-bit)
Optimiser (8-bit):  ~0.5 GB
Total estimate:     ~6 GB — safely within 16 GB

Changes vs v1
-------------
- MAX_NEW_TOKENS reduced 512→256 to fix completion clipping (clipped_ratio was 1.0)
- reward_fn signature fixed: TRL passes completions only, not (prompts, completions)
- reward_fn handles both string and message-dict completion formats
- build_prompt_dataset wraps prompts in Alpaca format matching SFT training
- use_vllm=False added to GRPOConfig to avoid Unsloth conflicts
- traceback.print_exc() added to reward_fn error handler for debugging
"""

# unsloth MUST be imported before transformers — do not move this line
import unsloth  # noqa: F401

import json
import os
import sys
import random
import traceback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH      = os.environ.get("KAIZEN_SFT_PATH", "./kaizen_sft_model")
GROUP_SIZE      = 4          # G — number of completions per prompt
MAX_NEW_TOKENS  = 256        # reduced from 512 — fixes clipped_ratio=1.0
KL_COEF         = 0.1
LEARNING_RATE   = 5e-6
MAX_STEPS       = 80
OUTPUT_DIR      = "./kaizen_grpo_model"
BATCH_SIZE      = 1          # 1 prompt at a time; GROUP_SIZE handles rollouts
GRAD_ACCUM      = 4
SEED            = 42

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_examples.json")
REWARD_PLOT_PATH = "./reward_curve.png"

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
# Reward tracking (global so reward_fn can append to it)
# ---------------------------------------------------------------------------

_reward_history: list[float] = []
_step_counter: int = 0


# ---------------------------------------------------------------------------
# Reward function bridge
# ---------------------------------------------------------------------------

# One shared env instance — broadcast=False means no WebSocket during training
_env = KaizenEnv(broadcast=False)


def reward_fn(completions: list, **kwargs) -> list[float]:
    """
    Reward function called by GRPOTrainer for each group of completions.

    TRL passes completions as the only positional argument. The format
    varies by TRL version — either plain strings or lists of message dicts
    (chat format). We handle both.

    Each completion is evaluated against a fresh environment episode.
    The environment automatically injects chaos at step 3, so completions
    that correctly identify and kill the chaos process receive the highest
    rewards (+3.0 chaos resolved + thermal/CPU improvement bonuses).

    Parameters
    ----------
    completions : list — the model completions to score (strings or message dicts)

    Returns
    -------
    list[float] — one reward per completion
    """
    global _step_counter
    rewards: list[float] = []

    for completion in completions:
        try:
            # Handle both string completions and message-dict completions.
            # TRL chat format: [{"role": "assistant", "content": "..."}]
            if isinstance(completion, list):
                completion_text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                completion_text = completion.get("content", str(completion))
            else:
                completion_text = str(completion)

            # Reset env for a fresh episode — defensive unpacking
            reset_result = _env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            # Advance to step 3 where chaos injects automatically.
            # Use a dummy wait action for warm-up steps.
            dummy_action = '{"tool_name": "wait", "reason": "warm-up"}'
            for _ in range(3):
                if _env.is_done:
                    break
                step_result = _env.step(dummy_action)
                obs = step_result[0]

            if _env.is_done:
                rewards.append(0.0)
                continue

            # Evaluate the model's completion against the chaos state.
            # env.step() handles parse_action, sandbox, reward, and chaos
            # resolution internally — index 1 is always the scalar reward.
            step_result = _env.step(completion_text)
            reward = float(step_result[1])
            rewards.append(reward)

        except Exception as exc:
            # Never let a reward computation crash the training loop.
            print(f"[GRPO] reward_fn error: {exc}")
            traceback.print_exc()
            rewards.append(-1.0)

    # Track mean reward for this batch
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        _reward_history.append(mean_reward)
        _step_counter += 1

        if _step_counter % 10 == 0:
            recent = _reward_history[-10:]
            print(
                f"[GRPO] Step {_step_counter:03d} | "
                f"Batch mean reward: {mean_reward:+.3f} | "
                f"10-step avg: {sum(recent)/len(recent):+.3f}"
            )

    return rewards


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_prompt_dataset(path: str) -> Dataset:
    """
    Build a dataset of prompts from golden_examples.json for GRPO.

    GRPO only needs the prompts (the 'instruction' field) — it generates
    its own completions via the policy and scores them with reward_fn.

    Prompts are wrapped in the Alpaca format to match the SFT training
    template exactly. A mismatch here was causing the model to generate
    suboptimal completions in v1.
    """
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Filter to only chaos-active examples (skip the 'wait' healthy examples).
    # GRPO learns more from high-signal reward variance — chaos examples
    # give the model meaningful +/- reward signal to learn from.
    chaos_examples = [
        ex for ex in examples
        if "Active Chaos : None" not in ex["instruction"]
    ]

    print(f"[GRPO] Using {len(chaos_examples)} chaos examples for GRPO prompts")

    # Wrap each prompt in Alpaca format to match SFT training template.
    # Without this, the model receives prompts in a different format than
    # it was trained on, degrading generation quality.
    prompts = []
    for ex in chaos_examples:
        formatted = (
            "Below is an observation of a system under stress. "
            "Analyse it and respond with your reasoning followed by exactly one JSON action.\n\n"
            f"### Observation:\n{ex['instruction']}\n\n### Response:"
        )
        prompts.append({"prompt": formatted})

    # Shuffle for variety across GRPO groups
    random.seed(SEED)
    random.shuffle(prompts)

    return Dataset.from_list(prompts)


# ---------------------------------------------------------------------------
# Reward curve plotting
# ---------------------------------------------------------------------------

def save_reward_plot(history: list[float], path: str) -> None:
    """
    Save a reward curve plot to disk.

    Uses matplotlib if available, otherwise writes a JSON fallback.
    The plot path is printed so it's easy to find in Colab's file browser.
    """
    try:
        import matplotlib                      # type: ignore
        matplotlib.use("Agg")                  # non-interactive backend
        import matplotlib.pyplot as plt        # type: ignore

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history, color="#4ade80", linewidth=1.5, label="Mean batch reward")

        # Running average
        window = min(10, len(history))
        if len(history) >= window:
            running_avg = [
                sum(history[max(0, i - window + 1):i + 1]) / min(window, i + 1)
                for i in range(len(history))
            ]
            ax.plot(running_avg, color="#f59e0b", linewidth=2.0,
                    linestyle="--", label=f"{window}-step running avg")

        ax.set_facecolor("#0f0f0f")
        fig.patch.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#888888")
        ax.spines[:].set_color("#1f1f1f")
        ax.set_xlabel("Training step", color="#888888")
        ax.set_ylabel("Mean group reward", color="#888888")
        ax.set_title("Kaizen OS — GRPO Reward Curve", color="#e2e2e2", pad=12)
        ax.legend(facecolor="#161616", labelcolor="#e2e2e2", framealpha=0.8)
        ax.axhline(y=0, color="#444444", linewidth=0.5, linestyle=":")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[GRPO] Reward curve saved to: {path}")

    except ImportError:
        # matplotlib not available — save raw data as JSON instead
        json_path = path.replace(".png", ".json")
        with open(json_path, "w") as f:
            json.dump({"reward_history": history}, f, indent=2)
        print(f"[GRPO] matplotlib not available. Reward history saved to: {json_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    print("[GRPO] ===== Kaizen OS — GRPO Reinforcement Learning =====")
    print(f"[GRPO] SFT model path : {MODEL_PATH}")
    print(f"[GRPO] Group size     : {GROUP_SIZE}")
    print(f"[GRPO] KL coefficient : {KL_COEF}")
    print(f"[GRPO] Learning rate  : {LEARNING_RATE}")
    print(f"[GRPO] Max steps      : {MAX_STEPS}")
    print(f"[GRPO] Output dir     : {OUTPUT_DIR}")
    print(f"[GRPO] Max new tokens : {MAX_NEW_TOKENS}")

    # Check SFT model exists
    if not os.path.isdir(MODEL_PATH):
        print(f"[GRPO] ERROR: SFT model not found at {MODEL_PATH}")
        print("[GRPO] Run training/sft_train.py first to produce the SFT adapter.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load SFT-adapted model
    # ------------------------------------------------------------------
    print("\n[GRPO] Loading SFT model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Enable training mode (Unsloth requires this before GRPO)
    FastLanguageModel.for_training(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[GRPO] Model loaded and set to training mode.")

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    print("\n[GRPO] Building prompt dataset...")
    dataset = build_prompt_dataset(DATASET_PATH)
    print(f"[GRPO] Dataset size: {len(dataset)} prompts")

    # ------------------------------------------------------------------
    # 3. GRPO configuration
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        # Core GRPO params
        num_generations=GROUP_SIZE,           # completions per prompt (G)
        max_completion_length=MAX_NEW_TOKENS, # 256 — prevents clipped_ratio=1.0
        max_prompt_length=1024,               # cap prompt to save VRAM
        temperature=0.8,
        top_p=0.95,
        # KL penalty — beta is the correct name in current TRL versions
        beta=KL_COEF,
        # Training params
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        seed=SEED,
        # Optimiser
        optim="adamw_8bit",
        weight_decay=0.01,
        # Logging
        logging_steps=5,
        output_dir=OUTPUT_DIR,
        report_to="none",
        # Memory savings
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        # Save
        save_strategy="no",
        # Explicitly disable vllm to avoid Unsloth conflicts
        use_vllm=False,
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
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[GRPO] Starting GRPO training...")
    print("[GRPO] The reward function will run live KaizenEnv episodes.")
    print(f"[GRPO] Each step generates {GROUP_SIZE} completions and scores them.")
    print(f"[GRPO] Reward range: -10.0 (protected kill) to ~+8.0 (full resolution)")
    print()

    if torch.cuda.is_available():
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Starting VRAM: {start_mem} GB")

    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Peak VRAM used: {peak_mem} GB")

    print(f"\n[GRPO] Training complete.")
    print(f"[GRPO] Runtime  : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"[GRPO] Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # ------------------------------------------------------------------
    # 6. Save final policy
    # ------------------------------------------------------------------
    print(f"\n[GRPO] Saving final policy to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[GRPO] Policy saved.")

    # ------------------------------------------------------------------
    # 7. Save reward curve
    # ------------------------------------------------------------------
    if _reward_history:
        print(f"\n[GRPO] Saving reward curve ({len(_reward_history)} data points)...")
        save_reward_plot(_reward_history, REWARD_PLOT_PATH)

        # Print a simple ASCII summary for judges reading Colab output
        if len(_reward_history) >= 2:
            first = _reward_history[0]
            last = _reward_history[-1]
            delta = last - first
            print(f"[GRPO] Reward summary: {first:+.3f} → {last:+.3f} (Δ {delta:+.3f})")
    else:
        print("[GRPO] No reward history recorded — check reward_fn is being called.")

    # ------------------------------------------------------------------
    # 8. Optional: push to HuggingFace Hub
    # ------------------------------------------------------------------
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_repo:
        grpo_repo = hf_repo.rstrip("/") + "-grpo"
        print(f"\n[GRPO] Pushing to HuggingFace Hub: {grpo_repo}")
        model.push_to_hub(grpo_repo)
        tokenizer.push_to_hub(grpo_repo)
        print("[GRPO] Push complete.")

    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_path = train()
    print(f"\n[GRPO] Done. Final model at: {output_path}")
    print(f"[GRPO] Load with: LLMAgent(model_name='{output_path}')")
    print(f"[GRPO] Reward curve: {REWARD_PLOT_PATH}")