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
GROUP_SIZE=4:       ~4 × 512 tokens × 4B = ~1 GB activation buffer
Ref model copy:     ~2.5 GB (frozen, 4-bit)
Optimiser (8-bit):  ~0.5 GB
Total estimate:     ~7 GB — safely within 16 GB
"""

import json
import os
import sys
import random
from transformers import GenerationConfig

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


def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Reward function called by GRPOTrainer for each group of completions.

    Each completion is evaluated against a fresh environment episode.
    The environment automatically injects chaos at step 3, so completions
    that correctly identify and kill the chaos process receive the highest
    rewards (+3.0 chaos resolved + thermal/CPU improvement bonuses).

    Parameters
    ----------
    prompts     : list[str] — the input prompts (one per completion in the group)
    completions : list[str] — the model completions to score

    Returns
    -------
    list[float] — one reward per completion
    """
    global _step_counter
    rewards: list[float] = []

    for prompt, completion in zip(prompts, completions):
        try:
            # Reset env for a fresh episode
            obs, _ = _env.reset()

            # Advance to step 3 where chaos injects automatically.
            # Use a dummy wait action for warm-up steps.
            dummy_action = '{"tool_name": "wait", "reason": "warm-up"}'
            for _ in range(3):
                if _env.is_done:
                    break
                obs, _, terminated, truncated, _ = _env.step(dummy_action)

            if _env.is_done:
                rewards.append(0.0)
                continue

            # Now evaluate the model's completion against the chaos state.
            # env.step() handles parse_action, sandbox, reward, and chaos
            # resolution internally — we just read back the scalar reward.
            _, reward, _, _, _ = _env.step(completion)
            rewards.append(float(reward))

        except Exception as exc:
            # Never let a reward computation crash the training loop.
            print(f"[GRPO] reward_fn error: {exc}")
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
    We format each prompt in the chat template format so it matches what
    the model was trained on during SFT.
    """
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Filter to only chaos-active examples (skip the 'wait' healthy examples)
    # GRPO learns more from high-signal reward variance
    chaos_examples = [
        ex for ex in examples
        if "Active Chaos : None" not in ex["instruction"]
    ]

    print(f"[GRPO] Using {len(chaos_examples)} chaos examples for GRPO prompts")

    # Format as chat messages matching the model's expected input format
    # We use the raw instruction text; the tokeniser's chat template wraps it
    prompts = []
    for ex in chaos_examples:
        prompts.append({
            "prompt": ex["instruction"],
        })

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
        num_generations=GROUP_SIZE,     # completions per prompt (G)
        # KL divergence penalty to prevent reward hacking
        kl_coef=KL_COEF,
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
    )

    # Generation config — separate from GRPOConfig
    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True,
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
        generation_config=generation_config,   # <-- pass here
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
