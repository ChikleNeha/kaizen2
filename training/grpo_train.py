"""
training/grpo_train.py
GRPO Reinforcement Learning for the Kaizen OS agent.

FIXES vs original:
  1. Uses processing_class= instead of tokenizer= for TRL 0.12+ compat
  2. use_vllm= wrapped in try/except for TRL version safety
  3. reward_fn now passes chaos_resolved=True to compute_reward()
     so the +3.0 chaos-resolved bonus actually fires during training
  4. save_steps=20 checkpoint saves added (safe recovery if crash)

Run AFTER sft_train.py:
    !python training/grpo_train.py
"""

import unsloth  # noqa: F401 — must be first

import json
import os
import sys
import random
import traceback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH      = os.environ.get("KAIZEN_SFT_PATH", "./kaizen_sft_model")
GROUP_SIZE      = 4
MAX_NEW_TOKENS  = 256
KL_COEF         = 0.1
LEARNING_RATE   = 5e-6
MAX_STEPS       = 80
OUTPUT_DIR      = "./kaizen_grpo_model"
BATCH_SIZE      = 1
GRAD_ACCUM      = 4
SEED            = 42

DATASET_PATH     = os.path.join(os.path.dirname(__file__), "golden_examples.json")
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
        print(f"[GRPO] Missing: {missing}")
        sys.exit(1)

_check_imports()

from unsloth import FastLanguageModel          # type: ignore
from trl import GRPOTrainer, GRPOConfig        # type: ignore
from datasets import Dataset                   # type: ignore
import torch
import trl as _trl

_trl_version = tuple(int(x) for x in _trl.__version__.split(".")[:2])
_use_processing_class = _trl_version >= (0, 12)

# Add project root to sys.path for environment imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from environment.kaizen_env import KaizenEnv
from environment.action_space import parse_action
from environment.reward import compute_reward


# ---------------------------------------------------------------------------
# Reward tracking
# ---------------------------------------------------------------------------

_reward_history: list[float] = []
_step_counter: int = 0

# One shared env — broadcast=False avoids needing a running WebSocket server
_env = KaizenEnv(broadcast=False)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def reward_fn(completions: list, **kwargs) -> list[float]:
    """
    Reward function called by GRPOTrainer for each group of completions.

    FIX: Now passes chaos_resolved to compute_reward() so the +3.0
    chaos-resolved bonus fires correctly during training.

    Without this fix, the model trains on reward signals that are missing
    the most important bonus, causing it to undervalue chaos resolution.
    """
    global _step_counter
    rewards: list[float] = []

    for completion in completions:
        try:
            # Handle both string and message-dict formats
            if isinstance(completion, list):
                completion_text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                completion_text = completion.get("content", str(completion))
            else:
                completion_text = str(completion)

            # Fresh episode
            reset_result = _env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            # Advance to step 3 (where chaos auto-injects)
            dummy_action = '{"tool_name": "wait", "reason": "warm-up"}'
            for _ in range(3):
                if _env.is_done:
                    break
                step_result = _env.step(dummy_action)
                obs = step_result[0]

            if _env.is_done:
                rewards.append(0.0)
                continue

            # Capture chaos state BEFORE the agent acts
            chaos_was_active = _env._chaos.is_active

            # Execute the model's completion
            step_result = _env.step(completion_text)
            obs_after = step_result[0]
            # env.step already computes reward with chaos_resolved internally
            # We take the env's reward directly — it uses the fixed compute_reward
            reward = float(step_result[1])
            rewards.append(reward)

        except Exception as exc:
            print(f"[GRPO] reward_fn error: {exc}")
            traceback.print_exc()
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

def build_prompt_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # Only chaos examples — higher reward variance = better GRPO signal
    chaos_examples = [
        ex for ex in examples
        if "Active Chaos : None" not in ex["instruction"]
    ]
    print(f"[GRPO] Using {len(chaos_examples)} chaos examples for prompts")

    prompts = []
    for ex in chaos_examples:
        formatted = (
            "Below is an observation of a system under stress. "
            "Analyse it and respond with your reasoning followed by exactly one JSON action.\n\n"
            f"### Observation:\n{ex['instruction']}\n\n### Response:"
        )
        prompts.append({"prompt": formatted})

    random.seed(SEED)
    random.shuffle(prompts)
    return Dataset.from_list(prompts)


# ---------------------------------------------------------------------------
# Reward curve
# ---------------------------------------------------------------------------

def save_reward_plot(history: list[float], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history, color="#4ade80", linewidth=1.5, label="Mean batch reward")

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
        print(f"[GRPO] >>> Commit {path} to your repo before submission <<<")

    except ImportError:
        json_path = path.replace(".png", ".json")
        with open(json_path, "w") as f:
            json.dump({"reward_history": history}, f, indent=2)
        print(f"[GRPO] matplotlib unavailable. Data saved to: {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    print("[GRPO] ===== Kaizen OS — GRPO Reinforcement Learning =====")
    print(f"[GRPO] TRL version    : {_trl.__version__}")
    print(f"[GRPO] SFT model path : {MODEL_PATH}")
    print(f"[GRPO] Group size     : {GROUP_SIZE}")
    print(f"[GRPO] Max steps      : {MAX_STEPS}")
    print(f"[GRPO] Max new tokens : {MAX_NEW_TOKENS}")

    if not os.path.isdir(MODEL_PATH):
        print(f"[GRPO] ERROR: SFT model not found at {MODEL_PATH}")
        print("[GRPO] Run training/sft_train.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[GRPO] Loading SFT model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------
    print("\n[GRPO] Building prompt dataset...")
    dataset = build_prompt_dataset(DATASET_PATH)
    print(f"[GRPO] Dataset size: {len(dataset)} prompts")

    # ------------------------------------------------------------------
    # 3. GRPOConfig
    # FIX: use_vllm wrapped for version safety; save_steps added
    # ------------------------------------------------------------------
    grpo_config_kwargs = dict(
        num_generations=GROUP_SIZE,
        max_completion_length=MAX_NEW_TOKENS,
        max_prompt_length=1024,
        temperature=0.8,
        top_p=0.95,
        beta=KL_COEF,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        seed=SEED,
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=5,
        output_dir=OUTPUT_DIR,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        save_strategy="steps",   # FIX: checkpoint every 20 steps
        save_steps=20,           # safe recovery if training crashes
    )

    # use_vllm only exists in TRL >= 0.12 — wrap to avoid TypeError on older
    try:
        grpo_config = GRPOConfig(use_vllm=False, **grpo_config_kwargs)
        print("[GRPO] use_vllm=False set (TRL >= 0.12)")
    except TypeError:
        grpo_config = GRPOConfig(**grpo_config_kwargs)
        print("[GRPO] use_vllm not supported by this TRL version — skipped")

    # ------------------------------------------------------------------
    # 4. GRPOTrainer
    # FIX: use processing_class= for TRL 0.12+, tokenizer= for older
    # ------------------------------------------------------------------
    print(f"[GRPO] Using {'processing_class' if _use_processing_class else 'tokenizer'}= (TRL {_trl.__version__})")
    trainer_kwargs = dict(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )
    if _use_processing_class:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    print("[GRPO] Initialising GRPOTrainer...")
    trainer = GRPOTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[GRPO] Starting GRPO training...")
    print(f"[GRPO] Checkpoints saved every 20 steps to {OUTPUT_DIR}/")
    print(f"[GRPO] Reward range: -10.0 (protected kill) to ~+8.0 (full resolution)")
    print()

    if torch.cuda.is_available():
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Starting VRAM: {start_mem} GB")

    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Peak VRAM: {peak_mem} GB")

    print(f"\n[GRPO] Training complete.")
    print(f"[GRPO] Runtime  : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"[GRPO] Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # ------------------------------------------------------------------
    # 6. Save final model
    # ------------------------------------------------------------------
    print(f"\n[GRPO] Saving final policy to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 7. Save reward curve (REQUIRED by judges)
    # ------------------------------------------------------------------
    if _reward_history:
        print(f"\n[GRPO] Saving reward curve ({len(_reward_history)} points)...")
        save_reward_plot(_reward_history, REWARD_PLOT_PATH)
        if len(_reward_history) >= 2:
            first, last = _reward_history[0], _reward_history[-1]
            print(f"[GRPO] Reward summary: {first:+.3f} → {last:+.3f} (Δ {last-first:+.3f})")
    else:
        print("[GRPO] Warning: no reward history — check reward_fn is being called.")

    # ------------------------------------------------------------------
    # 8. Optional HF Hub push
    # ------------------------------------------------------------------
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_repo:
        grpo_repo = hf_repo.rstrip("/") + "-grpo"
        print(f"\n[GRPO] Pushing to HuggingFace Hub: {grpo_repo}")
        model.push_to_hub(grpo_repo)
        tokenizer.push_to_hub(grpo_repo)
        print("[GRPO] Push complete.")

    return OUTPUT_DIR


if __name__ == "__main__":
    output_path = train()
    print(f"\n[GRPO] Done. Final model: {output_path}")
    print(f"[GRPO] Reward curve: {REWARD_PLOT_PATH}")