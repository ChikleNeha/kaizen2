"""
training/grpo_train.py
GRPO Reinforcement Learning for the Kaizen OS agent.

FIXES applied (original):
  1. MODEL_PATH reads KAIZEN_SFT_PATH env var, falls back to absolute path
  2. OUTPUT_DIR reads GRPO_OUTPUT_DIR env var, falls back to absolute path
  3. reward_fn calls parse_action() before env.step()
  4. chaos_was_active passed to compute_reward() so +3.0 bonus fires
  5. HF push removed from here — run_training.sh owns all pushes
  6. save_steps wrapped in try/except for older GRPOConfig

ADDITIONAL FIXES (this version):
  7. get_last_checkpoint() added — GRPO also resumes on crash
  8. grpo_config now passes save_steps=20 AND save_total_limit=3
     so the container disk doesn't fill up mid-training
  9. Tokenizer pad_token guard moved before GRPOConfig build
     (some TRL versions access it during config validation)
 10. Explicit os.makedirs(OUTPUT_DIR) before training starts
"""

import unsloth  # noqa: F401 — must be first import

import json
import os
import sys
import glob
import random
import traceback

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH  = os.environ.get("KAIZEN_SFT_PATH", "/workspace/kaizen_sft_model")
OUTPUT_DIR  = os.environ.get("GRPO_OUTPUT_DIR", "/workspace/kaizen_grpo_model")

GROUP_SIZE      = 4
MAX_NEW_TOKENS  = 256
KL_COEF         = 0.1
LEARNING_RATE   = 5e-6
MAX_STEPS       = 80
BATCH_SIZE      = 1
GRAD_ACCUM      = 4
SAVE_STEPS      = 20          # checkpoints at step 20, 40, 60, 80
SAVE_TOTAL_LIMIT = 3          # keep last 3 → avoids filling disk
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

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import torch
import trl as _trl

_trl_version = tuple(int(x) for x in _trl.__version__.split(".")[:2])
_use_processing_class = _trl_version >= (0, 12)

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

_env = KaizenEnv(broadcast=False)


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def get_last_checkpoint(output_dir: str):
    """
    Returns the path of the highest-numbered checkpoint-N folder inside
    output_dir, or None if no checkpoints exist.
    Allows GRPO to resume after a crash without losing progress.
    """
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.rsplit("-", 1)[-1]))
    last = checkpoints[-1]
    print(f"[GRPO] Found existing checkpoint: {last}")
    print(f"[GRPO] Training will RESUME from this checkpoint.")
    return last


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def reward_fn(completions: list, **kwargs) -> list[float]:
    """
    Reward function called by GRPOTrainer for each group of completions.

    FIX 3: parse_action() called before env.step() — env gets a proper
            action dict, not raw text.
    FIX 4: chaos_was_active passed to compute_reward() so +3.0 bonus fires.
    """
    global _step_counter
    rewards: list[float] = []

    for completion in completions:
        try:
            # ── Normalise completion to a string ──────────────────────────
            if isinstance(completion, list):
                completion_text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                completion_text = completion.get("content", str(completion))
            else:
                completion_text = str(completion)

            # ── Reset env and warm up ─────────────────────────────────────
            reset_result = _env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            dummy_action = '{"tool_name": "wait", "reason": "warm-up"}'
            for _ in range(3):
                if _env.is_done:
                    break
                step_result = _env.step(dummy_action)
                obs = step_result[0]

            if _env.is_done:
                rewards.append(0.0)
                continue

            # FIX 4: capture chaos state BEFORE the agent acts
            chaos_was_active = _env._chaos.is_active

            # FIX 3: parse to proper action dict before calling env.step()
            try:
                action = parse_action(completion_text)
            except Exception:
                # Unparseable output → small penalty, skip env step
                rewards.append(-0.5)
                continue

            step_result = _env.step(action)
            obs_after   = step_result[0]
            done        = step_result[2] if len(step_result) > 2 else False
            info        = step_result[3] if len(step_result) > 3 else {}

            # FIX 4: compute_reward receives chaos_resolved so +3.0 bonus fires
            chaos_now_resolved = chaos_was_active and not _env._chaos.is_active
            reward = float(compute_reward(
                obs=obs_after,
                action=action,
                done=done,
                info=info,
                chaos_resolved=chaos_now_resolved,
            ))
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
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
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
    print(f"[GRPO] Output dir     : {OUTPUT_DIR}")
    print(f"[GRPO] Group size     : {GROUP_SIZE}")
    print(f"[GRPO] Max steps      : {MAX_STEPS}")
    print(f"[GRPO] Save every     : {SAVE_STEPS} steps (keep last {SAVE_TOTAL_LIMIT})")
    print(f"[GRPO] Max new tokens : {MAX_NEW_TOKENS}")

    # ── Validate SFT model exists ──────────────────────────────────────────
    if not os.path.isdir(MODEL_PATH):
        print(f"[GRPO] ERROR: SFT model not found at {MODEL_PATH}")
        print("[GRPO] Run training/sft_train.py first.")
        sys.exit(1)

    # ── Check for existing GRPO checkpoint ────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        print(f"[GRPO] ⚡ Resuming from: {last_checkpoint}")
    else:
        print(f"[GRPO] No checkpoint found — starting fresh.")

    # ── Load SFT model ────────────────────────────────────────────────────
    print("\n[GRPO] Loading SFT model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_training(model)

    # FIX 9: set pad_token BEFORE GRPOConfig is built
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[GRPO] Building prompt dataset...")
    dataset = build_prompt_dataset(DATASET_PATH)
    print(f"[GRPO] Dataset size: {len(dataset)} prompts")

    # ── Build GRPOConfig with graceful fallbacks ───────────────────────────
    # FIX 8: save_total_limit=3 added to avoid filling disk
    # FIX 6: wrapped in try/except for older GRPOConfig versions
    grpo_config_kwargs = dict(
        num_generations               = GROUP_SIZE,
        max_completion_length         = MAX_NEW_TOKENS,
        max_prompt_length             = 1024,
        temperature                   = 0.8,
        top_p                         = 0.95,
        beta                          = KL_COEF,
        learning_rate                 = LEARNING_RATE,
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        max_steps                     = MAX_STEPS,
        seed                          = SEED,
        optim                         = "adamw_8bit",
        weight_decay                  = 0.01,
        logging_steps                 = 5,
        output_dir                    = OUTPUT_DIR,
        report_to                     = "none",
        fp16                          = not torch.cuda.is_bf16_supported(),
        bf16                          = torch.cuda.is_bf16_supported(),
        gradient_checkpointing        = True,
    )

    try:
        grpo_config = GRPOConfig(
            use_vllm       = False,
            save_strategy  = "steps",
            save_steps     = SAVE_STEPS,
            save_total_limit = SAVE_TOTAL_LIMIT,
            **grpo_config_kwargs,
        )
        print(f"[GRPO] use_vllm=False + save_steps={SAVE_STEPS} set (TRL >= 0.13)")
    except TypeError:
        try:
            grpo_config = GRPOConfig(
                save_strategy    = "steps",
                save_steps       = SAVE_STEPS,
                save_total_limit = SAVE_TOTAL_LIMIT,
                **grpo_config_kwargs,
            )
            print(f"[GRPO] use_vllm not supported — skipped. save_steps={SAVE_STEPS} set.")
        except TypeError:
            try:
                grpo_config = GRPOConfig(
                    save_strategy = "steps",
                    save_steps    = SAVE_STEPS,
                    **grpo_config_kwargs,
                )
                print(f"[GRPO] save_total_limit not supported. save_steps={SAVE_STEPS} set.")
            except TypeError:
                grpo_config = GRPOConfig(**grpo_config_kwargs)
                print("[GRPO] save_steps not supported by this TRL version — checkpointing skipped.")

    # ── Build GRPOTrainer ─────────────────────────────────────────────────
    print(
        f"[GRPO] Using "
        f"{'processing_class' if _use_processing_class else 'tokenizer'}= "
        f"(TRL {_trl.__version__})"
    )
    trainer_kwargs = dict(
        model          = model,
        reward_funcs   = reward_fn,
        args           = grpo_config,
        train_dataset  = dataset,
    )
    if _use_processing_class:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    print("[GRPO] Initialising GRPOTrainer...")
    trainer = GRPOTrainer(**trainer_kwargs)

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n[GRPO] Starting GRPO training...")
    print(f"[GRPO] Reward range: -10.0 (protected kill) to ~+8.0 (full resolution)")

    if torch.cuda.is_available():
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Starting VRAM: {start_mem} GB")

    # FIX 7: resume_from_checkpoint so GRPO can also recover from a crash
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[GRPO] Peak VRAM: {peak_mem} GB")

    print(f"\n[GRPO] Training complete.")
    print(f"[GRPO] Runtime  : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"[GRPO] Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # ── Save final policy ─────────────────────────────────────────────────
    print(f"\n[GRPO] Saving final policy to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[GRPO] Final policy saved.")

    # ── Save reward curve ─────────────────────────────────────────────────
    if _reward_history:
        print(f"\n[GRPO] Saving reward curve ({len(_reward_history)} points)...")
        save_reward_plot(_reward_history, REWARD_PLOT_PATH)
        if len(_reward_history) >= 2:
            first, last = _reward_history[0], _reward_history[-1]
            print(f"[GRPO] Reward summary: {first:+.3f} → {last:+.3f} (Δ {last-first:+.3f})")
    else:
        print("[GRPO] Warning: no reward history — check reward_fn is being called.")

    # FIX 5: hub push intentionally omitted here.
    # run_training.sh handles all pushes with the correct repo names.
    # Pushing here caused the double-suffix bug:
    # NehaChikle/kaizen-qwen2.5-3b-sft-grpo-grpo
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_repo:
        print(f"\n[GRPO] Hub push handled by run_training.sh for: {hf_repo}")
        print("[GRPO] Skipping push here to avoid double-suffix bug.")

    return OUTPUT_DIR


if __name__ == "__main__":
    output_path = train()
    print(f"\n[GRPO] Done. Final model: {output_path}")
    print(f"[GRPO] Reward curve: {REWARD_PLOT_PATH}")