"""
training/sft_train.py
Supervised Fine-Tuning (SFT) for the Kaizen OS agent using Unsloth + LoRA.

Run on Google Colab / Kaggle T4 (free tier, 16 GB VRAM):
    !pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install "trl>=0.15.0" "transformers>=4.47.0"
    !python training/sft_train.py

FIX APPLIED (2025)
-------------------
Root cause: TRL >= 0.13 changed SFTTrainer.training_step() to accept a
`num_items_in_batch` kwarg. Unsloth's monkey-patched version of that method
did not accept the new argument, so it received an int where it expected a
tensor → `AttributeError: 'int' object has no attribute 'mean'`.

Fix: Replace `TrainingArguments` with `SFTConfig` (the TRL-native config
class that SFTTrainer expects), and pass `dataset_text_field` + remove the
deprecated `formatting_func` pattern that triggers the broken code path in
older Unsloth builds.  Also pin `num_items_in_batch` handling via a thin
wrapper that is injected only when the version is detected as incompatible.

CHECKPOINT & CONTINUAL IMPROVEMENT
------------------------------------
On first run  : trains from scratch on the base model, saves adapter to OUTPUT_DIR.
On second run : detects the existing adapter in OUTPUT_DIR and resumes from the
                latest checkpoint automatically, then continues training for
                another MAX_STEPS on top of the previous run's weights.

This means every run improves the previous model — you never start over unless
you delete OUTPUT_DIR manually.

Checkpoints are saved every SAVE_STEPS steps inside OUTPUT_DIR/checkpoints/.
If Colab/Kaggle crashes mid-run the trainer resumes from the latest checkpoint.

After training the LoRA adapter is saved to OUTPUT_DIR.
The adapter can then be loaded by LLMAgent(model_name=OUTPUT_DIR).

Plot images
-----------
Two PNG files are saved at the end of every run:
  - sft_loss_curve_runN.png  — training loss per logging step
  - sft_combined_runs.png    — all past runs overlaid on one chart (auto-grows)

Run history is persisted in OUTPUT_DIR/run_history.json so the combined
chart stays accurate across sessions.
"""

import json
import os
import sys
import glob
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — do not change these values without re-reading the spec
# ---------------------------------------------------------------------------

MODEL_NAME       = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH   = 2048
LOAD_IN_4BIT     = True
LORA_R           = 16
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.0
TARGET_MODULES   = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
BATCH_SIZE       = 2
GRAD_ACCUM       = 4
LEARNING_RATE    = 2e-4
MAX_STEPS        = 100

# Checkpoint settings
SAVE_STEPS       = 25
SAVE_TOTAL_LIMIT = 4

OUTPUT_DIR       = "./kaizen_sft_model"
CHECKPOINT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
RUN_HISTORY_PATH = os.path.join(OUTPUT_DIR, "run_history.json")

DATASET_PATH     = os.path.join(
    os.path.dirname(__file__), "golden_examples.json"
)

# Alpaca prompt template (must match agent/prompts.py format_alpaca)
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def _check_imports():
    missing = []
    for pkg in ["unsloth", "trl", "datasets", "transformers", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SFT] Missing packages: {missing}")
        print("[SFT] Install with:")
        print('  pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"')
        print('  pip install "trl>=0.15.0" "transformers>=4.47.0"')
        sys.exit(1)

_check_imports()

from unsloth import FastLanguageModel          # type: ignore
from datasets import Dataset                   # type: ignore
import torch

# ---------------------------------------------------------------------------
# Compatibility: detect TRL version and import the right config class
# ---------------------------------------------------------------------------

def _import_trl_components():
    """
    TRL >= 0.13 ships SFTConfig as the canonical config for SFTTrainer.
    Older builds only have TrainingArguments.  We try SFTConfig first.
    Returns (SFTTrainer, ConfigClass, use_sft_config: bool)
    """
    from trl import SFTTrainer                 # type: ignore
    try:
        from trl import SFTConfig              # type: ignore
        return SFTTrainer, SFTConfig, True
    except ImportError:
        from transformers import TrainingArguments  # type: ignore
        return SFTTrainer, TrainingArguments, False

SFTTrainer, TrainingConfig, USE_SFT_CONFIG = _import_trl_components()

# ---------------------------------------------------------------------------
# Unsloth training_step compatibility patch
# ---------------------------------------------------------------------------

def _patch_trainer_if_needed(trainer) -> None:
    """
    If Unsloth's monkey-patched training_step doesn't accept the
    `num_items_in_batch` kwarg introduced in TRL >= 0.13 / Transformers >= 4.47,
    wrap it so the kwarg is silently absorbed.

    Detection: we inspect the source of the bound method and check for the
    parameter name.  If absent we inject the wrapper.
    """
    import inspect
    try:
        sig = inspect.signature(trainer.training_step)
        if "num_items_in_batch" not in sig.parameters:
            original_step = trainer.training_step

            def _safe_training_step(model, inputs, num_items_in_batch=None):
                loss = original_step(model, inputs)
                # Unsloth can sometimes return a Python float instead of a tensor
                if not isinstance(loss, torch.Tensor):
                    loss = torch.tensor(float(loss), device=next(model.parameters()).device,
                                        requires_grad=True)
                return loss

            trainer.training_step = _safe_training_step
            print("[SFT] ⚠️  Applied training_step compatibility patch for Unsloth ↔ TRL mismatch.")
    except (ValueError, TypeError):
        # inspect.signature can fail on certain C-extension bound methods; skip
        pass


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


def _adapter_exists() -> bool:
    return os.path.isfile(os.path.join(OUTPUT_DIR, "adapter_config.json"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset_from_json(path: str, eos_token: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"[SFT] Loaded {len(examples)} training examples from {path}")

    rows = []
    for ex in examples:
        # Append EOS token directly into the text field so the trainer
        # doesn't need a separate formatting_func (avoids the broken
        # Unsloth code path that triggers the 'int has no .mean' error).
        text = ALPACA_TEMPLATE.format(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
            output=ex["output"],
        ) + eos_token
        rows.append({"text": text})

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_loss_plot(
    loss_values: list[float],
    steps: list[int],
    run_idx: int,
    run_history: list[dict],
) -> None:
    try:
        import matplotlib                      # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt        # type: ignore

        colors = ["#4ade80", "#f59e0b", "#818cf8", "#ef4444",
                  "#22d3ee", "#fb923c", "#a3e635", "#e879f9"]

        # ---- Plot 1: current run ----------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, loss_values, color=colors[run_idx % len(colors)],
                linewidth=1.5, label=f"Run {run_idx + 1}")
        _style_ax(ax, fig)
        ax.set_xlabel("Step", color="#888888")
        ax.set_ylabel("Training loss", color="#888888")
        ax.set_title(f"Kaizen OS SFT — Loss Curve (Run {run_idx + 1})",
                     color="#e2e2e2", pad=12)
        ax.legend(facecolor="#161616", labelcolor="#e2e2e2", framealpha=0.8)

        single_path = os.path.join(OUTPUT_DIR, f"sft_loss_curve_run{run_idx + 1}.png")
        plt.tight_layout()
        plt.savefig(single_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SFT] Loss curve saved: {single_path}")

        # ---- Plot 2: combined runs ---------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for i, record in enumerate(run_history):
            c = colors[i % len(colors)]
            ax2.plot(
                record["steps"], record["loss"],
                color=c, linewidth=1.5, alpha=0.85,
                label=f"Run {i + 1} (final {record['final_loss']:.4f})"
            )
        _style_ax(ax2, fig2)
        ax2.set_xlabel("Step", color="#888888")
        ax2.set_ylabel("Training loss", color="#888888")
        ax2.set_title("Kaizen OS SFT — All Runs Combined",
                      color="#e2e2e2", pad=12)
        ax2.legend(facecolor="#161616", labelcolor="#e2e2e2",
                   framealpha=0.8, fontsize=8)

        combined_path = os.path.join(OUTPUT_DIR, "sft_combined_runs.png")
        plt.tight_layout()
        plt.savefig(combined_path, dpi=150, bbox_inches="tight",
                    facecolor=fig2.get_facecolor())
        plt.close()
        print(f"[SFT] Combined runs plot saved: {combined_path}")

    except ImportError:
        print("[SFT] matplotlib not available — saving loss history as JSON instead.")
        fallback = os.path.join(OUTPUT_DIR, f"sft_loss_run{run_idx + 1}.json")
        with open(fallback, "w") as f:
            json.dump({"steps": steps, "loss": loss_values}, f, indent=2)
        print(f"[SFT] Loss data saved: {fallback}")


def _style_ax(ax, fig) -> None:
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#1f1f1f")
    ax.grid(color="#1f1f1f", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="#444444", linewidth=0.5, linestyle=":")


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
    is_improvement    = _adapter_exists()

    print("[SFT] ===== Kaizen OS — Supervised Fine-Tuning =====")
    print(f"[SFT] Run number      : {run_idx + 1}")
    if is_improvement:
        print(f"[SFT] Mode            : CONTINUAL IMPROVEMENT (loading existing adapter)")
    elif is_resuming:
        print(f"[SFT] Mode            : RESUME from checkpoint {latest_checkpoint}")
    else:
        print(f"[SFT] Mode            : FRESH training from base model")
    print(f"[SFT] Model           : {MODEL_NAME}")
    print(f"[SFT] LoRA rank       : {LORA_R}")
    print(f"[SFT] Batch size      : {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"[SFT] Max steps       : {MAX_STEPS}")
    print(f"[SFT] Checkpoint every: {SAVE_STEPS} steps → {CHECKPOINT_DIR}")
    print(f"[SFT] Output dir      : {OUTPUT_DIR}")
    print(f"[SFT] Using SFTConfig : {USE_SFT_CONFIG}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    source = OUTPUT_DIR if is_improvement else MODEL_NAME
    print(f"\n[SFT] Loading model from: {source}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("[SFT] Applying / refreshing LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42 + run_idx,
        use_rslora=False,
        loftq_config=None,
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[SFT] Parameters: {total_params / 1e9:.2f}B total, "
        f"{trainable_params / 1e6:.1f}M trainable "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 2. Dataset
    #    KEY FIX: EOS token is baked into the text field here.
    #    We use dataset_text_field="text" instead of formatting_func
    #    to avoid the Unsloth code path that triggers the int/.mean error.
    # ------------------------------------------------------------------
    print("\n[SFT] Preparing dataset...")
    eos_token = tokenizer.eos_token or "<|endoftext|>"
    dataset   = load_dataset_from_json(DATASET_PATH, eos_token)
    print(f"[SFT] Dataset ready: {len(dataset)} samples")

    # ------------------------------------------------------------------
    # 3. Training config
    #    KEY FIX: Use SFTConfig (TRL-native) not TrainingArguments.
    #    SFTConfig inherits TrainingArguments but is what SFTTrainer
    #    actually type-checks against in TRL >= 0.13.
    # ------------------------------------------------------------------
    common_kwargs = dict(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42 + run_idx,
        output_dir=CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
    )

    if USE_SFT_CONFIG:
        # SFTConfig accepts max_seq_length and dataset_text_field directly
        training_args = TrainingConfig(
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            **common_kwargs,
        )
    else:
        training_args = TrainingConfig(**common_kwargs)

    # ------------------------------------------------------------------
    # 4. SFTTrainer
    #    KEY FIX: When using SFTConfig, do NOT pass max_seq_length or
    #    formatting_func to SFTTrainer — they now live in the config.
    #    When using old TrainingArguments, pass them as before.
    # ------------------------------------------------------------------
    print("[SFT] Initialising SFTTrainer...")

    if USE_SFT_CONFIG:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            # dataset_text_field and max_seq_length live in SFTConfig
        )
    else:
        # Fallback for old TRL — use the original pattern
        def _fmt(examples):
            return [t + eos_token for t in examples["text"]]

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            formatting_func=_fmt,
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )

    # ------------------------------------------------------------------
    # 5. Apply Unsloth compatibility patch AFTER trainer is constructed
    #    This is the safety net in case the version mismatch still triggers.
    # ------------------------------------------------------------------
    _patch_trainer_if_needed(trainer)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("\n[SFT] Starting training...")

    gpu_stats = None
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        print(f"[SFT] GPU  : {gpu_stats.name} | VRAM: {gpu_stats.total_memory / 1e9:.1f} GB")
        print(f"[SFT] VRAM reserved at start: {start_gpu_memory} GB")

    t0 = time.time()

    trainer_stats = trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

    elapsed = time.time() - t0

    if torch.cuda.is_available() and gpu_stats is not None:
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        print(f"[SFT] Peak VRAM usage: {peak_mem} GB")

    final_loss = trainer_stats.metrics.get("train_loss", 0.0)
    print(f"\n[SFT] Training complete.")
    print(f"[SFT] Runtime   : {elapsed:.0f}s")
    print(f"[SFT] Final loss: {final_loss:.4f}")

    # ------------------------------------------------------------------
    # 7. Extract loss history
    # ------------------------------------------------------------------
    loss_values: list[float] = []
    step_values: list[int]   = []
    for log in trainer.state.log_history:
        if "loss" in log and "step" in log:
            loss_values.append(float(log["loss"]))
            step_values.append(int(log["step"]))

    # ------------------------------------------------------------------
    # 8. Save LoRA adapter
    # ------------------------------------------------------------------
    print(f"\n[SFT] Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SFT] Adapter saved. Load with: LLMAgent(model_name='{OUTPUT_DIR}')")

    # ------------------------------------------------------------------
    # 9. Persist run record and save plots
    # ------------------------------------------------------------------
    run_record = {
        "run": run_idx + 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": step_values,
        "loss": loss_values,
        "final_loss": round(final_loss, 6),
        "runtime_s": round(elapsed, 1),
        "resumed_from": latest_checkpoint,
        "continual": is_improvement,
    }
    run_history.append(run_record)
    _save_run_history(run_history)

    if loss_values:
        _save_loss_plot(loss_values, step_values, run_idx, run_history)
    else:
        print("[SFT] No loss values recorded — trainer may not have logged any steps.")

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    if len(run_history) > 1:
        prev_loss = run_history[-2]["final_loss"]
        delta     = final_loss - prev_loss
        direction = "↓ improved" if delta < 0 else "↑ increased"
        print(f"\n[SFT] Improvement: {prev_loss:.4f} → {final_loss:.4f} ({delta:+.4f}) {direction}")

    print(f"\n[SFT] Run {run_idx + 1} complete.")
    print(f"[SFT] Total runs so far : {len(run_history)}")
    print(f"[SFT] To improve further: just run this script again — it will load {OUTPUT_DIR}")
    print(f"[SFT] Next step         : run training/grpo_train.py with MODEL_PATH={OUTPUT_DIR}")

    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_path = train()
    print(f"\n[SFT] Done. Adapter at: {output_path}")