"""
training/sft_train.py
Supervised Fine-Tuning (SFT) for the Kaizen OS agent using Unsloth + LoRA.

Run on Google Colab T4 (free tier, 16 GB VRAM):
    !pip install unsloth trl datasets accelerate bitsandbytes
    !python training/sft_train.py

CHECKPOINT & CONTINUAL IMPROVEMENT
------------------------------------
On first run  : trains from scratch on the base model, saves adapter to OUTPUT_DIR.
On second run : detects the existing adapter in OUTPUT_DIR and resumes from the
                latest checkpoint automatically, then continues training for
                another MAX_STEPS on top of the previous run's weights.

This means every run improves the previous model — you never start over unless
you delete OUTPUT_DIR manually.

Checkpoints are saved every SAVE_STEPS steps inside OUTPUT_DIR/checkpoints/.
If Colab crashes mid-run the trainer resumes from the latest checkpoint.

After training the LoRA adapter is saved to OUTPUT_DIR.
The adapter can then be loaded by LLMAgent(model_name=OUTPUT_DIR).

Plot images
-----------
Two PNG files are saved at the end of every run:
  - sft_loss_curve_runN.png  — training loss per logging step
  - sft_combined_runs.png    — all past runs overlaid on one chart (auto-grows)

Run history is persisted in OUTPUT_DIR/run_history.json so the combined
chart stays accurate across Colab sessions.
"""

import json
import os
import sys
import glob
import time
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
SAVE_STEPS       = 25        # save a checkpoint every N steps
SAVE_TOTAL_LIMIT = 4         # keep only the last N checkpoints to save disk space

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
        print("[SFT] Install with: pip install unsloth trl datasets accelerate bitsandbytes")
        sys.exit(1)

_check_imports()

from unsloth import FastLanguageModel          # type: ignore
from datasets import Dataset                   # type: ignore
from trl import SFTTrainer                     # type: ignore
from transformers import TrainingArguments     # type: ignore
import torch


# ---------------------------------------------------------------------------
# Run history helpers
# ---------------------------------------------------------------------------

def _load_run_history() -> list[dict]:
    """Load past run records from disk. Returns empty list if none exist."""
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
    Return path to the latest Hugging Face checkpoint directory inside
    CHECKPOINT_DIR, or None if no checkpoints exist.

    HF TrainingArguments saves checkpoints as checkpoint-<step> folders.
    We pick the one with the highest step number.
    """
    pattern = os.path.join(CHECKPOINT_DIR, "checkpoint-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Sort by step number extracted from directory name
    def _step(p: str) -> int:
        try:
            return int(os.path.basename(p).split("-")[-1])
        except ValueError:
            return -1
    candidates.sort(key=_step)
    return candidates[-1]


def _adapter_exists() -> bool:
    """True if a trained LoRA adapter already lives in OUTPUT_DIR."""
    return os.path.isfile(os.path.join(OUTPUT_DIR, "adapter_config.json"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset_from_json(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"[SFT] Loaded {len(examples)} training examples from {path}")

    texts = []
    for ex in examples:
        formatted = ALPACA_TEMPLATE.format(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
            output=ex["output"],
        )
        texts.append({"text": formatted, "raw_output": ex["output"]})

    return Dataset.from_list(texts)


def formatting_func(examples: dict, eos_token: str) -> list[str]:
    return [text + eos_token for text in examples["text"]]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_loss_plot(
    loss_values: list[float],
    steps: list[int],
    run_idx: int,
    run_history: list[dict],
) -> None:
    """
    Save two PNG files:
      1. sft_loss_curve_run<N>.png  — current run only
      2. sft_combined_runs.png      — all past runs overlaid
    """
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
    """Apply the Kaizen dark theme to a matplotlib axes."""
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
    run_idx     = len(run_history)          # 0-based index of this run

    latest_checkpoint = _find_latest_checkpoint()
    is_resuming       = latest_checkpoint is not None
    is_improvement    = _adapter_exists()   # True if this is run 2+

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

    # ------------------------------------------------------------------
    # 1. Load model
    #    - If a trained adapter already exists (run 2+): load it directly
    #      so further training keeps improving the same weights.
    #    - If no adapter exists (run 1): load the base model.
    # ------------------------------------------------------------------
    if is_improvement:
        print(f"\n[SFT] Loading existing adapter from {OUTPUT_DIR} for continual improvement...")
        source = OUTPUT_DIR
    else:
        print(f"\n[SFT] Loading base model: {MODEL_NAME}...")
        source = MODEL_NAME

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
        random_state=42 + run_idx,   # vary seed per run for different gradient paths
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
    # ------------------------------------------------------------------
    print("\n[SFT] Preparing dataset...")
    dataset   = load_dataset_from_json(DATASET_PATH)
    eos_token = tokenizer.eos_token

    def _fmt(examples):
        return formatting_func(examples, eos_token)

    # ------------------------------------------------------------------
    # 3. Training arguments — checkpointing enabled
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
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
        output_dir=CHECKPOINT_DIR,          # checkpoints go here
        save_strategy="steps",              # save every SAVE_STEPS steps
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
    )

    # ------------------------------------------------------------------
    # 4. SFTTrainer
    # ------------------------------------------------------------------
    print("[SFT] Initialising SFTTrainer...")
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
    # 5. Train (resume from checkpoint if one exists)
    # ------------------------------------------------------------------
    print("\n[SFT] Starting training...")

    gpu_stats = None
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        print(f"[SFT] GPU  : {gpu_stats.name} | VRAM: {gpu_stats.total_memory / 1e9:.1f} GB")
        print(f"[SFT] VRAM reserved at start: {start_gpu_memory} GB")

    t0 = time.time()

    # Pass resume_from_checkpoint so HF Trainer restores optimizer state too
    trainer_stats = trainer.train(
        resume_from_checkpoint=latest_checkpoint  # None on fresh run = start over
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
    # 6. Extract loss history from trainer logs
    # ------------------------------------------------------------------
    loss_values: list[float] = []
    step_values: list[int]   = []
    for log in trainer.state.log_history:
        if "loss" in log and "step" in log:
            loss_values.append(float(log["loss"]))
            step_values.append(int(log["step"]))

    # ------------------------------------------------------------------
    # 7. Save LoRA adapter (overwrites previous run — intentional)
    # ------------------------------------------------------------------
    print(f"\n[SFT] Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SFT] Adapter saved. Load with: LLMAgent(model_name='{OUTPUT_DIR}')")

    # ------------------------------------------------------------------
    # 8. Persist run record and save plots
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
    # 9. Summary for the user
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