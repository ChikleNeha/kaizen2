"""
training/sft_train.py
Supervised Fine-Tuning (SFT) for the Kaizen OS agent using Unsloth + LoRA.

CHANGES vs original:
  - save_strategy="steps", save_steps=25, save_total_limit=3  (checkpointing)
  - get_last_checkpoint() helper so training auto-resumes on Colab crash
  - trainer.train(resume_from_checkpoint=last_checkpoint)
  - load_best_model_at_end=False (required — no eval set with max_steps)
"""

import json
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME      = "unsloth/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH  = 2048
LOAD_IN_4BIT    = True
LORA_R          = 16
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.0
TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
BATCH_SIZE      = 2
GRAD_ACCUM      = 4
LEARNING_RATE   = 2e-4
MAX_STEPS       = 100
SAVE_STEPS      = 25          # checkpoints at step 25, 50, 75, 100
SAVE_TOTAL_LIMIT = 3          # keep last 3 checkpoints → saves disk space
LOSS_PLOT_PATH  = "./loss_curve.png"

# Use absolute path from env var so GRPO can find it reliably.
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "/workspace/kaizen_sft_model")

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_examples.json")

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


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
        print(f"[SFT] Missing packages: {missing}")
        sys.exit(1)

_check_imports()

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_last_checkpoint(output_dir: str):
    """
    Scans output_dir for checkpoint-N folders and returns the path of the
    highest-numbered one, or None if no checkpoints exist.
    Allows training to resume after a Colab crash without restarting from 0.
    """
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number (the integer after the last '-')
    checkpoints.sort(key=lambda x: int(x.rsplit("-", 1)[-1]))
    last = checkpoints[-1]
    print(f"[SFT] Found existing checkpoint: {last}")
    print(f"[SFT] Training will RESUME from this checkpoint.")
    return last


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
        texts.append({"text": formatted})
    return Dataset.from_list(texts)


def formatting_func(examples: dict, eos_token: str) -> list[str]:
    return [text + eos_token for text in examples["text"]]


def save_loss_plot(log_history: list[dict], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps  = [entry["step"] for entry in log_history if "loss" in entry]
        losses = [entry["loss"] for entry in log_history if "loss" in entry]

        if not steps:
            print("[SFT] No loss data to plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, color="#4ade80", linewidth=1.5, label="Training loss")
        ax.set_facecolor("#0f0f0f")
        fig.patch.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#888888")
        ax.spines[:].set_color("#1f1f1f")
        ax.set_xlabel("Step", color="#888888")
        ax.set_ylabel("Loss", color="#888888")
        ax.set_title("Kaizen OS — SFT Loss Curve", color="#e2e2e2", pad=12)
        ax.legend(facecolor="#161616", labelcolor="#e2e2e2", framealpha=0.8)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"[SFT] Loss curve saved to: {path}")
        print(f"[SFT] >>> Commit {path} to your repo before submission <<<")

    except ImportError:
        json_path = path.replace(".png", "_loss.json")
        data = {entry["step"]: entry["loss"] for entry in log_history if "loss" in entry}
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[SFT] matplotlib not available. Loss data saved to: {json_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    print("[SFT] ===== Kaizen OS — Supervised Fine-Tuning =====")
    print(f"[SFT] Model        : {MODEL_NAME}")
    print(f"[SFT] LoRA rank    : {LORA_R}")
    print(f"[SFT] Batch size   : {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"[SFT] Max steps    : {MAX_STEPS}")
    print(f"[SFT] Save every   : {SAVE_STEPS} steps (keep last {SAVE_TOTAL_LIMIT})")
    print(f"[SFT] Output dir   : {OUTPUT_DIR}")

    # ── Check for existing checkpoint (resume support) ────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint:
        print(f"[SFT] ⚡ Resuming from: {last_checkpoint}")
    else:
        print(f"[SFT] No checkpoint found — starting fresh.")

    print("\n[SFT] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("[SFT] Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
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

    print("\n[SFT] Preparing dataset...")
    dataset   = load_dataset_from_json(DATASET_PATH)
    eos_token = tokenizer.eos_token

    def _fmt(examples):
        return formatting_func(examples, eos_token)

    training_args = TrainingArguments(
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        warmup_steps                  = 5,
        max_steps                     = MAX_STEPS,
        learning_rate                 = LEARNING_RATE,
        fp16                          = not torch.cuda.is_bf16_supported(),
        bf16                          = torch.cuda.is_bf16_supported(),
        logging_steps                 = 10,
        optim                         = "adamw_8bit",
        weight_decay                  = 0.01,
        lr_scheduler_type             = "linear",
        seed                          = 42,
        output_dir                    = OUTPUT_DIR,
        report_to                     = "none",

        # ── Checkpointing ─────────────────────────────────────────────────
        save_strategy                 = "steps",
        save_steps                    = SAVE_STEPS,   # saves at 25, 50, 75, 100
        save_total_limit              = SAVE_TOTAL_LIMIT,  # keep last 3 only
        load_best_model_at_end        = False,        # no eval set → must be False
        # ──────────────────────────────────────────────────────────────────
    )

    print("[SFT] Initialising SFTTrainer...")

    import trl as _trl
    _trl_version = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    _use_processing_class = _trl_version >= (0, 12)
    print(
        f"[SFT] TRL version: {_trl.__version__} — "
        f"using {'processing_class' if _use_processing_class else 'tokenizer'}="
    )

    trainer_kwargs = dict(
        model              = model,
        train_dataset      = dataset,
        formatting_func    = _fmt,
        max_seq_length     = MAX_SEQ_LENGTH,
        dataset_num_proc   = 1,
        packing            = False,
        args               = training_args,
    )
    if _use_processing_class:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("\n[SFT] Starting training...")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"[SFT] GPU: {gpu_stats.name} | VRAM: {gpu_stats.total_memory / 1e9:.1f} GB")

    # ── Resume from checkpoint if one was found ────────────────────────────
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    if torch.cuda.is_available():
        peak_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        print(f"[SFT] Peak VRAM: {peak_mem} GB")

    print(f"\n[SFT] Training complete.")
    print(f"[SFT] Runtime   : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"[SFT] Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    print(f"\n[SFT] Saving loss curve to {LOSS_PLOT_PATH}...")
    save_loss_plot(trainer.state.log_history, LOSS_PLOT_PATH)

    # ── Save final LoRA adapter (separate from checkpoints) ───────────────
    print(f"\n[SFT] Saving final LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SFT] Adapter saved.")

    # ── Push to HF Hub ────────────────────────────────────────────────────
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_repo:
        print(f"\n[SFT] Pushing adapter to HuggingFace Hub: {hf_repo}")
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print("[SFT] Push complete.")
    else:
        print("\n[SFT] Tip: set HF_REPO_ID to push to HuggingFace Hub.")

    return OUTPUT_DIR


if __name__ == "__main__":
    output_path = train()
    print(f"\n[SFT] Done. Next: python training/grpo_train.py")