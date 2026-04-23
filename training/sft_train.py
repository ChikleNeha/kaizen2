"""
training/sft_train.py
Supervised Fine-Tuning (SFT) for the Kaizen OS agent using Unsloth + LoRA.

Run on Google Colab T4 (free tier, 16 GB VRAM):
    !pip install unsloth trl datasets accelerate bitsandbytes
    !python training/sft_train.py

After training the LoRA adapter is saved to OUTPUT_DIR.
The adapter can then be loaded by LLMAgent(model_name=OUTPUT_DIR).

HF Credits upgrade path
-----------------------
Change MODEL_NAME to a larger model — everything else stays identical.
Example: "unsloth/Qwen2.5-14B-Instruct" or "unsloth/Llama-3.1-8B-Instruct"
"""

import json
import os
import sys

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
OUTPUT_DIR       = "./kaizen_sft_model"

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

EOS_TOKEN_PLACEHOLDER = "{eos_token}"

# ---------------------------------------------------------------------------
# Imports (deferred so the config block is visible before heavy imports)
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
# Helpers
# ---------------------------------------------------------------------------

def load_dataset_from_json(path: str) -> Dataset:
    """
    Load the golden_examples.json file and return a HuggingFace Dataset.

    Each example is formatted using the Alpaca prompt template with the
    EOS token appended so the model learns where to stop generating.
    """
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"[SFT] Loaded {len(examples)} training examples from {path}")

    # We store the formatted text in a 'text' column — SFTTrainer expects this
    texts = []
    for ex in examples:
        formatted = ALPACA_TEMPLATE.format(
            instruction=ex["instruction"],
            input=ex.get("input", ""),
            output=ex["output"],
        )
        # EOS token will be appended after tokeniser is loaded (see train())
        texts.append({"text": formatted, "raw_output": ex["output"]})

    return Dataset.from_list(texts)


def formatting_func(examples: dict, eos_token: str) -> list[str]:
    """
    Called by SFTTrainer to format each batch.
    Appends the EOS token so the model learns sequence termination.
    """
    return [text + eos_token for text in examples["text"]]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    print("[SFT] ===== Kaizen OS — Supervised Fine-Tuning =====")
    print(f"[SFT] Model        : {MODEL_NAME}")
    print(f"[SFT] LoRA rank    : {LORA_R}")
    print(f"[SFT] Batch size   : {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"[SFT] Max steps    : {MAX_STEPS}")
    print(f"[SFT] Output dir   : {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 1. Load base model with Unsloth + LoRA
    # ------------------------------------------------------------------
    print("\n[SFT] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,              # auto-detect: float16 on GPU, float32 on CPU
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
        use_gradient_checkpointing="unsloth",  # saves VRAM on T4
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Print trainable parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[SFT] Parameters: {total_params / 1e9:.2f}B total, "
        f"{trainable_params / 1e6:.1f}M trainable "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 2. Load and format dataset
    # ------------------------------------------------------------------
    print("\n[SFT] Preparing dataset...")
    dataset = load_dataset_from_json(DATASET_PATH)
    eos_token = tokenizer.eos_token

    # Wrap formatting_func to capture eos_token via closure
    def _fmt(examples):
        return formatting_func(examples, eos_token)

    # ------------------------------------------------------------------
    # 3. Training arguments
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
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="none",        # set to "wandb" if wandb is configured
        save_strategy="no",      # save manually at end to avoid checkpoint bloat
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
        packing=False,            # packing=True can cause sequence mixing issues
        args=training_args,
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[SFT] Starting training...")
    print(f"[SFT] Steps: {MAX_STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM}")

    gpu_stats = None
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"[SFT] GPU: {gpu_stats.name} | VRAM: {gpu_stats.total_memory / 1e9:.1f} GB")
        print(f"[SFT] Reserved at start: {start_gpu_memory} GB")

    trainer_stats = trainer.train()

    if torch.cuda.is_available() and gpu_stats is not None:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        print(f"[SFT] Peak VRAM usage: {used_memory} GB")

    print(f"\n[SFT] Training complete.")
    print(f"[SFT] Runtime: {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"[SFT] Final loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")

    # ------------------------------------------------------------------
    # 6. Save LoRA adapter
    # ------------------------------------------------------------------
    print(f"\n[SFT] Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[SFT] Adapter saved. Load with: LLMAgent(model_name='{OUTPUT_DIR}')")

    # ------------------------------------------------------------------
    # 7. Optional: push to HuggingFace Hub
    # ------------------------------------------------------------------
    hf_repo = os.environ.get("HF_REPO_ID", "")
    if hf_repo:
        print(f"\n[SFT] Pushing adapter to HuggingFace Hub: {hf_repo}")
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        print("[SFT] Push complete.")
    else:
        print("\n[SFT] Tip: set HF_REPO_ID env var to push the adapter to HuggingFace Hub.")
        print("[SFT] Example: HF_REPO_ID=your-username/kaizen-os-sft python training/sft_train.py")

    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_path = train()
    print(f"\n[SFT] Done. Next step: run training/grpo_train.py with MODEL_PATH={output_path}")
