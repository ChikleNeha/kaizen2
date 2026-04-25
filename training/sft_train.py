import os, json, glob, torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import matplotlib.pyplot as plt

# --- PATHS (Relative to Repo Root) ---
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
FINAL_OUTPUT_DIR = "./models/sft_adapter"  # Saved in repo for push
CHECKPOINT_DIR = "/content/checkpoints/sft" # Local to Colab (don't push)
PLOT_DIR = "./plots"
DATASET_PATH = "./training/golden_examples.json"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

def get_last_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints: return None
    checkpoints.sort(key=lambda x: int(x.rsplit("-", 1)[-1]))
    return checkpoints[-1]

def save_loss_plot(log_history, run_id):
    steps = [entry["step"] for entry in log_history if "loss" in entry]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss', color='#4ade80')
    plt.title(f"Kaizen SFT Loss - Run {run_id}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/sft_loss_run_{run_id}.png")
    plt.close()

def run_sft_training(run_id=1):
    print(f"[SFT] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
    )

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    
    formatted_data = [{"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}{tokenizer.eos_token}"} for ex in data]
    dataset = Dataset.from_list(formatted_data)

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        output_dir=CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    last_ckpt = get_last_checkpoint(CHECKPOINT_DIR)
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save artifact to repo
    save_loss_plot(trainer.state.log_history, run_id)
    model.save_pretrained(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
    print(f"[SFT] Run {run_id} complete. Plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    run_sft_training()