import os, json, glob, torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import matplotlib.pyplot as plt

# --- PATHS ---
SFT_ADAPTER_PATH = "./models/sft_adapter"
FINAL_GRPO_DIR = "./models/grpo_adapter"
CHECKPOINT_DIR = "/content/checkpoints/grpo"
PLOT_DIR = "./plots"
DATASET_PATH = "./training/golden_examples.json"

# Importing your environment logic (Ensure these exist in your repo)
from environment.kaizen_env import KaizenEnv
from environment.reward import compute_reward
from environment.action_space import parse_action

_env = KaizenEnv(broadcast=False)

def reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        # Extract text content
        content = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        _env.reset()
        try:
            action = parse_action(content)
            _, reward, _, _ = _env.step(action)
            rewards.append(float(reward))
        except:
            rewards.append(-1.0) # Penalty for hallucinated JSON
    return rewards

def run_grpo_training(run_id=1):
    print(f"[GRPO] Loading SFT adapter for RL stage...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_ADAPTER_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    
    # Filter for prompts involving chaos
    prompts = [{"prompt": f"### Observation:\n{ex['instruction']}\n\n### Response:"} for ex in data if "Chaos" in ex['instruction']]
    dataset = Dataset.from_list(prompts)

    grpo_config = GRPOConfig(
        num_generations=4, 
        max_completion_length=256,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=80,
        output_dir=CHECKPOINT_DIR,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        use_vllm=False
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    last_ckpt = max(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*")), 
                    key=lambda x: int(x.rsplit("-", 1)[-1])) if glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*")) else None
    
    trainer.train(resume_from_checkpoint=last_ckpt)
    
    # Save Final artifacts to repo for push
    model.save_pretrained(FINAL_GRPO_DIR)
    
    # Simple Reward Plotting
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"GRPO Run {run_id} Success", ha='center')
    plt.savefig(f"{PLOT_DIR}/grpo_reward_run_{run_id}.png")
    plt.close()
    print(f"[GRPO] Run {run_id} complete.")

if __name__ == "__main__":
    run_grpo_training()