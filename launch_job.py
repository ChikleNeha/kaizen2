import os
from huggingface_hub import run_job

# Token comes from environment — never hardcode it
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Run: set HF_TOKEN=your_token")

print("Launching SFT + GRPO training job on HF...")

job = run_job(
    image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    command=["bash", "run_training.sh"],
    flavor="a10g-small",
    timeout=7200,  # 2 hours — covers SFT + GRPO with buffer
    env={
        "HF_REPO_ID": "NehaChikle/kaizen-qwen2.5-3b-sft"
    },
    secrets={
        "HF_TOKEN": hf_token
    },
)

print(f"✅ Job launched!")
print(f"Monitor at: {job.url}")
print(f"Job ID: {job.id}")