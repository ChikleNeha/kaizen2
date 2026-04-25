from huggingface_hub import HfApi

api = HfApi()

print("Launching SFT + GRPO training job on HF...")

job = api.create_job(
    repo_id="NehaChikle/kaizen-qwen2.5-3b-sft",
    job_type="training",
    hardware="a10g-small",
    environment={
        "HF_REPO_ID": "NehaChikle/kaizen-qwen2.5-3b-sft"
    },
    command="bash run_training.sh",
)

print(f"✅ Job created successfully!")
print(f"Job details: {job}")
print(f"Monitor at: https://huggingface.co/NehaChikle/kaizen-qwen2.5-3b-sft")