"""
launch_job.py
=============
Launches a HuggingFace Training Job that:
  1. Spins up a pytorch/pytorch GPU container (a10g-small)
  2. Clones your GitHub repo inside the container
  3. Installs all dependencies
  4. Runs SFT (sft_train.py) then GRPO (grpo_train.py)
  5. Pushes both LoRA adapters to HF Hub

Usage:
    python launch_job.py

Required environment variables (set these before running):
    HF_TOKEN   — your HuggingFace token with write access
                 Get it at: https://huggingface.co/settings/tokens
                 Windows:  set HF_TOKEN=hf_xxxx
                 Mac/Linux: export HF_TOKEN=hf_xxxx

Cost estimate (a10g-small at ~$3.15/hr):
    SFT  100 steps : ~25-35 min  → ~$1.75
    GRPO  80 steps : ~35-50 min  → ~$2.25
    Total           : ~60-85 min → ~$3.50-$4.50 out of $30
"""

import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Configuration — edit these if anything changes
# ---------------------------------------------------------------------------

GITHUB_REPO  = "https://github.com/ChikleNeha/kaizen2"
HF_REPO_ID   = "NehaChikle/kaizen-qwen2.5-3b-sft"   # existing model repo

# a10g-small: 24GB VRAM, enough for Qwen2.5-3B in 4-bit with room to spare
# If you want to be even safer on cost, change to "t4-medium" (~$0.90/hr)
HARDWARE_FLAVOR = "a10g-small"

# 2 hours = comfortable upper bound for SFT + GRPO; job auto-stops early if done
TIMEOUT_SECONDS = 7200

# Docker image — matches CUDA 12.4 that unsloth supports well
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"

# ---------------------------------------------------------------------------
# The command the container will run.
# We inline the entire run_training.sh content as a bash -c string.
# This avoids the "file not found" error from the previous attempt.
# ---------------------------------------------------------------------------

# Read run_training.sh from the same directory as this script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sh_path = os.path.join(_script_dir, "run_training.sh")

if not os.path.exists(_sh_path):
    print(f"ERROR: run_training.sh not found at {_sh_path}")
    print("Make sure run_training.sh is in the same directory as launch_job.py")
    sys.exit(1)

with open(_sh_path, "r") as f:
    _sh_content = f.read()

# The command: write the script to a temp file and execute it.
# This is more reliable than `bash -c '<very long string>'` which can hit
# argument length limits.
CONTAINER_COMMAND = [
    "bash", "-c",
    textwrap.dedent(f"""
        cat > /tmp/run_training.sh << 'ENDOFSCRIPT'
{_sh_content}
ENDOFSCRIPT
        chmod +x /tmp/run_training.sh
        bash /tmp/run_training.sh
    """).strip()
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Token check
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print()
        print("Set it with:")
        print("  Windows:   set HF_TOKEN=hf_your_token_here")
        print("  Mac/Linux: export HF_TOKEN=hf_your_token_here")
        print()
        print("Get your token at: https://huggingface.co/settings/tokens")
        print("(Needs 'write' permission)")
        sys.exit(1)

    if not hf_token.startswith("hf_"):
        print("WARNING: HF_TOKEN doesn't look like a HuggingFace token (should start with hf_)")
        print("Continuing anyway...")

    # -- Import
    try:
        from huggingface_hub import run_job, HfApi
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    # -- Verify the target model repo exists (or create it)
    print(f"Checking HF Hub repo: {HF_REPO_ID} ...")
    api = HfApi(token=hf_token)
    try:
        api.repo_info(HF_REPO_ID, repo_type="model")
        print(f"  ✓ Repo exists: https://huggingface.co/{HF_REPO_ID}")
    except Exception:
        print(f"  Repo not found — creating {HF_REPO_ID} ...")
        api.create_repo(HF_REPO_ID, repo_type="model", private=False, exist_ok=True)
        print(f"  ✓ Created: https://huggingface.co/{HF_REPO_ID}")

    # Also ensure the GRPO repo exists
    grpo_repo = HF_REPO_ID.replace("-sft", "-grpo")
    try:
        api.repo_info(grpo_repo, repo_type="model")
        print(f"  ✓ GRPO repo exists: https://huggingface.co/{grpo_repo}")
    except Exception:
        print(f"  Creating GRPO repo: {grpo_repo} ...")
        api.create_repo(grpo_repo, repo_type="model", private=False, exist_ok=True)
        print(f"  ✓ Created: https://huggingface.co/{grpo_repo}")

    # -- Print job summary before launching
    print()
    print("=" * 60)
    print("  KAIZEN OS — HF Training Job")
    print("=" * 60)
    print(f"  Hardware  : {HARDWARE_FLAVOR}  (~$3.15/hr)")
    print(f"  Timeout   : {TIMEOUT_SECONDS // 60} min")
    print(f"  Est. cost : $3.50 – $4.50")
    print(f"  Remaining : ~$25–26 after this job")
    print(f"  Source    : {GITHUB_REPO}")
    print(f"  SFT out   : https://huggingface.co/{HF_REPO_ID}")
    print(f"  GRPO out  : https://huggingface.co/{grpo_repo}")
    print("=" * 60)
    print()

    confirm = input("Launch job? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    # -- Launch
    print("\nLaunching HF Training Job...")

    try:
        job = run_job(
            image=DOCKER_IMAGE,
            command=CONTAINER_COMMAND,
            flavor=HARDWARE_FLAVOR,
            timeout=TIMEOUT_SECONDS,
            env={
                "HF_REPO_ID":   HF_REPO_ID,
                "GITHUB_REPO":  GITHUB_REPO,
                # Standard HF env vars — enables model caching and faster downloads
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "TRANSFORMERS_CACHE": "/tmp/hf_cache",
                "HF_HOME": "/tmp/hf_home",
            },
            secrets={
                # Passed as secret so token never appears in logs
                "HF_TOKEN": hf_token,
            },
            token=hf_token,
        )
    except Exception as e:
        print(f"\nERROR launching job: {e}")
        print()
        print("Common causes:")
        print("  - Invalid HF_TOKEN (needs write + inference permissions)")
        print("  - huggingface_hub version too old: pip install -U huggingface_hub")
        print(f"  - Invalid flavor name '{HARDWARE_FLAVOR}' — check available flavors:")
        print("    python -c \"from huggingface_hub import HfApi; [print(h) for h in HfApi().list_jobs_hardware()]\"")
        sys.exit(1)

    print()
    print("✅ Job launched successfully!")
    print()
    print(f"  Job ID  : {job.id}")
    print(f"  Monitor : {getattr(job, 'url', 'https://huggingface.co/jobs')}")
    print()
    print("Track logs with:")
    print(f"  python -c \"from huggingface_hub import fetch_job_logs; [print(l) for l in fetch_job_logs('{job.id}')]\"")
    print()
    print("Or stream logs live:")
    print(f"  python stream_logs.py {job.id}")
    print()
    print("When training is done:")
    print(f"  SFT adapter  → https://huggingface.co/{HF_REPO_ID}")
    print(f"  GRPO adapter → https://huggingface.co/{grpo_repo}")
    print()
    print("Then in your HF Space (NehaChikle/kaizen-os) set:")
    print(f"  KAIZEN_MODEL_NAME = {HF_REPO_ID}")
    print(f"  KAIZEN_DEMO_MODE  = false")

    # Save job ID to file so you can reference it later
    with open("last_job_id.txt", "w") as f:
        f.write(job.id)
    print(f"\n  Job ID saved to last_job_id.txt")

    return job


if __name__ == "__main__":
    main()