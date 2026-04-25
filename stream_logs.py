"""
stream_logs.py
==============
Streams logs from a running HuggingFace training job.

Usage:
    python stream_logs.py <job_id>
    python stream_logs.py last        ← reads job ID from last_job_id.txt

Fix: huggingface_hub >= 1.0 requires ALL job API args as keyword-only.
     inspect_job(job_id=...) not inspect_job(...)
"""

import os
import sys
import time

# ── Read job ID ───────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    if os.path.exists("last_job_id.txt"):
        with open("last_job_id.txt") as f:
            job_id = f.read().strip()
        print(f"Using job ID from last_job_id.txt: {job_id}")
    else:
        print("Usage: python stream_logs.py <job_id>")
        print("       python stream_logs.py last")
        sys.exit(1)
else:
    job_id = sys.argv[1]
    if job_id == "last":
        if os.path.exists("last_job_id.txt"):
            with open("last_job_id.txt") as f:
                job_id = f.read().strip()
            print(f"Using job ID from last_job_id.txt: {job_id}")
        else:
            print("ERROR: last_job_id.txt not found.")
            sys.exit(1)

# ── Auth ──────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set.")
    print("  export HF_TOKEN=hf_xxxx")
    sys.exit(1)

# ── Import ────────────────────────────────────────────────────────────────────
try:
    from huggingface_hub import fetch_job_logs, inspect_job
except ImportError:
    print("ERROR: huggingface_hub not installed.")
    print("  pip install huggingface_hub")
    sys.exit(1)

# ── Stream ────────────────────────────────────────────────────────────────────
print(f"=== Streaming logs for job: {job_id} ===")
print("(Press Ctrl+C to stop streaming — job will keep running)\n")

POLL_INTERVAL = 5   # seconds between retries when logs aren't ready yet
seen_lines    = 0

try:
    while True:
        # ── Check job status first (all kwargs keyword-only) ─────────────────
        try:
            job_info = inspect_job(job_id=job_id, token=HF_TOKEN)
            status   = getattr(job_info, "status", "unknown")
        except Exception as e:
            print(f"[stream] Could not get job status: {e}")
            status = "unknown"

        # ── Fetch logs (keyword-only args) ────────────────────────────────────
        try:
            lines = list(fetch_job_logs(job_id=job_id, token=HF_TOKEN))
            new_lines = lines[seen_lines:]
            for line in new_lines:
                print(line, flush=True)
            seen_lines = len(lines)
        except Exception as e:
            print(f"[stream] Log fetch error: {e}")

        # ── Exit if job is done ───────────────────────────────────────────────
        if status in ("completed", "failed", "cancelled", "error"):
            print(f"\n=== Job finished with status: {status} ===")
            break

        # ── Still running — wait and poll again ──────────────────────────────
        if status not in ("completed", "failed", "cancelled", "error"):
            time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:
    print(f"\nStopped streaming. Job is still running.")
    print(f"Resume with: python stream_logs.py {job_id}")