"""
stream_logs.py
==============
Stream live logs from a running HF Training Job.

Usage:
    python stream_logs.py                    # reads job ID from last_job_id.txt
    python stream_logs.py <job_id>           # use a specific job ID

The job ID looks like: job_abc123xyz
"""

import sys
import os
import time

def main():
    # -- Get job ID
    job_id = None

    if len(sys.argv) > 1:
        job_id = sys.argv[1].strip()
    elif os.path.exists("last_job_id.txt"):
        with open("last_job_id.txt") as f:
            job_id = f.read().strip()
        print(f"Using job ID from last_job_id.txt: {job_id}")
    else:
        print("ERROR: No job ID provided.")
        print("Usage: python stream_logs.py <job_id>")
        print("   or: python stream_logs.py  (reads from last_job_id.txt)")
        sys.exit(1)

    # -- Token
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None

    # -- Import
    try:
        from huggingface_hub import fetch_job_logs, inspect_job
    except ImportError:
        print("ERROR: huggingface_hub not installed. pip install huggingface_hub")
        sys.exit(1)

    print(f"\n=== Streaming logs for job: {job_id} ===")
    print("(Press Ctrl+C to stop streaming — job will keep running)\n")

    seen_lines = 0
    poll_interval = 5  # seconds between log fetches

    try:
        while True:
            # Check job status
            try:
                job_info = inspect_job(job_id, token=hf_token)
                status = str(job_info.status)
            except Exception as e:
                print(f"[stream] Could not get job status: {e}")
                status = "unknown"

            # Fetch all logs so far
            try:
                logs = list(fetch_job_logs(job_id, token=hf_token))
            except Exception as e:
                print(f"[stream] Log fetch error: {e}")
                time.sleep(poll_interval)
                continue

            # Print new lines only
            if len(logs) > seen_lines:
                for line in logs[seen_lines:]:
                    # fetch_job_logs returns strings
                    print(line, end="" if line.endswith("\n") else "\n")
                seen_lines = len(logs)

            # Stop if job is done
            terminal_statuses = {"completed", "failed", "cancelled", "error"}
            if any(t in status.lower() for t in terminal_statuses):
                print(f"\n=== Job {status} ===")

                if "completed" in status.lower():
                    print("\n✅ Training complete!")
                    print(f"  SFT adapter  → https://huggingface.co/NehaChikle/kaizen-qwen2.5-3b-sft")
                    print(f"  GRPO adapter → https://huggingface.co/NehaChikle/kaizen-qwen2.5-3b-grpo")
                    print()
                    print("Next: set these in your HF Space (NehaChikle/kaizen-os):")
                    print("  KAIZEN_MODEL_NAME = NehaChikle/kaizen-qwen2.5-3b-sft")
                    print("  KAIZEN_DEMO_MODE  = false")
                elif "failed" in status.lower() or "error" in status.lower():
                    print("\n❌ Job failed. Check the logs above for the error.")
                    print()
                    print("Common fixes:")
                    print("  - Unsloth install failed → the script falls back automatically")
                    print("  - OOM → switch to t4-medium in launch_job.py (cheaper + slower)")
                    print("  - GitHub clone failed → check GITHUB_REPO is a public repo")
                break

            # Still running
            print(f"[stream] Job {status} — {seen_lines} log lines so far. "
                  f"Polling again in {poll_interval}s...", end="\r")
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\n\nStopped streaming. Job is still running.")
        print(f"Resume with: python stream_logs.py {job_id}")


if __name__ == "__main__":
    main()