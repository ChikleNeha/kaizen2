"""
generate_baseline.py
====================
Uses DemoAgent (rule-based, no GPU) to generate a baseline reward curve.
This is the "before training" comparison for the submission.

Run from project root:
    python generate_baseline.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.kaizen_env import KaizenEnv
from agent.demo_agent import DemoAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_EPISODES = 50
OUTPUT_PATH  = "training/baseline_reward.png"
SEED         = 42

# ---------------------------------------------------------------------------
# Run DemoAgent episodes
# ---------------------------------------------------------------------------

def run_baseline():
    agent = DemoAgent()
    episode_rewards = []

    print(f"[BASELINE] Running {NUM_EPISODES} DemoAgent episodes...")

    for ep in range(NUM_EPISODES):
        env = KaizenEnv(broadcast=False)
        result = agent.run_episode(env, render=False)
        episode_rewards.append(result["total_reward"])

        if (ep + 1) % 10 == 0:
            avg = sum(episode_rewards[-10:]) / 10
            print(
                f"[BASELINE] Episode {ep+1:03d} | "
                f"Steps: {result['steps']} | "
                f"Reward: {result['total_reward']:+.3f} | "
                f"10-ep avg: {avg:+.3f}"
            )

    print(f"\n[BASELINE] Mean reward : {sum(episode_rewards)/len(episode_rewards):+.3f}")
    print(f"[BASELINE] Min reward  : {min(episode_rewards):+.3f}")
    print(f"[BASELINE] Max reward  : {max(episode_rewards):+.3f}")

    return episode_rewards

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_plot(rewards, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        window = 10
        running_avg = [
            sum(rewards[max(0, i - window + 1):i + 1]) / min(window, i + 1)
            for i in range(len(rewards))
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(rewards, color="#888888", linewidth=1.0,
                alpha=0.6, label="Episode reward (DemoAgent / rule-based)")
        ax.plot(running_avg, color="#f87171", linewidth=2.0,
                linestyle="--", label=f"{window}-ep running avg (baseline)")
        ax.axhline(y=0, color="#444444", linewidth=0.5, linestyle=":")

        ax.set_facecolor("#0f0f0f")
        fig.patch.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#888888")
        ax.spines[:].set_color("#1f1f1f")
        ax.set_xlabel("Episode", color="#888888")
        ax.set_ylabel("Total Episode Reward", color="#888888")
        ax.set_title(
            "Kaizen OS — Baseline (Rule-Based DemoAgent, Pre-Training)",
            color="#e2e2e2", pad=12
        )
        ax.legend(facecolor="#161616", labelcolor="#e2e2e2", framealpha=0.8)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[BASELINE] Plot saved → {path}")

    except ImportError:
        json_path = path.replace(".png", ".json")
        with open(json_path, "w") as f:
            json.dump({"baseline_rewards": rewards}, f, indent=2)
        print(f"[BASELINE] matplotlib missing. Data saved → {json_path}")
        print("[BASELINE] pip install matplotlib  then re-run.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rewards = run_baseline()
    save_plot(rewards, OUTPUT_PATH)

    print(f"\n[BASELINE] Next steps:")
    print(f"  git add {OUTPUT_PATH}")
    print(f"  git commit -m 'Add baseline DemoAgent reward curve (pre-training)'")
    print(f"  git push")