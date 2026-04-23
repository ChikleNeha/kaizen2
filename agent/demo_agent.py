"""
agent/demo_agent.py
Rule-based dummy agent for fast local dashboard testing.

Activated by setting KAIZEN_DEMO_MODE=true.
Does NOT require model download or GPU.
Generates realistic-looking reasoning and correct actions instantly.
Perfect for testing the dashboard UI and WebSocket pipeline locally.

At hackathon: always use the real LLMAgent (KAIZEN_DEMO_MODE=false).
"""

import time
import random
from typing import Any

from environment.action_space import parse_action


class DemoAgent:
    """
    Instant rule-based agent that mimics LLM output format.
    Used only for local dashboard testing — not for training or demo.
    """

    def __init__(self):
        print("[DemoAgent] Running in DEMO MODE — no LLM loaded.")
        print("[DemoAgent] Set KAIZEN_DEMO_MODE=false to use the real model.")

    def act(self, obs: dict[str, Any]) -> tuple[str, Any, str]:
        """
        Generate a realistic action for the observation.
        Mimics the LLM output format: reasoning text + JSON action.
        """
        # Simulate thinking time
        time.sleep(0.5)

        process_list = obs.get("process_list", [])
        cpu = obs.get("cpu_percent", 0)
        thermal = obs.get("thermal_celsius", 0)
        log = obs.get("log_snippet", "")
        step = obs.get("step", 0)

        # Find the most suspicious non-protected process
        killable = [p for p in process_list if not p.get("is_protected", False)]
        killable_sorted = sorted(killable, key=lambda p: p["cpu_percent"], reverse=True)

        # Check logs for named PIDs
        import re
        log_pid_match = re.search(r'\[(\d+)\]', log)
        log_pid = int(log_pid_match.group(1)) if log_pid_match else None

        # Decision logic
        if cpu < 40 and thermal < 70:
            reasoning = (
                f"Step {step}: System metrics look healthy. "
                f"CPU at {cpu:.1f}%, thermal at {thermal:.1f}°C. "
                f"No anomalies in the log. Waiting."
            )
            action_json = '{"tool_name": "wait", "reason": "System is operating within normal parameters."}'

        elif log_pid and any(p["pid"] == log_pid for p in killable):
            target = next(p for p in killable if p["pid"] == log_pid)
            reasoning = (
                f"Step {step}: Elevated metrics detected. CPU {cpu:.1f}%, thermal {thermal:.1f}°C. "
                f"System log references PID {log_pid} ({target['name']}) as the problematic process. "
                f"This process is using {target['cpu_percent']:.1f}% CPU and is not protected. "
                f"Killing PID {log_pid} to resolve the anomaly."
            )
            target_name = target["name"]
            target_cpu = target["cpu_percent"]
            action_json = (
                f'{{"tool_name": "kill_process", "pid": {log_pid}, '
                f'"reason": "{target_name} identified in system log as cause of elevated metrics. '
                f'CPU {target_cpu:.1f}%, not protected."}}'
            )

        elif killable_sorted and killable_sorted[0]["cpu_percent"] > 40:
            target = killable_sorted[0]
            reasoning = (
                f"Step {step}: CPU at {cpu:.1f}%, thermal at {thermal:.1f}°C — elevated. "
                f"Scanning process list. {target['name']} (PID {target['pid']}) "
                f"is consuming {target['cpu_percent']:.1f}% CPU and is not protected. "
                f"Inspecting logs before taking action."
            )
            action_json = f'{{"tool_name": "inspect_logs", "pid": {target["pid"]}}}'

        elif thermal > 80:
            reasoning = (
                f"Step {step}: Thermal at {thermal:.1f}°C is elevated. "
                f"No single obvious process responsible. Applying thermal mitigation."
            )
            action_json = '{"tool_name": "thermal_mitigation", "strategy": "throttle_cpu"}'

        else:
            reasoning = (
                f"Step {step}: Metrics slightly elevated but no clear threat. "
                f"Listing processes to get a full picture."
            )
            action_json = '{"tool_name": "list_processes"}'

        full_output = f"{reasoning}\n\n{action_json}"
        action, error = parse_action(full_output)
        return full_output, action, error

    def run_episode(self, env: Any, render: bool = False) -> dict[str, Any]:
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        from environment.observation_space import ObservationBuilder
        while not env.is_done:
            agent_obs = ObservationBuilder.to_agent_obs(obs)
            raw_output, action, error = self.act(agent_obs)
            obs, reward, terminated, truncated, info = env.step(raw_output)
            total_reward += reward
            steps += 1
            if render:
                env.render()

        return {
            "episode": info.get("episode", env.episode),
            "total_reward": round(total_reward, 3),
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": "DemoAgent (rule-based)",
            "parameters_approx": "0B",
            "quantisation": "none",
            "device": "cpu",
            "max_new_tokens": 0,
            "temperature": 0.0,
        }
