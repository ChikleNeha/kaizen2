"""
agent/llm_agent.py
Calls HF Serverless Inference API — no local GPU/CPU model loading needed.
"""

import json
import os
import re
import time
from typing import Any

import requests

from agent.prompts import SYSTEM_PROMPT, format_observation
from environment.action_space import AgenticOSAction, parse_action


class LLMAgent:
    DEFAULT_MODEL = "NehaChikle/kaizen-grpo"

    def __init__(self, model_name: str | None = None, max_new_tokens: int = 256, temperature: float = 0.3, **kwargs):
        self.model_name      = model_name or os.environ.get("KAIZEN_MODEL_NAME", self.DEFAULT_MODEL)
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.demo_mode       = False
        self._consecutive_failures = 0

        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"}

        print(f"[LLMAgent] ✅ Using HF Inference API → {self.model_name}")

    def act(self, obs: dict[str, Any]) -> tuple[str, AgenticOSAction | None, str]:
        observation_text = format_observation(obs)

        # Build prompt manually — no tokenizer needed
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{observation_text}<|im_end|>\n<|im_start|>assistant\n"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature":    self.temperature,
                "do_sample":      self.temperature > 0,
                "return_full_text": False,
            }
        }

        try:
            t0 = time.time()
            resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

            # 503 = model cold start — wait and retry once
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 20)
                print(f"[LLMAgent] Model loading, waiting {wait:.0f}s...")
                time.sleep(min(wait, 30))
                resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

            resp.raise_for_status()
            result = resp.json()

            if isinstance(result, list) and result:
                raw_output = result[0].get("generated_text", "")
            else:
                raw_output = str(result)

            elapsed = time.time() - t0
            print(f"[LLMAgent] Response in {elapsed:.2f}s")

        except Exception as e:
            print(f"[LLMAgent] API error: {e}")
            raw_output = '{"tool_name": "list_processes"}'

        repaired = self._repair_json(raw_output)
        action, error = parse_action(repaired)

        if action is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 2:
                print("[LLMAgent] 2 consecutive failures — forcing list_processes")
                self._consecutive_failures = 0
                action, error = parse_action('{"tool_name": "list_processes"}')
                raw_output += '\n\n[FALLBACK] {"tool_name": "list_processes"}'
        else:
            self._consecutive_failures = 0
            print(f"[LLMAgent] Action: {action.tool_name}")

        return repaired, action, error

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name":     self.model_name,
            "inference":      "HF Serverless API",
            "quantisation":   "server-side",
            "max_new_tokens": self.max_new_tokens,
            "temperature":    self.temperature,
        }

    # ── Keep _repair_json exactly as before ──────────────────────────
    def _repair_json(self, llm_output: str) -> str:
        json_str = self._extract_last_json(llm_output)
        if not json_str:
            start = llm_output.rfind('{')
            if start != -1:
                llm_output = llm_output + '"}'
                json_str = llm_output[start:]
            else:
                return llm_output
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            trimmed = re.sub(r',\s*"[^"]*"\s*:\s*[^,}\]]*$', '', json_str)
            if not trimmed.endswith('}'):
                trimmed += '}'
            try:
                data = json.loads(trimmed)
            except json.JSONDecodeError:
                return llm_output

        tool = data.get("tool_name", "")
        if tool == "prioritize_task":
            if "priority" not in data:
                data["priority"] = "normal"
            if data.get("priority") not in {"high", "normal", "low"}:
                data["priority"] = "normal"
            if "pid" in data:
                try:
                    data["pid"] = int(str(data["pid"]).strip().split()[0])
                except:
                    data = {"tool_name": "list_processes"}
        elif tool == "kill_process":
            if "pid" in data:
                try:
                    data["pid"] = int(str(data["pid"]).strip().split()[0])
                except:
                    data = {"tool_name": "inspect_logs", "pid": None}
            else:
                data = {"tool_name": "list_processes"}
            if data.get("tool_name") == "kill_process" and "reason" not in data:
                data["reason"] = "high resource usage detected"
        elif tool == "wait":
            if "reason" not in data:
                data["reason"] = "no immediate action required"
        elif tool not in {"kill_process","allocate_memory","thermal_mitigation","prioritize_task","inspect_logs","list_processes","wait"}:
            data = {"tool_name": "list_processes"}

        return json.dumps(data)

    def _extract_last_json(self, text: str) -> str:
        last_start = last_end = -1
        depth = 0
        in_string = escape = False
        candidate_start = -1
        for i, ch in enumerate(text):
            if escape: escape = False; continue
            if ch == '\\' and in_string: escape = True; continue
            if ch == '"': in_string = not in_string; continue
            if in_string: continue
            if ch == '{':
                if depth == 0: candidate_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_start, last_end = candidate_start, i
        if last_start == -1: return ""
        return text[last_start:last_end + 1]