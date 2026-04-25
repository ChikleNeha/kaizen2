"""
agent/llm_agent.py
LLM inference agent for Project Kaizen.

Loads a Qwen2.5-3B-Instruct model (base or LoRA-adapted) and runs inference
to produce structured actions from system observations.

Speed expectations
------------------
Local CPU (M1/M2/x86)  : 0.2 – 0.5 tok/s  →  unusable for live demo
Colab T4 (free)        : 8  – 12  tok/s  →  ~45s per step, acceptable
Colab A100 (credits)   : 40 – 60  tok/s  →  ~8s per step, demo-ready

Always run on GPU for the hackathon demo.

Robustness additions (v2)
--------------------------
- _repair_json()        : fixes the most common GRPO model output errors
                          (missing priority field, string pid, truncated JSON)
- consecutive_failures  : after 2 consecutive parse failures, forces a safe
                          list_processes fallback instead of wasting steps
- GPU verification      : prints device + dtype on load so you know immediately
                          if the model landed on CPU by mistake
- for_inference()       : calls Unsloth's inference optimisation if available
"""

import json
import re
import time
from typing import Any


from agent.prompts import SYSTEM_PROMPT, format_observation
from environment.action_space import AgenticOSAction, parse_action


class LLMAgent:
    """
    Wraps a causal LLM for OS management action generation.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path to a LoRA adapter directory.
        Defaults to the base Qwen2.5-3B-Instruct model.
    max_new_tokens : int
        Maximum tokens to generate per step.  256 is enough for
        3-sentence reasoning + JSON action and avoids clipping.
    temperature : float
        Sampling temperature.  0.3 for focused/deterministic output.
    use_unsloth : bool
        If True, attempt to load via Unsloth for 2x faster inference.
        Falls back to plain transformers if Unsloth is not installed.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        use_unsloth: bool = True,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._consecutive_failures: int = 0

        import torch
        self._torch = torch

        print(f"[LLMAgent] Loading model: {self.model_name}")
        print(f"[LLMAgent] max_new_tokens={max_new_tokens} | temperature={temperature}")

        self.model, self.tokenizer = self._load_model(use_unsloth)

        # Verify the model landed on GPU — critical for demo speed
        device = next(self.model.parameters()).device
        dtype  = next(self.model.parameters()).dtype
        print(f"[LLMAgent] Device : {device}")
        print(f"[LLMAgent] dtype  : {dtype}")

        if str(device) == "cpu":
            print(
                "[LLMAgent] WARNING: model is on CPU — inference will be very slow (~0.5 tok/s). "
                "Run on a GPU machine for the hackathon demo."
            )
        else:
            print(f"[LLMAgent] GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, use_unsloth: bool):
        """
        Load model + tokenizer.  Tries Unsloth first, falls back to
        plain transformers if Unsloth is not installed or load fails.
        """
        if use_unsloth:
            try:
                return self._load_with_unsloth()
            except Exception as e:
                print(f"[LLMAgent] Unsloth load failed ({e}), falling back to transformers.")

        return self._load_with_transformers()

    def _load_with_unsloth(self):
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            dtype=None,          # auto-detect bf16/fp16
            load_in_4bit=True,
        )
        # Unsloth inference optimisation — 2x faster generation
        FastLanguageModel.for_inference(model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("[LLMAgent] Loaded via Unsloth (4-bit, inference-optimised).")
        return model, tokenizer

    def _load_with_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

        bnb_config = None
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("[LLMAgent] bitsandbytes not available — loading in fp16.")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        model.eval()

        print("[LLMAgent] Loaded via transformers.")
        return model, tokenizer

    # ------------------------------------------------------------------
    # JSON repair
    # ------------------------------------------------------------------

    def _repair_json(self, llm_output: str) -> str:
        """
        Attempt to repair common GRPO model output errors before Pydantic
        validation.

        Handles:
        - prioritize_task: missing 'priority' field → default 'normal'
        - prioritize_task / kill_process: non-integer pid → parse or fallback
        - kill_process: missing 'reason' field → add default
        - Truncated JSON (no closing brace) → attempt to close it
        - Empty JSON body → replace with list_processes

        Returns the original string if the JSON block cannot be found or
        repaired.  The caller always falls back to parse_action() with
        whatever string is returned.
        """
        start = llm_output.find('{')
        end   = llm_output.rfind('}')

        # Attempt to close truncated JSON
        if start != -1 and end == -1:
            llm_output = llm_output + '"}'
            end = llm_output.rfind('}')

        if start == -1 or end == -1:
            return llm_output

        json_str = llm_output[start:end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try stripping trailing incomplete key-value pairs
            # e.g. {"tool_name": "kill_process", "pid": 2847, "rea  ← truncated
            trimmed = re.sub(r',\s*"[^"]*"\s*:\s*[^,}\]]*$', '', json_str)
            if not trimmed.endswith('}'):
                trimmed += '}'
            try:
                data = json.loads(trimmed)
            except json.JSONDecodeError:
                return llm_output  # give up, let parse_action handle the error

        tool = data.get("tool_name", "")

        # ---- prioritize_task fixes ------------------------------------
        if tool == "prioritize_task":
            # Fix missing priority
            if "priority" not in data:
                data["priority"] = "normal"
            # Fix invalid priority value
            valid_priorities = {"high", "normal", "low"}
            if data.get("priority") not in valid_priorities:
                data["priority"] = "normal"
            # Fix non-integer pid
            if "pid" in data:
                try:
                    data["pid"] = int(str(data["pid"]).strip().split()[0])
                except (ValueError, IndexError, AttributeError):
                    # Cannot fix pid — downgrade to list_processes
                    data = {"tool_name": "list_processes"}

        # ---- kill_process fixes ---------------------------------------
        elif tool == "kill_process":
            # Fix non-integer pid
            if "pid" in data:
                try:
                    data["pid"] = int(str(data["pid"]).strip().split()[0])
                except (ValueError, IndexError, AttributeError):
                    # Cannot fix — safer to inspect logs first
                    data = {"tool_name": "inspect_logs", "pid": None}
            else:
                data = {"tool_name": "list_processes"}
            # Fix missing reason
            if data.get("tool_name") == "kill_process" and "reason" not in data:
                data["reason"] = "high resource usage detected"

        # ---- allocate_memory fixes ------------------------------------
        elif tool == "allocate_memory":
            if "target_pid" in data:
                try:
                    data["target_pid"] = int(str(data["target_pid"]).strip().split()[0])
                except (ValueError, IndexError, AttributeError):
                    data = {"tool_name": "list_processes"}
            if "mb_to_free" in data:
                try:
                    mb = float(data["mb_to_free"])
                    # Clamp to valid range (0, 512]
                    data["mb_to_free"] = max(1.0, min(512.0, mb))
                except (ValueError, TypeError):
                    data["mb_to_free"] = 128.0

        # ---- wait fix -------------------------------------------------
        elif tool == "wait":
            if "reason" not in data:
                data["reason"] = "no immediate action required"

        # ---- unknown tool_name ----------------------------------------
        elif tool not in {
            "kill_process", "allocate_memory", "thermal_mitigation",
            "prioritize_task", "inspect_logs", "list_processes", "wait",
        }:
            # Replace with a safe fallback
            data = {"tool_name": "list_processes"}

        repaired_json = json.dumps(data)
        return llm_output[:start] + repaired_json + llm_output[end + 1:]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, obs: dict[str, Any]) -> tuple[str, AgenticOSAction | None, str]:
        """
        Generate an action from the current observation.

        Parameters
        ----------
        obs : dict
            Observation dict from KaizenEnv (agent-facing, active_chaos hidden).

        Returns
        -------
        raw_output : str   — full LLM output (reasoning + JSON)
        action     : AgenticOSAction | None
        error      : str   — empty string on success
        """
        observation_text = format_observation(obs)

        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": observation_text},
        ]

        # Format using the model's chat template
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # apply_chat_template may return a BatchEncoding dict or a raw tensor
        # depending on the transformers version — handle both
        if isinstance(tokenized, torch.Tensor):
            input_ids = tokenized
        else:
            input_ids = tokenized["input_ids"]  # unwrap BatchEncoding

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        prompt_len = input_ids.shape[-1]

        t0 = time.time()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,   # reduces looping / degenerate output
            )

        elapsed = time.time() - t0
        new_tokens = output_ids.shape[-1] - prompt_len
        tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        print(
            f"[LLMAgent] Generated {new_tokens} tokens in {elapsed:.2f}s "
            f"({tok_per_sec:.1f} tok/s)"
        )

        # Decode only the newly generated tokens
        raw_output = self.tokenizer.decode(
            output_ids[0][prompt_len:],
            skip_special_tokens=True,
        ).strip()

        # Attempt JSON repair before validation
        repaired_output = self._repair_json(raw_output)

        # Parse and validate with Pydantic
        action, error = parse_action(repaired_output)

        # Consecutive failure fallback — after 2 failures, force list_processes
        # so the agent gathers information instead of burning steps on parse errors
        if action is None:
            self._consecutive_failures += 1
            print(f"[LLMAgent] Parse error: {error}")

            if self._consecutive_failures >= 2:
                print(
                    f"[LLMAgent] {self._consecutive_failures} consecutive parse failures — "
                    "forcing list_processes fallback."
                )
                self._consecutive_failures = 0
                fallback_json = '{"tool_name": "list_processes"}'
                action, error = parse_action(fallback_json)
                # Append the fallback to raw_output so the dashboard shows it
                raw_output = raw_output + f"\n\n[FALLBACK] {fallback_json}"
                repaired_output = raw_output
        else:
            self._consecutive_failures = 0
            print(f"[LLMAgent] Action: {action.tool_name}")

        return repaired_output, action, error

    # ------------------------------------------------------------------
    # Episode runner (used by server/main.py and demo scripts)
    # ------------------------------------------------------------------

    def run_episode(
        self, env: Any, render: bool = False
    ) -> dict[str, Any]:
        """
        Run a full episode using this agent.

        Parameters
        ----------
        env    : KaizenEnv instance (already constructed)
        render : if True, call env.render() after each step

        Returns
        -------
        dict with episode summary: total_reward, steps, terminated, truncated
        """
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        from environment.observation_space import ObservationBuilder  # local import

        while not env.is_done:
            agent_obs = ObservationBuilder.to_agent_obs(obs)
            raw_output, action, error = self.act(agent_obs)
            obs, reward, terminated, truncated, info = env.step(raw_output)
            total_reward += reward
            steps += 1

            if render:
                env.render()

        return {
            "episode":       info.get("episode", env.episode),
            "total_reward":  round(total_reward, 3),
            "steps":         steps,
            "terminated":    terminated,
            "truncated":     truncated,
        }

    # ------------------------------------------------------------------
    # Metadata (used by server/main.py status endpoint)
    # ------------------------------------------------------------------

    def model_info(self) -> dict[str, Any]:
        device = str(next(self.model.parameters()).device)
        dtype  = str(next(self.model.parameters()).dtype)
        return {
            "model_name":         self.model_name,
            "parameters_approx":  "3B",
            "quantisation":       "4-bit NF4",
            "device":             device,
            "dtype":              dtype,
            "max_new_tokens":     self.max_new_tokens,
            "temperature":        self.temperature,
        }