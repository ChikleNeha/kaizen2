"""
agent/llm_agent.py
LLM-based agent for Project Kaizen.

Model
-----
Default : Qwen/Qwen2.5-3B-Instruct  (free, fits on T4 in 4-bit)
Override : set MODEL_NAME env var or pass model_name= to LLMAgent.__init__()
           At the hackathon, swap to a larger model by changing one line.

HF Credits upgrade path
-----------------------
At evaluation time with HF credits, change MODEL_NAME to e.g.:
  "Qwen/Qwen2.5-14B-Instruct"  or  "meta-llama/Llama-3.1-8B-Instruct"
Everything else in this file stays identical — the chat template and
parse_action() pipeline are model-agnostic.

After SFT / GRPO training, pass the local adapter path as model_name:
  agent = LLMAgent(model_name="./kaizen_grpo_model")
"""

import os
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agent.prompts import build_chat_messages
from environment.action_space import AgenticOSAction, parse_action

# ---------------------------------------------------------------------------
# Defaults — override via env var or constructor arg
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME: str = os.environ.get(
    "KAIZEN_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"
)

GENERATION_CONFIG: dict[str, Any] = {
    "max_new_tokens": 512,
    "temperature": 0.3,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    # Stop generation when the model closes the JSON object
    # (handled post-hoc via parse_action — no stop_strings needed)
}


# ---------------------------------------------------------------------------
# LLMAgent
# ---------------------------------------------------------------------------

class LLMAgent:
    """
    Wraps a HuggingFace causal LM and exposes a single act() method.

    The agent is stateless between calls — all context is provided via the
    obs dict on each call.  No conversation history is maintained; the
    system prompt + current observation is the full context every time.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path to a saved LoRA adapter directory.
    load_in_4bit : bool
        Quantise to 4-bit using bitsandbytes.  Required for T4 / free Colab.
        Set False if running on a machine with >= 24 GB VRAM.
    device_map : str
        Passed to from_pretrained().  "auto" handles multi-GPU automatically.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self._load_in_4bit = load_in_4bit
        self._device_map = device_map

        print(f"[LLMAgent] Loading model: {model_name}")
        print(f"[LLMAgent] 4-bit quantisation: {load_in_4bit}")

        self._tokenizer, self._model = self._load_model()
        print("[LLMAgent] Model ready.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load tokenizer and model.

        4-bit path  : uses BitsAndBytesConfig (nf4, double quant) — fits
                      Qwen2.5-3B comfortably in < 4 GB VRAM.
        Full-precision path : fp16 on CUDA, fp32 on CPU.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Ensure a pad token exists (Qwen models sometimes lack one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if self._load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map=self._device_map,
                trust_remote_code=True,
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self._device_map,
                trust_remote_code=True,
            )

        model.eval()
        return tokenizer, model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(
        self, obs: dict[str, Any]
    ) -> tuple[str, AgenticOSAction | None, str]:
        """
        Generate an action for the current observation.

        Parameters
        ----------
        obs : dict
            Current observation dict from KaizenEnv.

        Returns
        -------
        raw_output : str
            Full decoded LLM output (reasoning + JSON).
        action     : AgenticOSAction | None
            Validated action, or None if parsing failed.
        error      : str
            Empty string on success, error description on failure.
        """
        # Build chat messages and apply the model's chat template
        messages = build_chat_messages(obs)

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Move to same device as the model
        device = next(self._model.parameters()).device
        input_ids = input_ids.to(device)

        prompt_length = input_ids.shape[-1]

        t0 = time.monotonic()

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                do_sample=GENERATION_CONFIG["do_sample"],
                top_p=GENERATION_CONFIG["top_p"],
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        elapsed = time.monotonic() - t0

        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = output_ids[0][prompt_length:]
        raw_output = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        tokens_generated = len(new_tokens)
        print(
            f"[LLMAgent] Generated {tokens_generated} tokens "
            f"in {elapsed:.2f}s ({tokens_generated / elapsed:.1f} tok/s)"
        )

        # Validate output through the Pydantic parse layer
        action, error = parse_action(raw_output)

        if error:
            print(f"[LLMAgent] Parse error: {error}")
        else:
            print(f"[LLMAgent] Action: {action.tool_name}")  # type: ignore[union-attr]

        return raw_output, action, error

    # ------------------------------------------------------------------
    # Convenience: run a full episode
    # ------------------------------------------------------------------

    def run_episode(self, env: Any, render: bool = False) -> dict[str, Any]:
        """
        Run a complete episode with the given environment.

        Parameters
        ----------
        env    : KaizenEnv instance (already reset externally or reset here)
        render : bool — call env.render() after each step

        Returns
        -------
        dict with keys: episode, total_reward, steps, terminated, truncated
        """
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while not env.is_done:
            # Hide active_chaos from agent — partial observability
            from environment.observation_space import ObservationBuilder
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

    # ------------------------------------------------------------------
    # Model info helper (useful for HF Space UI)
    # ------------------------------------------------------------------

    def model_info(self) -> dict[str, Any]:
        """Return a summary dict for display in the dashboard / HF Space."""
        param_count = sum(p.numel() for p in self._model.parameters())
        return {
            "model_name": self.model_name,
            "parameters_approx": f"{param_count / 1e9:.1f}B",
            "quantisation": "4-bit NF4" if self._load_in_4bit else "fp16/fp32",
            "device": str(next(self._model.parameters()).device),
            "max_new_tokens": GENERATION_CONFIG["max_new_tokens"],
            "temperature": GENERATION_CONFIG["temperature"],
        }
