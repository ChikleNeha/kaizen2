"""
environment/kaizen_env.py
Main Gymnasium environment for Project Kaizen.

Episode flow
------------
step 0-2  : clean system — agent explores, builds context
step 3    : ChaosInjector fires; a rogue process appears
step 4-9  : agent must identify and neutralise the chaos process
            episode terminates early if chaos is resolved

Termination vs truncation
-------------------------
terminated = True  when chaos is resolved OR max steps reached after chaos
truncated  = True  when max steps reached WITHOUT resolving chaos
             (per Gymnasium convention: truncated means time-limit cut-off)

WebSocket broadcasting
----------------------
After every step a JSON payload matching the schema in Section 4.1 is
broadcast to all connected clients.  Pass broadcast=False to disable
(used during GRPO training to avoid needing a running server).
"""

import asyncio
import time
from typing import Any

import gymnasium as gym
import numpy as np

from environment.action_space import AgenticOSAction, parse_action
from environment.chaos import ChaosInjector
from environment.observation_space import ObservationBuilder
from environment.reward import compute_reward
from environment.sandbox import SandboxExecutor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS: int = 10
CHAOS_INJECT_STEP: int = 3   # chaos fires at the START of this step


# ---------------------------------------------------------------------------
# KaizenEnv
# ---------------------------------------------------------------------------

class KaizenEnv(gym.Env):
    """
    Gymnasium environment that models an OS under stress.

    Parameters
    ----------
    broadcast : bool
        Broadcast state over WebSocket after every step.
        Set False during training.
    use_docker : bool
        Forward actions to a Docker sandbox instead of the simulated executor.
    broadcaster : ConnectionManager | None
        Pre-built broadcaster from server/broadcast.py.  When None and
        broadcast=True the environment will attempt to import and use the
        module-level manager from server.broadcast.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        broadcast: bool = True,
        use_docker: bool = False,
        broadcaster: Any = None,
    ) -> None:
        super().__init__()

        self._broadcast_enabled = broadcast
        self._broadcaster = broadcaster  # injected from server/main.py

        # Sub-components
        self._obs_builder = ObservationBuilder()
        self._chaos = ChaosInjector()
        self._sandbox = SandboxExecutor(use_docker=use_docker)

        # Episode state
        self._current_obs: dict[str, Any] = {}
        self._protected_pids: set[int] = set()
        self._step: int = 0
        self._episode: int = 0
        self._cumulative_reward: float = 0.0
        self._reward_history: list[float] = []
        self._terminated: bool = False
        self._truncated: bool = False

        # Last step metadata (used for broadcasting)
        self._last_agent_reasoning: str = ""
        self._last_action_dict: dict[str, Any] = {}
        self._last_action_result: dict[str, Any] = {}
        self._last_reward: float = 0.0
        self._chaos_was_active_before_step: bool = False

        # Gymnasium spaces — using Dict / Box is verbose for an LLM env;
        # we use a simple Text observation space and a Discrete placeholder.
        # The real obs/action exchange is via string ↔ string.
        self.observation_space = gym.spaces.Text(min_length=0, max_length=8192)
        self.action_space = gym.spaces.Text(min_length=0, max_length=2048)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Returns
        -------
        obs  : dict   — initial observation
        info : dict   — metadata (episode number, protected PIDs)
        """
        super().reset(seed=seed)

        self._episode += 1
        self._step = 0
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._terminated = False
        self._truncated = False
        self._last_agent_reasoning = ""
        self._last_action_dict = {}
        self._last_action_result = {}
        self._last_reward = 0.0
        self._chaos_was_active_before_step = False

        # Reset sub-components
        self._obs_builder.reset()
        self._chaos.reset()

        # Collect initial observation
        self._obs_builder.set_step(self._step)
        self._current_obs = self._obs_builder.build()

        # Record protected PIDs for this episode
        self._protected_pids = ObservationBuilder.extract_protected_pids(
            self._current_obs
        )

        info = {
            "episode": self._episode,
            "protected_pids": list(self._protected_pids),
        }
        return self._current_obs, info

    def step(
        self, action_str: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action_str : str
            Raw LLM output string.  Parsed and validated internally.

        Returns
        -------
        obs_after  : dict
        reward     : float
        terminated : bool
        truncated  : bool
        info       : dict
        """
        if self._terminated or self._truncated:
            raise RuntimeError(
                "step() called on a finished episode. Call reset() first."
            )

        self._step += 1
        self._obs_builder.set_step(self._step)

        # ---- 1. Inject chaos at step 3 --------------------------------
        if self._step == CHAOS_INJECT_STEP and not self._chaos.is_active:
            self._current_obs = self._chaos.inject(self._current_obs)
            self._obs_builder.set_log_snippet(self._current_obs["log_snippet"])

        obs_before = self._current_obs.copy()
        # Track whether chaos was active BEFORE this step (used for termination)
        self._chaos_was_active_before_step = self._chaos.is_active
        # Agent receives a partially-observable view: active_chaos hidden.
        # Full obs is retained for reward computation and termination.
        agent_obs = ObservationBuilder.to_agent_obs(obs_before)

        # The LLM output contains chain-of-thought text followed by JSON.
        # We store everything before the first '{' as the reasoning string.
        reasoning_end = action_str.find('{')
        if reasoning_end > 0:
            self._last_agent_reasoning = action_str[:reasoning_end].strip()
        else:
            self._last_agent_reasoning = ""

        action, parse_error = parse_action(action_str)

        # ---- 3. Execute action in sandbox (only if parse succeeded) ---
        if action is not None:
            result = self._sandbox.execute(action, obs_before)
            self._last_action_result = result

            # Merge obs_delta into current observation
            obs_after = obs_before.copy()
            obs_after.update(result.get("obs_delta", {}))

            # Let the chaos injector check whether its process was killed
            if action.__class__.__name__ == "KillProcessAction":
                obs_after, extra_penalty = self._chaos.resolve(obs_after, action.pid)
                # extra_penalty is non-zero only for semantic_decoy wrong kills
                if extra_penalty != 0.0:
                    self._last_action_result["message"] += (
                        f" | SEMANTIC ERROR: wrong process killed (penalty {extra_penalty:.1f})"
                    )
            else:
                extra_penalty = 0.0

        else:
            # Parse failed — observation unchanged, record error
            obs_after = obs_before.copy()
            self._last_action_result = {
                "success": False,
                "message": f"Parse error: {parse_error}",
                "obs_delta": {},
            }
            extra_penalty = 0.0

        # Ensure step counter is consistent in obs_after
        obs_after["step"] = self._step
        self._current_obs = obs_after

        # ---- 4. Compute reward ----------------------------------------
        reward = compute_reward(
            obs_before=obs_before,
            obs_after=obs_after,
            action=action,
            action_error=parse_error,
            protected_pids=self._protected_pids,
        )
        # Add semantic wrong-kill penalty if applicable
        if action is not None and extra_penalty != 0.0:
            reward = round(reward + extra_penalty, 3)

        self._last_reward = reward
        self._cumulative_reward = round(self._cumulative_reward + reward, 3)
        self._reward_history.append(self._cumulative_reward)

        # ---- 5. Termination logic -------------------------------------
        # Use internal chaos state (not obs field) because active_chaos
        # is hidden from the agent-facing obs for partial observability.
        chaos_was_active = self._chaos_was_active_before_step
        chaos_resolved = chaos_was_active and not self._chaos.is_active
        max_steps_reached = self._step >= MAX_STEPS

        self._terminated = chaos_resolved or max_steps_reached
        self._truncated = max_steps_reached and not chaos_resolved

        # ---- 6. Capture action dict for broadcast --------------------
        if action is not None:
            self._last_action_dict = action.model_dump()
        else:
            self._last_action_dict = {"tool_name": "parse_error", "error": parse_error}

        # ---- 7. Broadcast -------------------------------------------
        if self._broadcast_enabled:
            self._schedule_broadcast()

        info = {
            "episode": self._episode,
            "parse_error": parse_error,
            "action_result": self._last_action_result,
            "chaos_active": self._current_obs.get("active_chaos"),
            "cumulative_reward": self._cumulative_reward,
        }

        return self._current_obs, reward, self._terminated, self._truncated, info

    def render(self) -> None:
        """Print a compact text summary of the current state."""
        obs = self._current_obs
        print(
            f"[Ep {self._episode:03d} | Step {self._step:02d}/{MAX_STEPS}] "
            f"CPU={obs.get('cpu_percent', 0):.1f}%  "
            f"RAM={obs.get('ram_percent', 0):.1f}%  "
            f"Thermal={obs.get('thermal_celsius', 0):.1f}°C  "
            f"Chaos={obs.get('active_chaos') or 'None'}  "
            f"Reward={self._last_reward:+.3f}  "
            f"Cumulative={self._cumulative_reward:+.3f}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def _build_broadcast_payload(self) -> dict[str, Any]:
        """Construct the exact WebSocket payload schema from Section 4.1."""
        return {
            "type": "state_update",
            "episode": self._episode,
            "step": self._step,
            "max_steps": MAX_STEPS,
            "obs": self._current_obs,
            "agent_reasoning": self._last_agent_reasoning,
            "action": self._last_action_dict,
            "action_result": self._last_action_result,
            "reward": self._last_reward,
            "cumulative_reward": self._cumulative_reward,
            "reward_history": self._reward_history,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }

    def _schedule_broadcast(self) -> None:
        """
        Schedule a WebSocket broadcast without blocking the env step.

        Uses asyncio.get_event_loop() so this works whether the caller is
        inside an async context (FastAPI) or a sync context (training loop).
        If no broadcaster is injected, attempts to import the module-level
        manager from server.broadcast.
        """
        broadcaster = self._broadcaster
        if broadcaster is None:
            try:
                from server.broadcast import manager as default_manager
                broadcaster = default_manager
            except ImportError:
                return  # server not running — silently skip

        payload = self._build_broadcast_payload()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(broadcaster.broadcast(payload))
            else:
                loop.run_until_complete(broadcaster.broadcast(payload))
        except RuntimeError:
            # No event loop in this thread — create one for this call
            asyncio.run(broadcaster.broadcast(payload))

    # ------------------------------------------------------------------
    # Properties (used by server/main.py)
    # ------------------------------------------------------------------

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def current_obs(self) -> dict[str, Any]:
        return self._current_obs

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def is_done(self) -> bool:
        return self._terminated or self._truncated
