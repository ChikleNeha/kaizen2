"""
environment/chaos.py
Chaos event injector for KaizenEnv.

Chaos events are purely simulated — they inject a fake process entry into
the observation dict and perturb metric values.  No real OS processes are
created or modified.  The ChaosInjector is owned by KaizenEnv and called
at step 3 of each episode.

Partial Observability (v2)
--------------------------
The ``active_chaos`` field is HIDDEN from the agent's observation.
It is always set to None in the obs dict the agent receives.
The agent must infer the chaos type from:
  - cpu_percent / thermal_celsius / ram_percent spikes
  - log_snippet content
  - process_list anomalies
This is the key novelty over rule-based systems: a rule sees "cpu > 85%
→ kill top process". The agent reads "hospital_monitor.exe is at 45% CPU
and the log says it's mid-sync — don't kill it, kill the renderer."

Scenario catalogue (6 total)
-----------------------------
Original (3):
  memory_leak, cpu_hog, thermal_spike

New (3):
  zombie_storm     — 100 zombie child processes dragging scheduler
  fork_bomb        — exponential child spawning, all same CPU%, must read
                     logs to identify parent
  semantic_decoy   — THE WINNING SCENARIO: two processes at identical CPU%.
                     One is hospital_monitor (critical), one is ad_renderer
                     (killable). Only the log reveals which to kill.
                     NO rule-based system can solve this.
"""

import random
from typing import Any

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

CHAOS_SCENARIOS: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------------ #
    # Original 3
    # ------------------------------------------------------------------ #
    "memory_leak": {
        "fake_process": {
            "pid": 2847,
            "name": "memory_leak_sim",
            "cpu_percent": 67.4,
            "memory_mb": 512.0,
            "status": "running",
            "is_protected": False,
        },
        "cpu_delta": 45.0,
        "thermal_delta": 28.0,
        "ram_delta": 30.0,
        "log": (
            "ERROR [2847]: memory allocation exceeded threshold. "
            "RSS growing unbounded. Consider terminating process."
        ),
        # chaos_pid is the PID that must be killed to resolve
        "resolution_pid": 2847,
    },
    "cpu_hog": {
        "fake_process": {
            "pid": 3991,
            "name": "rogue_compute",
            "cpu_percent": 89.2,
            "memory_mb": 128.0,
            "status": "running",
            "is_protected": False,
        },
        "cpu_delta": 60.0,
        "thermal_delta": 35.0,
        "ram_delta": 5.0,
        "log": (
            "WARNING [3991]: CPU usage sustained above 85% for 120s. "
            "Process may be stuck in infinite loop."
        ),
        "resolution_pid": 3991,
    },
    "thermal_spike": {
        "fake_process": {
            "pid": 4420,
            "name": "thermal_stress",
            "cpu_percent": 45.0,
            "memory_mb": 64.0,
            "status": "running",
            "is_protected": False,
        },
        "cpu_delta": 20.0,
        "thermal_delta": 42.0,
        "ram_delta": 8.0,
        "log": (
            "CRITICAL: Core temperature 91°C. Thermal throttling imminent. "
            "Background workloads should be suspended."
        ),
        "resolution_pid": 4420,
    },

    # ------------------------------------------------------------------ #
    # New scenario 1: zombie_storm
    # A process leaked 100 zombie children. The scheduler is saturated
    # with reaping overhead. CPU shows moderate elevation, RAM is fine,
    # but the system feels sluggish. The log names the zombie parent.
    # Rule-based: impossible — zombie count isn't a standard metric.
    # Agent: reads log, identifies parent PID, kills it to reap children.
    # ------------------------------------------------------------------ #
    "zombie_storm": {
        "fake_process": {
            "pid": 5512,
            "name": "orphan_manager",
            "cpu_percent": 38.4,
            "memory_mb": 96.0,
            "status": "zombie",
            "is_protected": False,
        },
        "cpu_delta": 35.0,
        "thermal_delta": 18.0,
        "ram_delta": 12.0,
        "log": (
            "WARNING [5512]: orphan_manager has spawned 100 zombie children. "
            "Process table near capacity. Scheduler overhead critical. "
            "Terminate parent PID 5512 to trigger zombie reaping."
        ),
        "resolution_pid": 5512,
    },

    # ------------------------------------------------------------------ #
    # New scenario 2: fork_bomb
    # A process is exponentially spawning children. All children share
    # the same name and similar CPU%. The log names the parent PID.
    # Rule-based: would kill the wrong child (highest CPU, wrong target).
    # Agent: reads log, identifies the PARENT (not the busiest child).
    # ------------------------------------------------------------------ #
    "fork_bomb": {
        "fake_process": {
            "pid": 6631,
            "name": "fork_worker",
            "cpu_percent": 44.1,
            "memory_mb": 256.0,
            "status": "running",
            "is_protected": False,
        },
        # Decoy child — same name, higher CPU — rule would kill this one
        "decoy_process": {
            "pid": 6700,
            "name": "fork_worker",
            "cpu_percent": 71.3,
            "memory_mb": 128.0,
            "status": "running",
            "is_protected": False,
        },
        "cpu_delta": 55.0,
        "thermal_delta": 30.0,
        "ram_delta": 20.0,
        "log": (
            "CRITICAL [6631]: fork_worker (PID 6631) is the fork bomb parent. "
            "Spawning children exponentially. Child count: 47 and growing. "
            "Kill parent PID 6631 to halt propagation. "
            "WARNING: killing children individually will not stop the bomb."
        ),
        "resolution_pid": 6631,
    },

    # ------------------------------------------------------------------ #
    # New scenario 3: semantic_decoy — THE WINNING DEMO SCENARIO
    #
    # Two processes at IDENTICAL CPU usage (~45%).
    # hospital_monitor: critical clinical sync process — DO NOT KILL.
    # ad_renderer:      background ad engine — KILL THIS ONE.
    #
    # A rule-based system CANNOT distinguish them by metrics alone.
    # The only signal is the log, which says:
    #   "hospital_monitor is in active patient data sync — preserve."
    #   "ad_renderer is non-critical background process — safe to terminate."
    #
    # This is the proof point: LLM > cgroups/systemd/nice.
    # The agent must read the log to make the correct semantic decision.
    # ------------------------------------------------------------------ #
    "semantic_decoy": {
        # The KILLABLE process — ad_renderer
        "fake_process": {
            "pid": 7741,
            "name": "ad_renderer",
            "cpu_percent": 45.2,
            "memory_mb": 180.0,
            "status": "running",
            "is_protected": False,
        },
        # The PROTECTED-BY-CONTEXT process — same CPU, must NOT be killed
        "decoy_process": {
            "pid": 7700,
            "name": "hospital_monitor",
            "cpu_percent": 44.8,    # nearly identical CPU to ad_renderer
            "memory_mb": 192.0,
            "status": "running",
            "is_protected": False,  # not system-protected, but semantically critical
        },
        "cpu_delta": 38.0,
        "thermal_delta": 22.0,
        "ram_delta": 15.0,
        "log": (
            "INFO [7700]: hospital_monitor — active patient data sync in progress. "
            "Interruption may cause data loss. DO NOT terminate during sync window. "
            "INFO [7741]: ad_renderer — non-critical background advertising engine. "
            "Safe to terminate at any time. Recommend killing to reduce thermal load."
        ),
        # Must kill ad_renderer (7741), NOT hospital_monitor (7700)
        "resolution_pid": 7741,
        # Killing the wrong one (hospital_monitor) triggers a catastrophic penalty
        "wrong_kill_pid": 7700,
        "wrong_kill_penalty": -15.0,
    },
}


# ---------------------------------------------------------------------------
# ChaosInjector
# ---------------------------------------------------------------------------

class ChaosInjector:
    """
    Manages a single active chaos event per episode.

    Key change in v2: active_chaos is HIDDEN from the agent.
    The obs dict the agent sees always has active_chaos=None.
    The internal _active_chaos field is used only by the environment
    for reward computation and resolution detection.

    Usage
    -----
    injector = ChaosInjector()
    obs = injector.inject(obs)          # at step 3
    obs = injector.resolve(obs, pid)    # when agent kills a PID
    injector.reset()                    # at episode start
    """

    def __init__(self) -> None:
        self.active_chaos: str | None = None
        self.chaos_pid: int | None = None
        self._wrong_kill_pid: int | None = None
        self._wrong_kill_penalty: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear state at the start of a new episode."""
        self.active_chaos = None
        self.chaos_pid = None
        self._wrong_kill_pid = None
        self._wrong_kill_penalty = 0.0

    # ------------------------------------------------------------------
    # Inject
    # ------------------------------------------------------------------

    def inject(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Randomly select and activate a chaos scenario.

        PARTIAL OBSERVABILITY: active_chaos is set to None in the returned
        obs. The agent must infer chaos from metrics and logs only.
        Internal state (self.active_chaos) is preserved for reward logic.
        """
        scenario_name = random.choice(list(CHAOS_SCENARIOS.keys()))
        scenario = CHAOS_SCENARIOS[scenario_name]

        self.active_chaos = scenario_name
        self.chaos_pid = scenario["resolution_pid"]
        self._wrong_kill_pid = scenario.get("wrong_kill_pid")
        self._wrong_kill_penalty = scenario.get("wrong_kill_penalty", 0.0)

        obs = obs.copy()

        # Inject primary chaos process
        obs["process_list"] = obs["process_list"] + [scenario["fake_process"].copy()]

        # Inject decoy process if scenario has one (fork_bomb, semantic_decoy)
        if "decoy_process" in scenario:
            obs["process_list"] = obs["process_list"] + [scenario["decoy_process"].copy()]

        # Perturb metrics
        obs["cpu_percent"]      = min(100.0, obs["cpu_percent"]      + scenario["cpu_delta"])
        obs["thermal_celsius"]  = min(110.0, obs["thermal_celsius"]  + scenario["thermal_delta"])
        obs["ram_percent"]      = min(100.0, obs["ram_percent"]      + scenario["ram_delta"])

        # KEY CHANGE: active_chaos is HIDDEN from agent
        obs["active_chaos"] = None
        obs["log_snippet"]  = scenario["log"]

        return obs

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(self, obs: dict[str, Any], killed_pid: int) -> tuple[dict[str, Any], float]:
        """
        Attempt to resolve the active chaos event.

        Returns (updated_obs, bonus_reward).
        bonus_reward is 0.0 if nothing changed, +0.0 if wrong kill
        (reward.py handles penalties), or signals resolution.

        Now also handles:
        - Wrong kill in semantic_decoy → catastrophic penalty signal
        - Decoy kill in fork_bomb → no resolution, chaos continues
        """
        if self.active_chaos is None:
            return obs, 0.0

        # Wrong kill in semantic_decoy scenario
        if killed_pid == self._wrong_kill_pid:
            # Return penalty signal — reward.py will add it
            return obs, self._wrong_kill_penalty

        # Not the resolution PID
        if killed_pid != self.chaos_pid:
            return obs, 0.0

        # Correct resolution
        scenario = CHAOS_SCENARIOS[self.active_chaos]

        obs = obs.copy()

        # Remove the chaos process and any decoy
        pids_to_remove = {killed_pid}
        obs["process_list"] = [
            p for p in obs["process_list"] if p["pid"] not in pids_to_remove
        ]

        # Recover 90% of injected delta
        obs["cpu_percent"]     = max(5.0,  obs["cpu_percent"]     - scenario["cpu_delta"]     * 0.9)
        obs["thermal_celsius"] = max(40.0, obs["thermal_celsius"] - scenario["thermal_delta"] * 0.9)
        obs["ram_percent"]     = max(10.0, obs["ram_percent"]     - scenario["ram_delta"]     * 0.9)

        obs["active_chaos"] = None
        obs["log_snippet"]  = (
            "System nominal. Anomalous process terminated. Metrics stabilising."
        )

        # Clear internal state
        self.active_chaos        = None
        self.chaos_pid           = None
        self._wrong_kill_pid     = None
        self._wrong_kill_penalty = 0.0

        return obs, 0.0  # resolution bonus handled by reward.py

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.active_chaos is not None

    def get_chaos_pid(self) -> int | None:
        return self.chaos_pid

    def get_scenario_log(self, scenario_name: str) -> str:
        if scenario_name not in CHAOS_SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario_name}")
        return CHAOS_SCENARIOS[scenario_name]["log"]

    def get_wrong_kill_info(self) -> tuple[int | None, float]:
        """Return (wrong_kill_pid, penalty) for the active scenario."""
        return self._wrong_kill_pid, self._wrong_kill_penalty

