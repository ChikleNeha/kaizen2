"""
environment/reward.py
Reward function for KaizenEnv.

All reward components are additive.  The function is pure — it only reads
from its arguments and never touches any global state, making it safe to
call from both the live environment and the GRPO reward bridge.

Reward components (in evaluation order)
----------------------------------------
1.  Parse failure          : -1.0   (immediate return, no further scoring)
2.  Protected kill         : -10.0  (immediate return, hard safety constraint)
3.  Thermal improvement    : delta_thermal  × 0.15
4.  CPU improvement        : delta_cpu      × 0.10
5.  Critical process alive : +0.5 per critical process still running
6.  Chaos resolved         : +3.0
7.  System nominal         : +1.0   (cpu < 40 AND thermal < 70)
8.  Unnecessary kill       : -2.0   (killed a process using < 5% CPU)
"""

from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Process names considered critical — their presence in obs_after earns a bonus.
CRITICAL_PROCESS_NAMES: frozenset[str] = frozenset(
    {"kernel_task", "nav_service", "kernel", "systemd"}
)

# Thresholds for the "system nominal" bonus
NOMINAL_CPU_THRESHOLD: float = 40.0
NOMINAL_THERMAL_THRESHOLD: float = 70.0

# CPU usage below which killing a process is penalised as unnecessary
UNNECESSARY_KILL_CPU_THRESHOLD: float = 5.0

# Reward magnitudes
PARSE_FAILURE_REWARD: float = -1.0
PROTECTED_KILL_REWARD: float = -10.0
THERMAL_IMPROVEMENT_SCALE: float = 0.15
CPU_IMPROVEMENT_SCALE: float = 0.10
CRITICAL_ALIVE_BONUS: float = 0.5
CHAOS_RESOLVED_BONUS: float = 3.0
NOMINAL_BONUS: float = 1.0
UNNECESSARY_KILL_PENALTY: float = -2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward(
    obs_before: dict[str, Any],
    obs_after: dict[str, Any],
    action: Any,
    action_error: str,
    protected_pids: set[int],
) -> float:
    """
    Compute the scalar reward for a single environment step.

    Parameters
    ----------
    obs_before     : observation dict collected BEFORE the action executed
    obs_after      : observation dict collected AFTER the action executed
    action         : validated AgenticOSAction instance, or None on parse failure
    action_error   : non-empty string describing parse/validation failure,
                     empty string on success
    protected_pids : set of PIDs that must never be killed (populated at reset)

    Returns
    -------
    float
        Scalar reward, rounded to 3 decimal places.
    """
    # ------------------------------------------------------------------
    # 1. Parse failure — hard stop, no further components scored
    # ------------------------------------------------------------------
    if action is None:
        return PARSE_FAILURE_REWARD

    reward: float = 0.0

    # ------------------------------------------------------------------
    # 2. Protected process kill — hard stop, overrides all other components
    # ------------------------------------------------------------------
    if hasattr(action, "pid") and action.pid in protected_pids:
        return PROTECTED_KILL_REWARD

    # ------------------------------------------------------------------
    # 3. Thermal improvement
    #    Positive when temperature dropped, negative when it rose.
    # ------------------------------------------------------------------
    delta_thermal: float = (
        obs_before["thermal_celsius"] - obs_after["thermal_celsius"]
    )
    reward += delta_thermal * THERMAL_IMPROVEMENT_SCALE

    # ------------------------------------------------------------------
    # 4. CPU improvement
    #    Positive when CPU load dropped, negative when it rose.
    # ------------------------------------------------------------------
    delta_cpu: float = obs_before["cpu_percent"] - obs_after["cpu_percent"]
    reward += delta_cpu * CPU_IMPROVEMENT_SCALE

    # ------------------------------------------------------------------
    # 5. Critical process uptime bonus
    #    Each critical process that is still alive after the action earns
    #    a fixed bonus.  This incentivises the agent to avoid collateral
    #    damage when issuing broad actions like kill_background.
    # ------------------------------------------------------------------
    after_names: set[str] = {p["name"] for p in obs_after["process_list"]}
    critical_alive: set[str] = CRITICAL_PROCESS_NAMES.intersection(after_names)
    reward += len(critical_alive) * CRITICAL_ALIVE_BONUS

    # ------------------------------------------------------------------
    # 6. Chaos resolved bonus
    #    Awarded when the chaos event was active before the action and is
    #    gone after it.
    # ------------------------------------------------------------------
    if obs_before["active_chaos"] is not None and obs_after["active_chaos"] is None:
        reward += CHAOS_RESOLVED_BONUS

    # ------------------------------------------------------------------
    # 7. System nominal bonus
    #    Small positive signal for keeping the system in a healthy state
    #    even when no chaos is active — reinforces conservative behaviour.
    # ------------------------------------------------------------------
    if (
        obs_after["cpu_percent"] < NOMINAL_CPU_THRESHOLD
        and obs_after["thermal_celsius"] < NOMINAL_THERMAL_THRESHOLD
    ):
        reward += NOMINAL_BONUS

    # ------------------------------------------------------------------
    # 8. Unnecessary kill penalty
    #    Penalise the agent for killing a process that was barely using
    #    any CPU — teaches selectivity.
    # ------------------------------------------------------------------
    if action.__class__.__name__ == "KillProcessAction":
        killed_process = next(
            (p for p in obs_before["process_list"] if p["pid"] == action.pid),
            None,
        )
        if (
            killed_process is not None
            and killed_process["cpu_percent"] < UNNECESSARY_KILL_CPU_THRESHOLD
        ):
            reward += UNNECESSARY_KILL_PENALTY

    return round(float(reward), 3)
