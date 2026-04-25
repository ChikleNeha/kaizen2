"""
environment/reward.py
Reward function for KaizenEnv.

FIX (v2): Added chaos_resolved parameter.
Previously the +3.0 chaos-resolved bonus never fired because
obs["active_chaos"] is always None (partial observability design hides it
from the observation dict). The env already computes chaos_resolved as a
bool from internal chaos state — we now accept it as an explicit parameter.

Two callers need updating:
  - kaizen_env.py: compute_reward(..., chaos_resolved=chaos_resolved)
  - grpo_train.py reward_fn: same
"""

from typing import Any


def compute_reward(
    obs_before: dict[str, Any],
    obs_after: dict[str, Any],
    action: Any,
    action_error: str,
    protected_pids: set[int],
    chaos_resolved: bool = False,   # <-- FIX: explicit flag, not obs field
) -> float:
    """
    Compute the scalar reward for one environment step.

    Parameters
    ----------
    obs_before     : observation dict BEFORE the action
    obs_after      : observation dict AFTER the action
    action         : parsed AgenticOSAction (or None on parse failure)
    action_error   : parse error string (empty if action is not None)
    protected_pids : set of PIDs that must never be killed
    chaos_resolved : True when the chaos event was active before this step
                     and is no longer active after it.  Passed from
                     KaizenEnv which tracks chaos state internally.

    Returns
    -------
    float : scalar reward, rounded to 3 decimal places
    """
    reward = 0.0

    # ------------------------------------------------------------------
    # Hard failures — return immediately, no partial credit
    # ------------------------------------------------------------------

    # Parse failure: model output was not valid JSON / unknown action
    if action is None:
        return -1.0

    # Protected process kill: catastrophic penalty
    if hasattr(action, 'pid') and action.pid in protected_pids:
        return -10.0

    # ------------------------------------------------------------------
    # Continuous improvement rewards
    # ------------------------------------------------------------------

    # Thermal improvement: 0.15 per degree cooled
    delta_thermal = obs_before["thermal_celsius"] - obs_after["thermal_celsius"]
    reward += delta_thermal * 0.15

    # CPU improvement: 0.10 per percent point reduced
    delta_cpu = obs_before["cpu_percent"] - obs_after["cpu_percent"]
    reward += delta_cpu * 0.10

    # Critical process uptime bonus: reward keeping vital processes alive
    critical_names = {"kernel_task", "nav_service", "kernel", "systemd"}
    after_names = {p["name"] for p in obs_after["process_list"]}
    critical_alive = critical_names.intersection(after_names)
    reward += len(critical_alive) * 0.5

    # ------------------------------------------------------------------
    # Chaos resolution bonus
    # FIX: use chaos_resolved parameter instead of obs["active_chaos"]
    # obs["active_chaos"] is always None (partial observability hides it)
    # ------------------------------------------------------------------
    if chaos_resolved:
        reward += 3.0

    # ------------------------------------------------------------------
    # System nominal bonus
    # ------------------------------------------------------------------
    if obs_after["cpu_percent"] < 40 and obs_after["thermal_celsius"] < 70:
        reward += 1.0

    # ------------------------------------------------------------------
    # Unnecessary kill penalty
    # ------------------------------------------------------------------
    if action.__class__.__name__ == "KillProcessAction":
        killed_process = next(
            (p for p in obs_before["process_list"] if p["pid"] == action.pid),
            None,
        )
        if killed_process and killed_process["cpu_percent"] < 5.0:
            reward -= 2.0

    return round(float(reward), 3)