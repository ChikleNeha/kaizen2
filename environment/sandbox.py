"""
environment/sandbox.py
Sandbox executor for KaizenEnv actions.

Execution model
---------------
For the hackathon demo the sandbox runs in SIMULATED mode — no real OS
processes are killed or modified.  Every action returns a realistic mock
result and an obs_delta dict that KaizenEnv merges into the current
observation.

Docker upgrade path
-------------------
If Docker is available on the host, the executor can be switched to
DOCKER mode by passing use_docker=True to SandboxExecutor.__init__().
In Docker mode the sandbox.py file is copied into a python:3.11-slim
container (see docker/Dockerfile.sandbox) and actions are forwarded via
subprocess stdin/stdout as JSON.  For the demo, simulated mode is always
used so judges never risk their machines being modified.

Action → result mapping (simulated mode)
-----------------------------------------
kill_process       : checks PID in obs; removes it from process_list delta
allocate_memory    : reduces ram_percent by 5–15 %
thermal_mitigation : reduces thermal_celsius by 8–12 °C
prioritize_task    : acknowledges priority change, no metric delta
inspect_logs       : returns log_snippet from obs
list_processes     : returns formatted process table
wait               : no-op acknowledgement
"""

import json
import random
import shutil
import subprocess
from typing import Any

from environment.action_space import (
    AgenticOSAction,
    AllocateMemoryAction,
    InspectLogsAction,
    KillProcessAction,
    ListProcessesAction,
    PrioritizeTaskAction,
    ThermalMitigationAction,
    WaitAction,
)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
# Every execute() call returns a dict with these keys:
#   success   : bool
#   message   : str   — human-readable result shown in the dashboard
#   obs_delta : dict  — fields to merge into the obs after execution
#                       (may be empty if the action has no metric effect)


def _result(success: bool, message: str, obs_delta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": success,
        "message": message,
        "obs_delta": obs_delta or {},
    }


# ---------------------------------------------------------------------------
# Simulated executor helpers
# ---------------------------------------------------------------------------

def _execute_kill_process(action: KillProcessAction, obs: dict[str, Any]) -> dict[str, Any]:
    """
    Simulate killing a process.

    Rules
    -----
    - If the PID exists in the process list and is not protected → success.
    - If the PID exists but is protected → denied.
    - If the PID is not found → not found error.
    """
    process_list: list[dict] = obs.get("process_list", [])
    target = next((p for p in process_list if p["pid"] == action.pid), None)

    if target is None:
        return _result(
            False,
            f"PID {action.pid} not found in process list.",
        )

    if target.get("is_protected", False):
        return _result(
            False,
            f"Protected process '{target['name']}' (PID {action.pid}) — kill denied. "
            "System integrity requires this process to remain running.",
        )

    # Remove process from the list in obs_delta
    updated_list = [p for p in process_list if p["pid"] != action.pid]

    # Simulate CPU relief proportional to the killed process's usage
    cpu_relief = target.get("cpu_percent", 0.0) * 0.85
    mem_relief_mb = target.get("memory_mb", 0.0)

    # Convert memory relief to RAM percent relief (rough approximation)
    total_ram_mb = 8192.0  # assume 8 GB host for simulation
    ram_relief = (mem_relief_mb / total_ram_mb) * 100.0

    new_cpu = max(5.0, obs.get("cpu_percent", 50.0) - cpu_relief)
    new_ram = max(10.0, obs.get("ram_percent", 50.0) - ram_relief)

    return _result(
        True,
        f"Process '{target['name']}' (PID {action.pid}) terminated. "
        f"CPU load reducing by ~{cpu_relief:.1f}%.",
        obs_delta={
            "process_list": updated_list,
            "cpu_percent": round(new_cpu, 2),
            "ram_percent": round(new_ram, 2),
        },
    )


def _execute_allocate_memory(action: AllocateMemoryAction, obs: dict[str, Any]) -> dict[str, Any]:
    """
    Simulate freeing memory from a target process.
    Reduces ram_percent by a value proportional to mb_to_free.
    """
    process_list: list[dict] = obs.get("process_list", [])
    target = next((p for p in process_list if p["pid"] == action.target_pid), None)

    if target is None:
        return _result(
            False,
            f"PID {action.target_pid} not found — cannot reallocate memory.",
        )

    total_ram_mb = 8192.0
    relief_percent = (action.mb_to_free / total_ram_mb) * 100.0
    # Clamp to a realistic 5–15 % window
    relief_percent = max(5.0, min(15.0, relief_percent))
    new_ram = max(10.0, obs.get("ram_percent", 50.0) - relief_percent)

    return _result(
        True,
        f"Freed {action.mb_to_free:.0f} MB from PID {action.target_pid} "
        f"({target['name']}). RAM usage reduced by ~{relief_percent:.1f}%.",
        obs_delta={"ram_percent": round(new_ram, 2)},
    )


def _execute_thermal_mitigation(
    action: ThermalMitigationAction, obs: dict[str, Any]
) -> dict[str, Any]:
    """
    Simulate a thermal mitigation strategy.
    Each strategy reduces thermal_celsius by 8–12 °C.
    throttle_cpu also reduces cpu_percent slightly.
    kill_background also trims the process list.
    """
    reduction = round(random.uniform(8.0, 12.0), 2)
    new_thermal = max(40.0, obs.get("thermal_celsius", 70.0) - reduction)
    delta: dict[str, Any] = {"thermal_celsius": round(new_thermal, 2)}

    if action.strategy == "throttle_cpu":
        cpu_reduction = round(random.uniform(5.0, 15.0), 2)
        delta["cpu_percent"] = max(5.0, round(obs.get("cpu_percent", 50.0) - cpu_reduction, 2))
        msg = (
            f"CPU throttled. Thermal reduced by {reduction:.1f}°C. "
            f"CPU reduced by {cpu_reduction:.1f}%."
        )

    elif action.strategy == "kill_background":
        # Remove the lowest-priority non-protected process to simulate kill_background
        killable = [
            p for p in obs.get("process_list", [])
            if not p.get("is_protected", False)
        ]
        if killable:
            victim = min(killable, key=lambda p: p["cpu_percent"])
            updated_list = [
                p for p in obs.get("process_list", [])
                if p["pid"] != victim["pid"]
            ]
            delta["process_list"] = updated_list
            msg = (
                f"Background process '{victim['name']}' (PID {victim['pid']}) suspended. "
                f"Thermal reduced by {reduction:.1f}°C."
            )
        else:
            msg = f"No background processes to kill. Thermal reduced by {reduction:.1f}°C via passive cooling."

    else:  # reduce_clock
        msg = (
            f"Clock frequency reduced. Thermal reduced by {reduction:.1f}°C. "
            "Performance may be temporarily degraded."
        )

    return _result(True, msg, obs_delta=delta)


def _execute_prioritize_task(
    action: PrioritizeTaskAction, obs: dict[str, Any]
) -> dict[str, Any]:
    process_list: list[dict] = obs.get("process_list", [])
    target = next((p for p in process_list if p["pid"] == action.pid), None)

    if target is None:
        return _result(False, f"PID {action.pid} not found — cannot reprioritise.")

    return _result(
        True,
        f"Priority of '{target['name']}' (PID {action.pid}) updated to '{action.priority}'.",
        obs_delta={},
    )


def _execute_inspect_logs(action: InspectLogsAction, obs: dict[str, Any]) -> dict[str, Any]:
    """
    Return log information.
    If pid is None, return the system-level log_snippet.
    If pid is set, return process-specific log lines (simulated).
    """
    if action.pid is None:
        log = obs.get("log_snippet", "No log data available.")
        return _result(True, f"System logs:\n{log}")

    process_list: list[dict] = obs.get("process_list", [])
    target = next((p for p in process_list if p["pid"] == action.pid), None)

    if target is None:
        return _result(False, f"PID {action.pid} not found — no process logs available.")

    # Simulate process-specific log lines based on current metrics
    cpu = target.get("cpu_percent", 0.0)
    mem = target.get("memory_mb", 0.0)
    status = target.get("status", "unknown")

    simulated_log = (
        f"[{action.pid}] {target['name']} — status: {status}\n"
        f"[{action.pid}] CPU: {cpu:.1f}%  MEM: {mem:.0f}MB\n"
        f"[{action.pid}] Last event: {'high resource usage detected' if cpu > 50 else 'nominal operation'}"
    )
    return _result(True, f"Process logs for PID {action.pid}:\n{simulated_log}")


def _execute_list_processes(obs: dict[str, Any]) -> dict[str, Any]:
    """Return a formatted process table as the result message."""
    process_list: list[dict] = obs.get("process_list", [])

    if not process_list:
        return _result(True, "No processes found.")

    header = f"{'PID':>6}  {'NAME':<22}  {'CPU%':>6}  {'MEM MB':>8}  {'STATUS':<10}  PROTECTED"
    separator = "-" * 72
    rows = [
        f"{p['pid']:>6}  {p['name']:<22}  {p['cpu_percent']:>6.1f}  "
        f"{p['memory_mb']:>8.1f}  {p['status']:<10}  "
        f"{'YES' if p.get('is_protected') else 'no'}"
        for p in sorted(process_list, key=lambda x: x["cpu_percent"], reverse=True)
    ]
    table = "\n".join([header, separator] + rows)
    return _result(True, table)


def _execute_wait(action: WaitAction) -> dict[str, Any]:
    return _result(
        True,
        f"Agent waited one step. Reason: {action.reason}",
    )


# ---------------------------------------------------------------------------
# Docker mode (upgrade path — not used during demo)
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    """Return True if the Docker CLI is present and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=3,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


class _DockerSandbox:
    """
    Forwards action execution to a Docker container running sandbox.py.
    The container receives the action + obs as JSON on stdin and returns
    a result dict on stdout.

    This class is instantiated only when use_docker=True AND Docker is
    confirmed available.  For the hackathon demo, SandboxExecutor always
    falls back to SimulatedSandbox.
    """

    CONTAINER_IMAGE = "kaizen-sandbox:latest"

    def execute(self, action: AgenticOSAction, obs: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(
            {"action": action.model_dump(), "obs": obs}
        ).encode()
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "--network=none",
                    "--memory=128m", "--cpus=0.5",
                    self.CONTAINER_IMAGE,
                    "python", "-c",
                    "import sys, json, sandbox; "
                    "d=json.load(sys.stdin); "
                    "print(json.dumps(sandbox.dispatch(d['action'], d['obs'])))",
                ],
                input=payload,
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                return _result(False, f"Docker sandbox error: {result.stderr.decode()[:200]}")
            return json.loads(result.stdout.decode())
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            return _result(False, f"Docker execution failed: {exc}")


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class SandboxExecutor:
    """
    Unified executor that dispatches to simulated or Docker mode.

    Parameters
    ----------
    use_docker : bool
        If True, attempt to use Docker.  Falls back to simulated mode
        automatically if Docker is unavailable.  Default: False.
    """

    def __init__(self, use_docker: bool = False) -> None:
        self._use_docker = use_docker and _docker_available()
        self._docker = _DockerSandbox() if self._use_docker else None

        mode = "Docker" if self._use_docker else "Simulated"
        print(f"[SandboxExecutor] Running in {mode} mode.")

    def execute(self, action: AgenticOSAction, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a validated action against the current observation.

        Parameters
        ----------
        action : AgenticOSAction
            A validated Pydantic action instance (never None — callers must
            check parse_action() return value before calling execute).
        obs    : dict
            Current observation dict from ObservationBuilder.

        Returns
        -------
        dict with keys: success (bool), message (str), obs_delta (dict)
        """
        if self._use_docker and self._docker is not None:
            return self._docker.execute(action, obs)

        # Simulated dispatch
        if isinstance(action, KillProcessAction):
            return _execute_kill_process(action, obs)

        if isinstance(action, AllocateMemoryAction):
            return _execute_allocate_memory(action, obs)

        if isinstance(action, ThermalMitigationAction):
            return _execute_thermal_mitigation(action, obs)

        if isinstance(action, PrioritizeTaskAction):
            return _execute_prioritize_task(action, obs)

        if isinstance(action, InspectLogsAction):
            return _execute_inspect_logs(action, obs)

        if isinstance(action, ListProcessesAction):
            return _execute_list_processes(obs)

        if isinstance(action, WaitAction):
            return _execute_wait(action)

        # Should never reach here given Pydantic validation upstream
        return _result(False, f"Unrecognised action type: {type(action).__name__}")
