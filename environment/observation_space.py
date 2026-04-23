"""
environment/observation_space.py
Builds real-time system observations using psutil.

All reads are non-destructive / read-only against the host OS.
Thermal simulation is used automatically on platforms that do not
expose hardware sensors (Windows, macOS, most cloud VMs).
"""

import random
import time
from typing import Any

import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Process names whose presence marks a process as protected.
# Matching is substring-based and case-insensitive.
PROTECTED_NAME_FRAGMENTS: tuple[str, ...] = (
    "kernel",
    "systemd",
    "init",
    "nav_service",
    "windowserver",
    "explorer",
)

# Maximum number of processes to include in the observation.
MAX_PROCESSES = 8

# Known valid psutil status strings
VALID_STATUSES = {"running", "sleeping", "zombie", "disk-sleep", "stopped", "dead"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_protected(name: str) -> bool:
    """Return True if the process name contains any protected fragment."""
    lower = name.lower()
    return any(fragment in lower for fragment in PROTECTED_NAME_FRAGMENTS)


def _normalise_status(raw: str) -> str:
    """
    Map psutil status strings to the three values the observation schema
    recognises.  Unknown statuses fall through to 'sleeping'.
    """
    if raw in ("running",):
        return "running"
    if raw in ("zombie", "dead"):
        return "zombie"
    return "sleeping"


def _get_thermal(cpu_percent: float) -> float:
    """
    Return the current CPU temperature in degrees Celsius.

    Uses psutil.sensors_temperatures() when available (Linux with lm-sensors,
    some BSD systems).  On Windows / macOS / headless cloud VMs the call
    either raises AttributeError or returns an empty dict; in those cases we
    simulate a plausible value that tracks CPU load.
    """
    try:
        sensors = psutil.sensors_temperatures()  # type: ignore[attr-defined]
        if sensors:
            # Prefer coretemp or k10temp; fall back to whatever is first.
            for key in ("coretemp", "k10temp", "cpu_thermal", "cpu-thermal"):
                if key in sensors and sensors[key]:
                    return float(sensors[key][0].current)
            # First available sensor
            first_key = next(iter(sensors))
            if sensors[first_key]:
                return float(sensors[first_key][0].current)
    except (AttributeError, StopIteration):
        pass

    # Simulation: base 50°C + 0.4 per CPU% + small noise
    return round(50.0 + (cpu_percent * 0.4) + random.uniform(-2.0, 2.0), 2)


def _collect_processes() -> list[dict[str, Any]]:
    """
    Return the top MAX_PROCESSES processes sorted by CPU usage (descending).

    psutil requires two samples to compute CPU percent accurately; we use
    interval=0.0 (non-blocking) with a prior call so the first reading is
    not always 0.  For the very first call the values will be 0 — the chaos
    injector's fake processes dominate anyway.
    """
    procs: list[dict[str, Any]] = []

    for proc in psutil.process_iter(
        attrs=["pid", "name", "cpu_percent", "memory_info", "status"]
    ):
        try:
            info = proc.info  # type: ignore[attr-defined]
            name: str = info["name"] or "unknown"
            mem_bytes: int = (
                info["memory_info"].rss if info["memory_info"] else 0
            )
            status_raw: str = info["status"] or "sleeping"
            procs.append(
                {
                    "pid": int(info["pid"]),
                    "name": name,
                    "cpu_percent": float(info["cpu_percent"] or 0.0),
                    "memory_mb": round(mem_bytes / (1024 * 1024), 2),
                    "status": _normalise_status(status_raw),
                    "is_protected": _is_protected(name),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process vanished or is inaccessible — skip silently.
            continue

    # Sort by CPU descending, take top MAX_PROCESSES
    procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
    return procs[:MAX_PROCESSES]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """
    Stateful builder that produces observation dicts for KaizenEnv.

    The builder holds:
    - ``_start_time``  : wall-clock time of the last env reset, used to
                         compute uptime_seconds.
    - ``_step``        : current step counter, set externally by KaizenEnv.
    - ``_log_snippet`` : last log line, overwritten by the chaos injector.
    """

    def __init__(self) -> None:
        self._start_time: float = time.monotonic()
        self._step: int = 0
        self._log_snippet: str = "System nominal. No anomalies detected."

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Called by KaizenEnv.reset() to reinitialise timing and step."""
        self._start_time = time.monotonic()
        self._step = 0
        self._log_snippet = "System nominal. No anomalies detected."

        # Warm up CPU percent measurements so first real obs is non-zero.
        # The interval=0.1 call blocks briefly but gives a meaningful sample.
        psutil.cpu_percent(interval=0.1)

    def set_step(self, step: int) -> None:
        self._step = step

    def set_log_snippet(self, log: str) -> None:
        self._log_snippet = log

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """
        Collect a fresh observation from the host system.

        Returns a dict matching the schema defined in Section 1.1 of the
        project spec.  The ``active_chaos`` field is always None here; the
        ChaosInjector overlays it when active.
        """
        cpu_percent = float(psutil.cpu_percent(interval=0.1))
        ram_percent = float(psutil.virtual_memory().percent)
        thermal_celsius = _get_thermal(cpu_percent)
        uptime_seconds = round(time.monotonic() - self._start_time, 2)
        process_list = _collect_processes()

        return {
            "cpu_percent": round(cpu_percent, 2),
            "ram_percent": round(ram_percent, 2),
            "thermal_celsius": round(thermal_celsius, 2),
            "uptime_seconds": uptime_seconds,
            "process_list": process_list,
            "active_chaos": None,
            "step": self._step,
            "log_snippet": self._log_snippet,
        }

    # ------------------------------------------------------------------
    # Convenience: extract protected PIDs from an observation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_protected_pids(obs: dict[str, Any]) -> set[int]:
        """Return the set of PIDs marked is_protected in the given observation."""
        return {
            p["pid"]
            for p in obs.get("process_list", [])
            if p.get("is_protected", False)
        }

    @staticmethod
    def to_agent_obs(obs: dict[str, Any]) -> dict[str, Any]:
        """
        Return a copy of obs with active_chaos hidden (set to None).

        This enforces partial observability — the agent must infer whether
        a chaos event is active from cpu/thermal/ram spikes and log content.
        The full obs (with active_chaos) is retained internally for reward
        computation and termination logic.

        This is the key novelty over rule-based systems: a rule engine reads
        metrics and applies thresholds. The agent reads logs and applies
        semantic understanding. Only the agent can solve the semantic_decoy
        scenario where two processes have identical CPU usage and only the
        log distinguishes the critical one from the killable one.
        """
        agent_obs = obs.copy()
        agent_obs["active_chaos"] = None
        return agent_obs
