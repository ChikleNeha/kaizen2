"""
environment/action_space.py
Pydantic v2 action models for the Kaizen OS agent.
All actions are validated via model_validate() before execution.
"""

import json
from typing import Literal, Union

from pydantic import BaseModel, Field, ValidationError


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------

class KillProcessAction(BaseModel):
    tool_name: Literal["kill_process"]
    pid: int = Field(..., description="PID of process to kill")
    reason: str = Field(..., description="Agent's reasoning for killing this process")


class AllocateMemoryAction(BaseModel):
    tool_name: Literal["allocate_memory"]
    target_pid: int = Field(..., description="PID to reallocate memory from")
    mb_to_free: float = Field(..., gt=0, le=512)


class ThermalMitigationAction(BaseModel):
    tool_name: Literal["thermal_mitigation"]
    strategy: Literal["throttle_cpu", "kill_background", "reduce_clock"]


class PrioritizeTaskAction(BaseModel):
    tool_name: Literal["prioritize_task"]
    pid: int
    priority: Literal["high", "normal", "low"]


class InspectLogsAction(BaseModel):
    tool_name: Literal["inspect_logs"]
    pid: int | None = None  # None = system logs, int = process-specific logs


class ListProcessesAction(BaseModel):
    tool_name: Literal["list_processes"]


class WaitAction(BaseModel):
    tool_name: Literal["wait"]
    reason: str


# ---------------------------------------------------------------------------
# Union type used throughout the codebase
# ---------------------------------------------------------------------------

AgenticOSAction = Union[
    KillProcessAction,
    AllocateMemoryAction,
    ThermalMitigationAction,
    PrioritizeTaskAction,
    InspectLogsAction,
    ListProcessesAction,
    WaitAction,
]

# ---------------------------------------------------------------------------
# Action registry — maps tool_name strings to their model class
# ---------------------------------------------------------------------------

ACTION_MAP: dict[str, type[BaseModel]] = {
    "kill_process": KillProcessAction,
    "allocate_memory": AllocateMemoryAction,
    "thermal_mitigation": ThermalMitigationAction,
    "prioritize_task": PrioritizeTaskAction,
    "inspect_logs": InspectLogsAction,
    "list_processes": ListProcessesAction,
    "wait": WaitAction,
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_action(llm_output: str) -> tuple[AgenticOSAction | None, str]:
    """
    Parse an LLM output string into a validated AgenticOSAction.

    Returns
    -------
    (action, error_message)
        If parsing succeeds : (action, "")
        If parsing fails    : (None, reason_string)

    Strategy
    --------
    Finds the first '{' and last '}' in the output, extracts that substring
    as JSON, looks up the tool_name key, then delegates to the appropriate
    Pydantic model for full field validation.  Any failure at any stage
    returns (None, descriptive_error) so the environment can assign -1 reward
    immediately without executing anything.
    """
    # Locate the JSON object boundaries
    start = llm_output.find('{')
    end = llm_output.rfind('}')

    if start == -1 or end == -1:
        return None, "No JSON object found in LLM output"

    json_str = llm_output[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"

    tool_name = data.get("tool_name", "")

    if tool_name not in ACTION_MAP:
        return None, f"Unknown tool_name: '{tool_name}'. Valid options: {list(ACTION_MAP.keys())}"

    try:
        action = ACTION_MAP[tool_name].model_validate(data)
        return action, ""
    except ValidationError as exc:
        # Return a compact summary of all field errors
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        return None, f"Validation error for '{tool_name}': {errors}"
