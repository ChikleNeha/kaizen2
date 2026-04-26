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
    # Extract the LAST complete JSON object in the output.
    # The GRPO model sometimes emits reasoning as a JSON block followed by
    # the action JSON — first+last braces spans both and produces "Extra data".
    # Scanning character-by-character and taking the last complete object
    # fixes this without changing any downstream validation logic.
    last_start = -1
    last_end   = -1
    candidate  = -1
    depth      = 0
    in_string  = False
    escape     = False

    for i, ch in enumerate(llm_output):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                candidate = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and candidate != -1:
                last_start = candidate
                last_end   = i
                # Don't break — keep scanning to find the LAST object

    if last_start == -1:
        return None, "No JSON object found in LLM output"

    json_str = llm_output[last_start:last_end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    
    # ← this continues inside parse_action, after the json.loads try/except

    if not isinstance(data, dict):
        return None, "JSON root is not an object"

    tool_name = data.get("tool_name", "")
    if not tool_name:
        return None, "Missing 'tool_name' field in JSON"

    model_cls = ACTION_MAP.get(tool_name)
    if model_cls is None:
        known = ", ".join(ACTION_MAP.keys())
        return None, f"Unknown tool_name '{tool_name}'. Known tools: {known}"

    try:
        action = model_cls.model_validate(data)
        return action, ""
    except ValidationError as exc:
        first_error = exc.errors()[0]
        return None, f"Validation error in {tool_name}: {first_error['msg']} @ {first_error['loc']}"