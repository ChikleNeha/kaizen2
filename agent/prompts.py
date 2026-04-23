"""
agent/prompts.py
System prompt and observation formatter for the Kaizen OS LLM agent.

The system prompt is injected once as the first message in the chat
template.  format_observation() converts the raw obs dict into the
structured text block the model reads each step.
"""

from typing import Any

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous OS management agent. Your job is to monitor system health and take corrective actions to prevent crashes, thermal throttling, and performance degradation.

You will receive a JSON observation of the current system state. You must:
1. Think through the observation carefully (chain-of-thought reasoning)
2. Identify the most critical issue by reading BOTH metrics AND logs
3. Output EXACTLY ONE action as a JSON object

RULES:
- Never kill a protected process (is_protected: true)
- Always prioritize thermal safety over performance
- If CPU > 85% or thermal > 85°C, this is a critical emergency — act immediately
- READ THE LOGS before killing any process — the log may reveal which process is safe to kill even if two processes have similar CPU usage
- Never kill a process whose log indicates it is performing critical work (patient data sync, financial transactions, active user sessions)
- If no action is needed, use the wait action

IMPORTANT — PARTIAL OBSERVABILITY:
You will NOT be told what type of chaos event is active. You must infer it from:
- Sudden spikes in cpu_percent, thermal_celsius, or ram_percent
- Log messages naming specific PIDs as problematic
- Process names and their resource consumption patterns
- Zombie process status entries
This is intentional — a real OS agent must diagnose problems, not just react to labelled alerts.

OUTPUT FORMAT — you must output this exact structure and nothing else after your reasoning:
{
  "tool_name": "<action_name>",
  "<param1>": <value1>,
  "reason": "<your reasoning>"
}

AVAILABLE TOOLS:
- kill_process: {"tool_name": "kill_process", "pid": <int>, "reason": "<str>"}
- allocate_memory: {"tool_name": "allocate_memory", "target_pid": <int>, "mb_to_free": <float>}
- thermal_mitigation: {"tool_name": "thermal_mitigation", "strategy": "throttle_cpu"|"kill_background"|"reduce_clock"}
- prioritize_task: {"tool_name": "prioritize_task", "pid": <int>, "priority": "high"|"normal"|"low"}
- inspect_logs: {"tool_name": "inspect_logs", "pid": <int or null>}
- list_processes: {"tool_name": "list_processes"}
- wait: {"tool_name": "wait", "reason": "<str>"}"""


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def format_observation(obs: dict[str, Any]) -> str:
    """
    Convert an observation dict into a structured text block for the LLM.

    The format is designed to be:
    - Scannable  : clear section headers, aligned columns
    - Compact    : fits within a 512-token budget alongside the system prompt
    - Unambiguous: PROTECTED flag is visually prominent so the model never
                   misses it

    Parameters
    ----------
    obs : dict
        Observation dict from ObservationBuilder.build() (possibly with
        chaos overlay applied by ChaosInjector).

    Returns
    -------
    str
        Formatted observation string ready to be passed as the user message
        in the chat template.
    """
    process_lines = "\n".join([
        f"  PID {p['pid']:>5} | {p['name']:<24} | "
        f"CPU: {p['cpu_percent']:>5.1f}% | "
        f"RAM: {p['memory_mb']:>7.0f} MB | "
        f"{'*** PROTECTED ***' if p['is_protected'] else 'killable'}"
        for p in obs.get("process_list", [])
    ])

    if not process_lines:
        process_lines = "  (no processes visible)"

    chaos_line = obs.get("active_chaos") or "None"

    return (
        f"=== SYSTEM STATE — STEP {obs.get('step', 0)} ===\n"
        f"CPU Usage    : {obs.get('cpu_percent', 0.0):.1f}%\n"
        f"RAM Usage    : {obs.get('ram_percent', 0.0):.1f}%\n"
        f"Thermal      : {obs.get('thermal_celsius', 0.0):.1f}°C\n"
        f"Uptime       : {obs.get('uptime_seconds', 0.0):.0f}s\n"
        f"\n"
        f"PROCESSES:\n"
        f"{process_lines}\n"
        f"\n"
        f"SYSTEM LOGS:\n"
        f"{obs.get('log_snippet', 'No log data.')}\n"
        f"=== END STATE ===\n"
        f"\n"
        f"Analyse the above and respond with your chain-of-thought reasoning "
        f"followed by exactly one JSON action."
    )


# ---------------------------------------------------------------------------
# Chat template builder
# ---------------------------------------------------------------------------

def build_chat_messages(obs: dict[str, Any]) -> list[dict[str, str]]:
    """
    Construct the messages list for tokenizer.apply_chat_template().

    Returns a two-element list:
      [{"role": "system", "content": SYSTEM_PROMPT},
       {"role": "user",   "content": <formatted observation>}]

    This is the canonical input format for Qwen2.5-Instruct and most
    HuggingFace instruct-tuned models.

    Parameters
    ----------
    obs : dict
        Current observation dict.

    Returns
    -------
    list[dict[str, str]]
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_observation(obs)},
    ]


# ---------------------------------------------------------------------------
# Alpaca prompt template (used in SFT training)
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(instruction: str, input_text: str, output: str) -> str:
    """
    Format a single training example using the Alpaca template.

    Parameters
    ----------
    instruction : str   — the system state observation string
    input_text  : str   — always "" for this dataset
    output      : str   — chain-of-thought + JSON action

    Returns
    -------
    str   — full formatted training string including EOS token placeholder
    """
    return ALPACA_TEMPLATE.format(
        instruction=instruction,
        input=input_text,
        output=output,
    )
