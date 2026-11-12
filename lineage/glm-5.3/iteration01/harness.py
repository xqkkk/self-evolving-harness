"""The agent loop: a bounded bash tool, the task's MCP tools as first-class
tools, and a guard that keeps the conversation inside the model's window."""

import json
import time

from . import context
from .mcp import Bridge
from .prompts import SYSTEM_PROMPT

BASH_TIMEOUT_S = 420.0     # a command that outlives this is killed and reported
PROMPT_TRIGGER = 110_000   # measured prompt tokens that make us trim the history

BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": ("Run a shell command in the task container and return its output. "
                        "stdin is closed, so a program waiting for input gets EOF instead "
                        "of hanging the run; bound long work with timeout_s or with a "
                        "background job writing to a log you poll."),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."},
                "timeout_s": {"type": "number", "description": "Optional: kill the command after this many seconds."},
            },
            "required": ["command"],
        },
    },
}


async def run_bash(world, arguments, t0):
    args = arguments if isinstance(arguments, dict) else {}
    cmd = str(args.get("command") or "")
    if not cmd.strip():
        return "error: no command was given"
    try:
        limit = max(min(float(args.get("timeout_s") or BASH_TIMEOUT_S), 3600.0), 5.0)
    except (TypeError, ValueError):
        limit = BASH_TIMEOUT_S
    if "<" not in cmd:  # no heredoc in it, so closing stdin is safe
        cmd = f"( {cmd} ) </dev/null"
    try:
        r = await world.exec(cmd, timeout_s=limit)
    except Exception as e:
        return (f"[command did not finish after {limit:.0f}s and was killed: "
                f"{type(e).__name__}: {e}] Raise timeout_s or use `timeout <secs>`, or "
                "run it in the background writing to a log and poll that log — a program "
                "waiting on stdin or a reply sits here until its bound expires.")
    parts = [r.stdout or ""] + (["[stderr]\n" + r.stderr] if r.stderr else [])
    parts.append(f"[exit {r.return_code}] [t+{int(time.monotonic() - t0)}s elapsed]")
    return context.clip("\n".join(parts))


async def run(instruction, model, world):
    t0 = time.monotonic()
    stats = {"execs": 0, "mcp_calls": 0, "trims": 0, "unknown_tools": 0,
             "peak_prompt_tokens": 0, "mcp_servers": [], "mcp_tools": 0}
    bridge = Bridge(world)
    defs, names = await bridge.discover(instruction)
    stats["mcp_servers"], stats["mcp_tools"] = [s[0] for s in bridge.servers], len(defs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}]
    tools, note, retries, backup = [BASH] + defs, None, 0, None

    def trim(keep_last=8, target=200000):
        nonlocal note
        stats["trims"] += context.compact(messages, keep_last=keep_last, hard=True, target_chars=target)
        if note is None:
            note = {"role": "system", "content": context.TRIM_NOTE}
            messages.insert(1, note)

    try:
        while True:
            before = model.input_tokens
            try:
                reply = await model.chat(messages, tools=tools)
            except Exception as e:
                retries += 1              # a call that raises ends the whole run, so
                if retries > 3:           # try again while there is anything to try
                    raise
                if "context" in str(e).lower():
                    trim(keep_last=2, target=0)   # it would not fit: shed history
                elif backup is not None:  # a trim is not guaranteed to be accepted,
                    messages[:] = backup  # so put the history back and try once more
                    backup = None
                continue
            retries, backup = 0, None
            messages.append(reply.message)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                if call.name == "bash":
                    stats["execs"] += 1
                    out = await run_bash(world, call.arguments, t0)
                elif call.name in bridge.tools:
                    stats["mcp_calls"] += 1
                    out = context.clip(await bridge.call(call.name, call.arguments))
                    out += f"\n[t+{int(time.monotonic() - t0)}s elapsed]"
                else:
                    stats["unknown_tools"] += 1
                    out = f"unknown tool {call.name!r}: use `bash` or this task's MCP tools"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            used = model.input_tokens - before    # measured prompt length of that call
            stats["peak_prompt_tokens"] = max(stats["peak_prompt_tokens"], used)
            if used > PROMPT_TRIGGER:             # trim once the turn is on the record
                backup = [dict(m) for m in messages]
                trim()
    finally:  # /logs/agent is collected with the trial: leave the next round a report
        payload = dict(stats, elapsed_s=round(time.monotonic() - t0), model_calls=model.calls,
                       input_tokens=model.input_tokens, output_tokens=model.output_tokens)
        script = ("mkdir -p /logs/agent && cat > /logs/agent/harness_report.json <<'HR_EOF'\n"
                  + json.dumps(payload, indent=1) + "\nHR_EOF")
        try:
            await world.exec(script, timeout_s=30)
        except Exception:
            pass
