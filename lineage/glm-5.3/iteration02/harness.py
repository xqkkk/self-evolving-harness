"""The agent loop: a bounded bash tool, the task's MCP tools as first-class
tools, a durable plan, and two guards that keep a run from ending empty.

The context guard is the old one: measure the real prompt length, clip each
result to what the window can afford, shed replayed reasoning when the
measurement says to. The finish guard is new, and exists because the plainest
way to lose a task is to stop one reply too early — the model writes the
answer into its final message, the run ends, and nothing outside the
conversation ever sees it. So a reply with no tool call is bounced back once
with the check it needs to make, and only a second such reply ends the run.

Between them sits the clock: tool results carry the elapsed seconds, and at a
few milestones the harness adds what it can see of the world — how many files
have actually changed — next to the reminder that the budget is unknown and
runs get cut off without warning.
"""

import json
import time

from . import context, observe
from .mcp import Bridge
from .prompts import FINISH_CHECK, SYSTEM_PROMPT, milestone_note

BASH_TIMEOUT_S = 420.0     # a command that outlives this is killed and reported
PROMPT_TRIGGER = 110_000   # measured prompt tokens that make us trim the history
MILESTONES_S = (420.0, 900.0, 1800.0, 2700.0)
MAX_BOUNCES = 2            # finish checks: the second only if work happened since
PLAN_FILE = "/tmp/.harness_plan.md"

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

PLAN = {
    "type": "function",
    "function": {
        "name": "plan",
        "description": ("Set or update the plan you are working from: an ordered list of "
                        "steps, each a short title plus a status. A one-line summary of it "
                        "rides along on every tool result, so it survives whatever the "
                        "harness drops from the conversation later. Rewrite the whole list "
                        "on every call; keep titles short and keep the list honest — mark "
                        "a step done only when it is."),
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "description": "The steps, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "What this step is."},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                        },
                        "required": ["title"],
                    },
                },
            },
            "required": ["steps"],
        },
    },
}


def footer(t0, plan):
    """The line every tool result ends with: the clock, and where the plan is."""
    parts = [f"t+{int(time.monotonic() - t0)}s elapsed"]
    line = observe.render_plan(plan)
    if line:
        parts.append(line)
    return "[" + " | ".join(parts) + "]"


async def run_bash(world, arguments, t0, plan, hist_chars):
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
                "waiting on stdin or a reply sits here until its bound expires.\n"
                + footer(t0, plan))
    parts = [r.stdout or ""] + (["[stderr]\n" + r.stderr] if r.stderr else [])
    parts.append(f"[exit {r.return_code}] " + footer(t0, plan))
    return context.clip("\n".join(parts), limit=context.clip_budget(hist_chars))


async def run_plan(world, steps, t0, plan):
    """Record the plan, keep a copy in the container, echo it back in full."""
    if not isinstance(steps, list):
        return "error: `steps` must be a list of {title, status}"
    clean = []
    for s in steps:
        if isinstance(s, dict) and str(s.get("title") or "").strip():
            clean.append({"title": str(s["title"]).strip()[:200],
                          "status": str(s.get("status") or "pending")})
    plan[:] = clean
    body = "\n".join(f"- [{s['status']}] {s['title']}" for s in clean) or "(empty plan)"
    try:
        await world.exec(
            "mkdir -p /tmp && cat > " + PLAN_FILE + " <<'HPLAN_EOF'\n" + body
            + "\nHPLAN_EOF", timeout_s=20)
    except Exception:
        pass            # the summary in the footer is the fallback copy
    return ("plan recorded (" + footer(t0, plan) + ")\n" + body
            + "\nIt is kept at " + PLAN_FILE + "; read it back from there if the "
            "conversation loses it.")


async def run(instruction, model, world):
    t0 = time.monotonic()
    stats = {"execs": 0, "mcp_calls": 0, "trims": 0, "unknown_tools": 0,
             "peak_prompt_tokens": 0, "mcp_servers": [], "mcp_tools": 0,
             "bounces": 0, "milestones": 0, "plan_updates": 0,
             "files_changed_last": None, "final_reply_chars": None}
    bridge = Bridge(world)
    defs, names = await bridge.discover(instruction)
    stats["mcp_servers"], stats["mcp_tools"] = [s[0] for s in bridge.servers], len(defs)
    facts = await observe.survey(world)     # one look at the container, so the
    stats["survey_chars"] = len(facts or "")  # first turn is not spent on `ls`
    messages = [{"role": "system", "content": SYSTEM_PROMPT + (
                     "\n\n[harness] One look at this container, taken before your "
                     "first turn — what is installed and what is where. It is a "
                     "starting point, not the whole picture; check anything that "
                     "matters.\n" + facts if facts else "")},
                {"role": "user", "content": instruction}]
    tools, note, retries, backup = [BASH] + defs + [PLAN], None, 0, None
    plan, bounces, worked, milestone_i = [], 0, False, 0

    def trim(keep_last=8, target=200000):
        nonlocal note
        stats["trims"] += context.compact(messages, keep_last=keep_last, hard=True, target_chars=target)
        if note is None:
            note = {"role": "system", "content": context.TRIM_NOTE}
            messages.insert(1, note)

    try:
        await observe.mark_start(world)
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
                if bounces < MAX_BOUNCES and (bounces == 0 or worked):
                    bounces += 1          # a reply reaches nobody outside this run:
                    worked = False        # give the deliverable one last chance
                    stats["bounces"] = bounces
                    messages.append({"role": "user", "content": FINISH_CHECK})
                    continue
                stats["final_reply_chars"] = len(reply.text or "")
                break
            worked = True
            for call in reply.tool_calls:
                if call.name == "bash":
                    stats["execs"] += 1
                    out = await run_bash(world, call.arguments, t0, plan,
                                         context.size(messages))
                elif call.name == "plan":
                    stats["plan_updates"] += 1
                    out = await run_plan(world, call.arguments.get("steps")
                                         if isinstance(call.arguments, dict) else None,
                                         t0, plan)
                elif call.name in bridge.tools:
                    stats["mcp_calls"] += 1
                    out = context.clip(await bridge.call(call.name, call.arguments),
                                       limit=context.clip_budget(context.size(messages)))
                    out += "\n" + footer(t0, plan)
                else:
                    stats["unknown_tools"] += 1
                    out = f"unknown tool {call.name!r}: use `bash`, `plan`, or this task's MCP tools"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            while milestone_i < len(MILESTONES_S) and time.monotonic() - t0 >= MILESTONES_S[milestone_i]:
                milestone_i += 1
            if milestone_i > stats["milestones"]:
                stats["milestones"] = milestone_i
                stats["files_changed_last"] = await observe.changed_files(world)
                messages.append({"role": "user", "content": milestone_note(
                    time.monotonic() - t0, stats["files_changed_last"])})
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
