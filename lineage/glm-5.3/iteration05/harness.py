"""The agent loop: a bounded bash tool, the task's MCP tools as first-class
tools, a durable plan, and three guards that keep a run from ending empty.

The context guard measures the real prompt length, clips each result to what
the window can afford, and — once a rejection or the trigger says the window
is nearer than assumed — compacts until the history is *actually* inside it,
learning from the rejection where the limit lies. A failed call is retried,
but never endlessly unchanged, and a reply the gateway cut off mid-generation
is answered with a smaller request rather than resent as it was: the task
statement, the plan and the kept tool output all live in files the
conversation cannot lose.

The finish guard exists because the plainest way to lose a task is to stop
one reply too early — the model writes the answer into its final message, the
run ends, and nothing outside the conversation ever sees it. So a reply with
no tool call is bounced back once with the checks it needs to make —
deliverable in place, nothing new arrived unhandled, nothing still open — and
only a second such reply ends the run.

Between them sits the clock: tool results carry the elapsed seconds, a
command that ate a quarter of the run says so, and at a few milestones the
harness adds what it can see of the world — how many files have actually
changed, whether a plan exists at all — next to the reminder that the budget
is unknown and runs get cut off without warning.
"""

import json
import time

from . import context, observe
from .mcp import Bridge
from .prompts import FINISH_CHECK, GEN_CUTOFF_NOTE, SLOW_CALL_NOTE, SYSTEM_PROMPT, milestone_note

BASH_TIMEOUT_S = 300.0     # a command that outlives this is killed and reported
PROMPT_TRIGGER = 90_000    # measured prompt tokens that make us trim the history
SLOW_CALL_S = 240.0        # a reply that took this long is thinking in the wrong place
MILESTONES_S = (300.0, 720.0, 1500.0, 2600.0)
TASK_FILE = "/tmp/.harness_task.md"
MAX_BOUNCES = 2            # finish checks: the second only if work happened since


PLAN_FILE = "/tmp/.harness_plan.md"

def _generation_cutoff(err):
    """Did the gateway stop the model mid-reply rather than reject the request?

    A cut-off leaves no reply to use, and resending the same request buys
    another cut at the same price — minutes of clock for nothing. The words
    differ per gateway; the shape ("the reply was too long / too slow to
    finish") is what we match, and a context rejection is handled first.
    """
    low = err.lower()
    return (("generation" in low and "cut" in low) or "max_tokens" in low
            or "token limit" in low or "length limit" in low)

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


async def run_bash(world, arguments, t0, plan, hist_chars, stats):
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
    t1 = time.monotonic()
    try:
        r = await world.exec(cmd, timeout_s=limit)
    except Exception as e:
        return (f"[command did not finish after {limit:.0f}s and was killed: "
                f"{type(e).__name__}: {e}] Raise timeout_s or use `timeout <secs>`, or "
                "run it in the background writing to a log and poll that log — a program "
                "waiting on stdin or a reply sits here until its bound expires.\n"
                + footer(t0, plan))
    parts = [r.stdout or ""] + (["[stderr]\n" + r.stderr] if r.stderr else [])
    spent, took = time.monotonic() - t0, time.monotonic() - t1
    stats["longest_exec_s"] = max(stats.get("longest_exec_s", 0), round(took))
    if took > 60 and took > 0.2 * spent:  # a probe that ate the run's clock
        parts.append(f"[this command took {int(took)}s — {int(100 * took / spent)}% "
                     "of the run so far; bound the next such probe with "
                     "`timeout <secs>`, or run it in the background and poll]")
    parts.append(f"[exit {r.return_code}] " + footer(t0, plan))
    return await context.keep("\n".join(parts),
                               limit=context.clip_budget(hist_chars), world=world,
                               stats=stats)


async def write_brief(world, instruction):
    """The task statement, verbatim on disk: the one text a trim must not eat.

    Every later safety net points here — the trim note, the plan's footer —
    so a model whose conversation was compacted can re-read what it was asked
    for instead of reconstructing it from a summary of a summary.
    """
    try:
        await world.exec("mkdir -p /tmp && cat > " + TASK_FILE
                         + " <<'HTASK_EOF'\n" + instruction + "\nHTASK_EOF", timeout_s=20)
        return True
    except Exception:
        return False


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
    repeats = {}      # (tool, arguments) -> times called unchanged, in a row
    stats = {"execs": 0, "mcp_calls": 0, "trims": 0, "unknown_tools": 0,
             "peak_prompt_tokens": 0, "mcp_servers": [], "mcp_tools": 0,
             "bounces": 0, "milestones": 0, "plan_updates": 0,
             "files_changed_last": None, "final_reply_chars": None,
             "context_errors": 0, "prompt_trigger": PROMPT_TRIGGER,
             "longest_exec_s": 0, "model_retries": 0,
             "gen_cutoffs": 0, "long_calls": 0, "kept_files": 0,
             "repeat_calls": 0}
    bridge = Bridge(world)
    defs, names = await bridge.discover(instruction)
    stats["mcp_servers"], stats["mcp_tools"] = [s[0] for s in bridge.servers], len(defs)
    facts = await observe.survey(world)     # one look at the container, so the
    stats["survey_chars"] = len(facts or "")  # first turn is not spent on `ls`
    brief = await write_brief(world, instruction)
    opening = ""
    if facts:
        opening = ("\n\n[harness] One look at this container, taken before your "
                   "first turn — what is installed and what is where. It is a "
                   "starting point, not the whole picture; check anything that "
                   "matters.\n" + facts)
    if brief:
        opening += ("\n[harness] The task statement is saved verbatim at " + TASK_FILE
                    + ", and your plan once recorded at " + PLAN_FILE
                    + ". Anything the conversation drops can be re-read from there.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT + opening},
                {"role": "user", "content": instruction}]
    tools, note, retries, backup = [BASH] + defs + [PLAN], None, 0, None
    plan, bounces, worked, milestone_i, trigger = [], 0, False, 0, PROMPT_TRIGGER

    def trim(keep_last=8, target=160_000):
        nonlocal note
        stats["trims"] += context.compact(messages, keep_last=keep_last, hard=True, target_chars=target)
        if note is None:
            note = {"role": "system", "content": context.TRIM_NOTE}
            messages.insert(1, note)

    try:
        await observe.mark_start(world)
        while True:
            before = model.input_tokens
            t_call = time.monotonic()
            try:
                reply = await model.chat(messages, tools=tools)
            except Exception as e:
                retries += 1              # a call that raises ends the whole run, so
                if retries > 3:           # try again while there is anything to try
                    raise
                if "context" in str(e).lower():   # a rejection names the window;
                    # a cut-off does not, so it is told apart after this one
                    # The rejection itself says where the window is: pull the
                    # trigger below the prompt that failed, so the run does
                    # not walk back into the same wall next turn.
                    stats["context_errors"] += 1
                    trigger = min(trigger, max(24_000, int(stats["peak_prompt_tokens"] * 0.6)))
                    stats["prompt_trigger"] = trigger
                    trim(keep_last=2, target=24_000)
                elif _generation_cutoff(str(e)):
                    stats["gen_cutoffs"] += 1
                    # teach it once, then only shrink: the note says what to do,
                    # the smaller history makes the next reply cheaper anyway
                    if stats["gen_cutoffs"] <= 2:
                        messages.append({"role": "user", "content": GEN_CUTOFF_NOTE})
                    trim(keep_last=6, target=90_000)
                elif backup is not None:  # a trim is not guaranteed to be accepted,
                    messages[:] = backup  # so put the history back and try once more
                    backup = None
                elif retries >= 2:
                    # The same request has now failed twice, unchanged. A
                    # gateway's own limits — a generation cut-off, a stream
                    # that never came back — do not care that the request is
                    # legal, and resending it identically only burns the clock
                    # the task needed. Send something smaller and carry on.
                    stats["model_retries"] = retries
                    trim(keep_last=4, target=60_000)
                continue
            retries, backup = 0, None
            took_s = time.monotonic() - t_call
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
                                         context.size(messages), stats)
                elif call.name == "plan":
                    stats["plan_updates"] += 1
                    out = await run_plan(world, call.arguments.get("steps")
                                         if isinstance(call.arguments, dict) else None,
                                         t0, plan)
                elif call.name in bridge.tools:
                    stats["mcp_calls"] += 1
                    key = (call.name, json.dumps(call.arguments, sort_keys=True))
                    seen = repeats.get(key, 0) + 1     # identical world-changing call
                    repeats[key] = seen
                    out = await context.keep(await bridge.call(call.name, call.arguments),
                                             limit=context.clip_budget(
                                                 context.size(messages)),
                                             world=world, stats=stats)
                    if seen >= 3:
                        stats["repeat_calls"] = seen
                        out += (f"\n[harness] this is call {seen} of the same "
                                f"{call.name} with identical arguments. If it errored, an "
                                "identical retry cannot fix it — read the error and change "
                                f"the arguments. If it succeeded, you have done it {seen} "
                                f"times, and a judge may count {seen} actions.")
                    out += "\n" + footer(t0, plan)
                else:
                    stats["unknown_tools"] += 1
                    out = f"unknown tool {call.name!r}: use `bash`, `plan`, or this task's MCP tools"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
            if took_s > SLOW_CALL_S and messages[-1].get("role") == "tool":
                stats["long_calls"] += 1
                if stats["long_calls"] <= 2:      # a lesson, not a metronome
                    messages.append({"role": "user", "content": SLOW_CALL_NOTE})
            while milestone_i < len(MILESTONES_S) and time.monotonic() - t0 >= MILESTONES_S[milestone_i]:
                milestone_i += 1
            if milestone_i > stats["milestones"]:
                stats["milestones"] = milestone_i
                stats["files_changed_last"] = await observe.changed_files(world)
                messages.append({"role": "user", "content": milestone_note(
                    time.monotonic() - t0, stats["files_changed_last"],
                    observe.render_plan(plan))})
            used = model.input_tokens - before    # measured prompt length of that call
            stats["peak_prompt_tokens"] = max(stats["peak_prompt_tokens"], used)
            if used > trigger:                   # trim once the turn is on the record
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
