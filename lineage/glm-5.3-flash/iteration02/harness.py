"""Agent harness: one bash tool plus, when the task declares MCP servers, those
servers' tools exposed natively. Shaped by past trials: (1) MCP tasks burned
5-10 model calls hand-rolling MCP clients over bash, with avoidable failures -
now a stdlib-only bridge is installed in the container before the first model
call and each server's tools are exposed with their real schemas, degrading to
bash-only with a note if that fails; (2) provider errors are retried, but
gateway generation cut-offs get dedicated handling (each one costs ~15 min of
sampling, so they are capped, and the history is shrunk plus a brevity note is
injected before the retry instead of resending the same request); (3) runs
died cancelled-with-deliverable or ended with the reply trapped as plain text
in tool-driven environments - so tool results carry escalating pace notes, an
end-of-run check gives one chance to act through the environment's own tools,
old tool outputs are compacted before the token cap is hit, and bash commands
cannot hang for the whole budget.
"""

import asyncio
import json
import shlex
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRIDGE_SOURCE = (HERE / "mcp_bridge.py").read_text(encoding="utf-8")

MCP_DIR = "/tmp/.harness_mcp"
BRIDGE_DELIM = "HARNESS_BRIDGE_EOF_7c31"
MCP_LIST_TIMEOUT_S = 120.0
MCP_CALL_TIMEOUT_S = 600.0
TOOL_OUTPUT_LIMIT = 40000
TOOL_HEAD = 26000
TOOL_TAIL = 10000
MCP_NOTE_HEADER = "MCP servers reachable from inside the task container:"
EXEC_TIMEOUT_S = 1800.0          # one command must not be able to eat the run
COMPACT_AT_CHARS = 700_000       # stay well under the gateway's input-token cap

BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": ("Run a bash command in the task container and return its output."
                    " A command is killed after 30 minutes; run long jobs in the"
                    " background (nohup ... >log 2>&1 &) and poll the log instead."),
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string", "description": "The command to run."}},
                   "required": ["command"]}}}

SYSTEM_PROMPT = """You are an autonomous agent completing a task in a Linux container. Work through tool calls
until the task is done, then reply with a final message and no tool calls.

Working method:
1. Pin down the deliverable before acting: exactly what artifact, answer, or effect is required (file paths, formats, names, recipients), and any constraints or acceptance criteria the task states.
2. Inventory before you build: look at the working directory, the inputs, and what is already installed (interpreters, packages, CLI tools); prefer what exists over downloading or rebuilding; sample data formats before parsing.
3. Work in small verified steps: run a command, read the result, adjust. When something fails, read the actual error and change something - do not repeat the same call expecting a different result.
4. Be economical with output: search and slice (grep -n, sed -n, head, tail) instead of dumping whole files; batch related commands into one call.
5. Checkpoint the deliverable: as soon as a required artifact exists in acceptable form, write it to the requested location; refine afterwards if time allows. Never leave a task without its deliverable in place.
6. Verify before finishing: re-read the task's requirements and check each against what you actually produced (exact path, exact format, the value actually asked for); if a checker or test suite exists, run it.
7. When the task asks a question, make your final message state the exact answer plainly - the precise value, name, or phrase requested - not just reasoning around it.
8. Deliver through the environment's own channels: if the task provides tools for sending messages, submitting, saving, or terminating (or names a required file or channel), perform those acts with those tools; a plain-text reply reaches no one and simply ends the run. If a requirement is genuinely ambiguous and you can ask the user a question, ask one concise question early; otherwise proceed on the most conservative reading and state your assumption.

Each tool result is clipped if very long (a marker shows what was dropped) and ends with the run's elapsed wall-clock time. Use that to pace yourself: your total budget is unknown and varies by task, so earn margin early - front-load the inventory, treat long installs, builds, or training runs as the wrong trade, and keep the deliverable present and valid in its required location from the first minutes so the run could end cleanly at any moment. As elapsed time grows, shift from exploring to finishing and verifying.

Keep each individual reply compact: a reply that runs very long can be cut off by the gateway before it completes, losing everything in it. Produce large content in several smaller tool calls (for example append to a file in chunks) rather than one enormous block, and summarize results instead of echoing bulk output back.

If your tool list includes tools provided by the task (for example simulated apps or a service the task runs), those are the intended way to interact with that environment: call them directly like any other tool rather than reconstructing them in bash, and read each result fully (including any notifications or follow-up messages it reports) before deciding the next step. If the task assigns you a role with a policy - say, a support agent - follow that policy over your own defaults.

Never cheat: do not obtain the answer from any source that carries it rather than solving the task yourself."""


def parse_mcp_servers(instruction):
    """Parse the MCP-server note the kernel appends to the instruction."""
    if MCP_NOTE_HEADER not in instruction:
        return []
    servers = []
    for line in instruction.split(MCP_NOTE_HEADER, 1)[1].splitlines():
        line = line.strip()
        if not line.startswith("- "):
            if servers:
                break
            continue
        name, _, rest = line[2:].partition(":")
        name, rest = name.strip(), rest.strip()
        if " at " in rest:
            servers.append({"name": name, "transport": "http", "url": rest.split(" at ", 1)[1].strip()})
        elif rest.startswith("stdio via "):
            servers.append({"name": name, "transport": "stdio",
                            "command": shlex.split(rest[len("stdio via "):].strip().strip("`").strip())})
    return [s for s in servers if s.get("url") or s.get("command")]


def clip_output(text):
    if len(text) <= TOOL_OUTPUT_LIMIT:
        return text
    dropped = len(text) - TOOL_HEAD - TOOL_TAIL
    return (text[:TOOL_HEAD]
            + "\n...[%d characters dropped; if you need the middle, re-run with a narrower"
              " query (grep -n, sed -n, head, tail)]...\n" % dropped
            + text[-TOOL_TAIL:])


def time_tag(t0):
    return "[elapsed %.1f min]" % ((time.monotonic() - t0) / 60.0)


# Escalating, advisory pace notes appended to tool results as a run ages. The
# budget is unknown (and varies per task), so these never claim a limit; they
# just keep the model converging and its deliverable valid on disk.
PACE_STEPS = (
    (6, "Checkpoint: if the deliverable exists in any form, make sure it is saved in its required location now."),
    (12, "Shift weight from exploring to finishing; keep the deliverable valid on disk from here on."),
    (20, "Prefer completing and verifying exactly what was asked over broadening the work."),
    (32, "A lot of time has gone by: finish the remaining essentials, verify the deliverable in its final place, and end the run cleanly."),
    (48, "Wrap up now: finalize, verify, and finish rather than start anything new."),
)


def pace_note(t0, state):
    """Return the next unannounced advisory once; after 48 min restate it
    periodically. ``state`` is a small dict carried across calls."""
    m = (time.monotonic() - t0) / 60.0
    for threshold, text in PACE_STEPS:
        if m >= threshold and threshold not in state:
            state[threshold] = True
            return " [harness pace] " + text
    if m >= 48 and m - state.get("_last_firm", -12.0) >= 12.0:
        state[48] = True
        state["_last_firm"] = m
        return " [harness pace] " + PACE_STEPS[-1][1]
    return ""


# Tool-driven environments often need an explicit action (send, submit, save,
# terminate) for a reply or a finish to take effect; a bare text reply reaches
# no one. When the model ends with plain text, give it one chance to act before
# the run closes.
FINISH_CHECK = (
    "[harness] Your reply contained no tool call, so the run is about to end."
    " In this environment a plain-text reply does not reach any application or"
    " user. If any required step is still outstanding - sending a message,"
    " submitting, saving, or terminating through this environment's own tools -"
    " do it now. If the text above is already the complete final answer and no"
    " tool action is needed, reply with that same final answer unchanged."
)

def _last_json(text):
    for line in reversed((text or "").splitlines()):
        if line.strip().startswith("{"):
            try:
                out = json.loads(line)
            except Exception:
                continue
            if isinstance(out, dict):
                return out
    return None


PY_CMD = "$(command -v python3 || command -v python)"

def _target_args(server):
    if server["transport"] == "http":
        return "--url " + shlex.quote(server["url"])
    return "--stdio " + " ".join(shlex.quote(c) for c in server["command"])


def _safe_name(name, existing):
    clean = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))[:60] or "tool"
    taken = {t["tool_name"] for t in existing}
    if clean not in taken:
        return clean
    i = 2
    while "%s_%d" % (clean, i) in taken:
        i += 1
    return "%s_%d" % (clean, i)


def _params(schema):
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


async def _list_tools(world, server):
    last = "no attempt"
    for _ in range(2):
        r = await world.exec("%s %s/bridge.py list %s" % (PY_CMD, MCP_DIR, _target_args(server)),
                             timeout_s=MCP_LIST_TIMEOUT_S + 30)
        env = _last_json(r.stdout)
        if env and env.get("ok"):
            return env.get("tools") or []
        last = ((env or {}).get("error") if env else None) or (r.stderr or r.stdout or "no output")[-300:]
        await asyncio.sleep(2.0)
    return "tool listing failed: %s" % last


async def setup_mcp(world, servers):
    """Write the bridge into the container; discover each server's tools.
    Returns (tools, notes) where tools carry tool_name/mcp_name/description/
    parameters/server and notes describe anything that failed."""
    install = ("mkdir -p %(dir)s && cat > %(dir)s/bridge.py <<'%(delim)s'\n%(src)s\n%(delim)s\n"
               "PY=$(command -v python3 || command -v python); echo \"PY=$PY\""
               % {"dir": MCP_DIR, "delim": BRIDGE_DELIM, "src": BRIDGE_SOURCE})
    r = await world.exec(install, timeout_s=120)
    if r.return_code != 0 or "PY=" not in (r.stdout or ""):
        return [], ["bridge install failed (rc=%s): %s"
                    % (r.return_code, (r.stderr or r.stdout or "")[-300:])]
    tools, notes = [], []
    for s in servers:
        got = await _list_tools(world, s)
        if isinstance(got, str):
            notes.append("server %s: %s" % (s["name"], got))
            continue
        if not got:
            notes.append("server %s: no tools listed" % s["name"])
        for t in got:
            tools.append({"tool_name": _safe_name(t["name"], tools), "mcp_name": t["name"],
                          "description": (t.get("description") or "")[:2000],
                          "parameters": _params(t.get("inputSchema")), "server": s})
    return tools, notes


async def call_mcp(world, tool, args):
    cmd = "%s %s/bridge.py call %s --name %s --args %s --timeout %s" % (
        PY_CMD, MCP_DIR, _target_args(tool["server"]), shlex.quote(tool["mcp_name"]),
        shlex.quote(json.dumps(args or {})), MCP_CALL_TIMEOUT_S)
    try:
        r = await world.exec(cmd, timeout_s=MCP_CALL_TIMEOUT_S + 90)
    except Exception as exc:
        return "MCP transport error: %s: %s" % (type(exc).__name__, exc)
    env = _last_json(r.stdout)
    if not env:
        return "MCP error: no usable response (exit %s); stderr: %s" % (
            r.return_code, (r.stderr or "")[-300:])
    if not env.get("ok"):
        return "MCP error: %s" % json.dumps(env.get("error"), ensure_ascii=True)[:2000]
    parts = [item.get("text") or "" if isinstance(item, dict) and item.get("type") == "text"
             else "[non-text content: %s]" % json.dumps(item, ensure_ascii=True)[:200]
             for item in env.get("content") or []]
    out = "\n".join(p for p in parts if p)
    structured = env.get("structuredContent")
    if structured is not None and structured != {}:
        out = (out + "\n" if out else "") + json.dumps(structured, ensure_ascii=True)
    if not out:
        out = "(tool returned no content)"
    return ("The tool reported an error:\n" if env.get("isError") else "") + out


async def exec_bash(world, args):
    command = args.get("command") if isinstance(args, dict) else None
    if not isinstance(command, str) or not command.strip():
        return "bash error: no command given"
    try:
        r = await world.exec(command, timeout_s=EXEC_TIMEOUT_S)
    except Exception as exc:
        return ("bash error: %s: %s (the command was killed after %ds; rerun it in the"
                " background with `nohup ... >log 2>&1 &`, poll the log, or narrow the"
                " command)" % (type(exc).__name__, exc, EXEC_TIMEOUT_S))
    parts = [r.stdout]
    if r.stderr:
        parts.append("[stderr]\n" + r.stderr)
    parts.append("[exit %s]" % r.return_code)
    return "\n".join(p for p in parts if p)


def _is_context_error(text):
    text = text.lower()
    return ("context length" in text or "context_length" in text or "too large" in text
            or "too long" in text or "input tokens" in text)


def _shrink_history(messages, keep_last=6, floor=300):
    """Trim old tool outputs in place (never delete messages: the replay chain
    pairs every tool result with its call). Used to recover from context
    overflow and to shrink a request before retrying after a cut-off."""
    n_tool = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            n_tool += 1
            if n_tool > keep_last:
                c = messages[i].get("content")
                if isinstance(c, str) and len(c) > floor + 100:
                    messages[i]["content"] = c[:floor] + "...[older output trimmed]"


def _history_chars(messages):
    return sum(len(m.get("content") or "") for m in messages
               if isinstance(m.get("content"), str))


def _maybe_compact(messages):
    """Proactively trim old tool outputs as history grows, so the request stays
    far below the gateway's input-token cap on long runs."""
    if _history_chars(messages) > COMPACT_AT_CHARS:
        _shrink_history(messages, keep_last=10)


def _is_cutoff(text):
    return ("generationcutoff" in text or "generation limit" in text
            or "cut at" in text)


CUTOFF_NOTES = (
    "[harness] Your previous reply was cut off by the gateway before it finished"
    " (too long), so it never reached the tool layer. This reply must be much"
    " shorter: prefer one concise tool call over prose, split large content"
    " across several calls, and never emit a whole file in one response when"
    " writing it in appends would do.",
    "[harness] A second reply was also cut off. Reply with the single smallest"
    " useful next tool call - or, if the task is done, a one-paragraph final"
    " answer - and nothing else.",
)


async def call_model(model, messages, tools):
    """model.chat with retries for transient provider failures.

    Generation cut-offs are handled separately from ordinary transient errors:
    each costs minutes of gateway sampling, so resending the identical request
    is capped at one retry per stage, and each retry first shrinks the history
    and injects a brevity demand. Injected notes are dropped again once a reply
    gets through, so they do not shape the rest of the run.
    """
    attempt = 0
    cutoffs = 0
    injected = []
    while True:
        try:
            reply = await model.chat(messages, tools=tools)
            for i in reversed(injected):
                del messages[i]  # the note did its job; do not shape later turns
            return reply
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text = ("%s: %s" % (type(exc).__name__, exc)).lower()
            if _is_context_error(text) and attempt <= 3:
                attempt += 1
                _shrink_history(messages)
                continue
            if _is_cutoff(text) and cutoffs < 2:
                attempt += 1
                _shrink_history(messages, keep_last=2 if cutoffs else 6)
                injected.append(len(messages))
                messages.append({"role": "system", "content": CUTOFF_NOTES[cutoffs]})
                cutoffs += 1
                continue
            attempt += 1
            transient = isinstance(exc, json.JSONDecodeError) or any(
                s in text for s in ("generationcutoff", "cutoff", "timeout", "timed out",
                                    "rate limit", "429", "502", "503", "504", "overloaded",
                                    "connection", "reset by peer", "temporarily",
                                    "unavailable", "incomplete", "eof"))
            if not transient or attempt >= 3:
                raise
            await asyncio.sleep(min(20.0, 3.0 * attempt))


async def run(instruction, model, world):
    t0 = time.monotonic()
    servers = parse_mcp_servers(instruction)
    try:
        mcp_tools, mcp_notes = (await setup_mcp(world, servers) if servers else ([], []))
    except Exception as exc:
        mcp_tools, mcp_notes = [], ["setup failed: %s: %s" % (type(exc).__name__, exc)]

    system = SYSTEM_PROMPT
    if mcp_notes:
        system += ("\n\nMCP status: " + " ".join(mcp_notes)
                   + "\nIf the task depends on these servers, connect to them yourself via"
                     " bash; the task text names their endpoints and transports.")
    mcp_by_name = {t["tool_name"]: t for t in mcp_tools}
    tools = [BASH_TOOL] + [
        {"type": "function", "function": {"name": t["tool_name"], "description": t["description"],
                                          "parameters": t["parameters"]}} for t in mcp_tools]

    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    repeat = {"keys": []}
    pace = {}
    finish_checked = False
    degraded = False
    while True:
        try:
            reply = await call_model(model, messages, tools)
        except Exception as exc:
            # Nothing has happened yet in this run: if task tools look like the
            # problem (e.g. the gateway rejects their schemas), fall back to
            # bash-only once instead of ending with zero progress.
            if not mcp_tools or degraded:
                raise
            degraded = True
            mcp_notes.append("task tools dropped after %s: %s" % (type(exc).__name__, exc))
            messages[0]["content"] += ("\n\nMCP status: " + " ".join(mcp_notes)
                                       + "\nIf the task depends on these servers, connect to"
                                         " them yourself via bash; the task text names their"
                                         " endpoints and transports.")
            tools = [BASH_TOOL]
            mcp_tools, mcp_by_name = [], {}
            reply = await call_model(model, messages, tools)
        messages.append(reply.message)
        if not reply.tool_calls:
            if mcp_tools and not finish_checked:
                # One chance to deliver/terminate through the environment's own
                # tools; whatever comes back ends the run.
                finish_checked = True
                messages.append({"role": "user", "content": FINISH_CHECK})
                continue
            return
        for call in reply.tool_calls:
            args = call.arguments if isinstance(call.arguments, dict) else {}
            if call.name == "bash":
                out = await exec_bash(world, args)
            elif call.name in mcp_by_name:
                out = await call_mcp(world, mcp_by_name[call.name], args)
            else:
                out = "unknown tool %r; available: bash, MCP tools: %s" % (
                    call.name, ", ".join(sorted(mcp_by_name)) or "none")
            out = clip_output(out)
            key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=True)[:400], out[:400])
            repeat["keys"] = (repeat["keys"] + [key])[-3:]
            if len(repeat["keys"]) == 3 and len(set(repeat["keys"])) == 1:
                out += ("\n[loop guard] This call and its result are nearly identical to the"
                        " previous ones. If you are polling for a change, poll differently (or"
                        " wait first); otherwise stop repeating: change something in how you"
                        " attack this, or finish the task.")
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": out + pace_note(t0, pace) + "\n" + time_tag(t0)})
            _maybe_compact(messages)
