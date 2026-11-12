"""Agent harness: one bash tool plus, when the task declares MCP servers, those
servers' tools exposed natively. Shaped by past trials: (1) MCP tasks burned
5-10 model calls hand-rolling MCP clients over bash, with avoidable failures -
now a stdlib-only bridge is installed in the container before the first model
call and each server's tools are exposed with their real schemas, degrading to
bash-only with a note if that fails; (2) provider generation cutoffs used to
kill runs permanently - transient model-call failures are retried and a
context overflow trims old tool outputs; (3) tool results are clipped, tagged
with elapsed wall-clock time, and identical repeated calls get a loop guard.
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

BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": "Run a bash command in the task container and return its output.",
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

Each tool result is clipped if very long (a marker shows what was dropped) and ends with the run's elapsed wall-clock time. Use that to pace yourself: leave a margin for verification and delivery instead of exploring until you are cut off. Long installs, builds, or training runs are usually the wrong trade; get a working deliverable first, improve it after.

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
        r = await world.exec(command)
    except Exception as exc:
        return "bash error: %s: %s" % (type(exc).__name__, exc)
    parts = [r.stdout]
    if r.stderr:
        parts.append("[stderr]\n" + r.stderr)
    parts.append("[exit %s]" % r.return_code)
    return "\n".join(p for p in parts if p)


def _is_context_error(text):
    text = text.lower()
    return ("context length" in text or "context_length" in text or "too large" in text
            or "too long" in text or "input tokens" in text)


def _shrink_history(messages, keep_last=6):
    """Trim old tool outputs in place so a context-overflow retry can fit."""
    n_tool = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            n_tool += 1
            if n_tool > keep_last:
                c = messages[i].get("content")
                if isinstance(c, str) and len(c) > 400:
                    messages[i]["content"] = c[:300] + "...[older output trimmed]"


async def call_model(model, messages, tools):
    """model.chat with retries for transient provider failures."""
    attempt = 0
    while True:
        try:
            return await model.chat(messages, tools=tools)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            attempt += 1
            text = ("%s: %s" % (type(exc).__name__, exc)).lower()
            if _is_context_error(text) and attempt <= 3:
                _shrink_history(messages)
                continue
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
    mcp_tools, mcp_notes = (await setup_mcp(world, servers) if servers else ([], []))

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
    repeat = {"key": None, "count": 0}
    while True:
        reply = await call_model(model, messages, tools)
        messages.append(reply.message)
        if not reply.tool_calls:
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
            repeat["count"] = repeat["count"] + 1 if key == repeat["key"] else 1
            repeat["key"] = key
            if repeat["count"] >= 3:
                out += ("\n[loop guard] This call and its result are nearly identical to the"
                        " previous %d. If you are polling for a change, poll differently (or"
                        " wait first); otherwise stop repeating: change something in how you"
                        " attack this, or finish the task." % repeat["count"])
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": out + "\n" + time_tag(t0)})
