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
in tool-driven environments - so tool results carry escalating pace notes and
bash commands cannot hang for the whole budget; (4) round 3 showed the
round-2 end-of-run check fired only once and its escape hatch won: in every
failed tool-environment trial the final message was composed as bare text,
never sent, and the single soft nudge drew back an identical repeat. The
finish check now escalates over up to three nudges, names the environment's
delivery/termination tools, and tones itself by whether a delivery-style tool
has been called; old tool outputs are compacted before the token cap is hit.
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
EXEC_TIMEOUT_S = 420.0           # one foreground exec early in the run; the cap
EXEC_FLOOR_S = 150.0             # shrinks to this floor as the run ages so a hung
EXEC_SHRINK_PER_MIN = 15.0       # command always returns control with time to
                                 # recover and deliver before the unknown wall
COMPACT_AT_CHARS = 430_000       # ~110-140k tokens of true request weight; past
                                 # ~670k chars (incl. replayed reasoning) requests
                                 # have been refused by the gateway
WATCHDOG_S = 990.0               # per model call: a successful generation can need at most
                                 # 3 gateway samples (~300 s each); past this it is dead and
                                 # only the run's own timeout would notice
FIRST_CALL_WATCHDOG_S = 420.0    # before any tool call has landed there is nothing to
                                 # protect: a hung orientation call would eat a whole short
                                 # budget (observed: a 15-min run died inside call #1), while
                                 # successful first calls never exceeded ~5.5 min in 60 trials


class _WatchdogTimeout(Exception):
    """One model call produced nothing within WATCHDOG_S - longer than any
    successful generation could still take - so the attempt is dead. Without
    this, a hung call burns the rest of the run silently."""

BASH_TOOL = {"type": "function", "function": {
    "name": "bash",
    "description": ("Run a bash command in the task container and return its output."
                    " stdin is closed, and the exec is killed after a few minutes"
                    " (tighter late in the run). A backgrounded child keeps the exec"
                    " open: detach long or blocking jobs (`setsid nohup CMD >log 2>&1"
                    " </dev/null &`) and poll the log in later calls."),
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string", "description": "The command to run."}},
                   "required": ["command"]}}}

VIEW_IMAGE_TOOL = {"type": "function", "function": {
    "name": "view_image",
    "description": ("Show a local image file to yourself inside the conversation, so you can actually"
                    " see it. For video or PDFs, extract pages/frames to image files first (ffmpeg:"
                    " every N seconds `-vf fps=1/N,scale=800:-1 f_%03d.jpg`, scene changes"
                    " `-vf \"select=gt(scene\\,0.3),scale=800:-1\" -vsync vfr`, or a tiled contact sheet"
                    " `-vf fps=1/N,scale=320:-1,tile=4x5 sheet.jpg`); then view those files. Use for any"
                    " input you would need to look at: photos, screenshots, charts, diagrams, scoreboards."),
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "Path of the image file inside the container."},
                                  "max_side": {"type": "integer", "description": "Downscale so the longer side is at most this many pixels (default 1024)."}},
                   "required": ["path"]}}}

# Attached images are counted at a token-equivalent weight, not their base64
# length: vision encoders bill an image at hundreds-to-low-thousands of tokens
# regardless of byte size, while the base64 text is ~100-250k chars - counting
# it verbatim would make one picture look like half a context window.
IMG_CHAR_COST = 60_000
IMG_KEEP = 4              # newest attached images kept in history; older are pruned

_IMAGE_SNIPPET = r'''
import base64, io, sys
p, ms = sys.argv[1], int(sys.argv[2])
raw = open(p, "rb").read()
ext = p.rsplit(".", 1)[-1].lower() if "." in p else "png"
fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp", "bmp": "bmp"}.get(ext, "png")
data = raw
try:
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = max(w, h)
    if s > ms:
        im = im.resize((max(1, w * ms // s), max(1, h * ms // s)))
    for side in (ms, min(ms, 640)):
        w2, h2 = im.size
        s2 = max(w2, h2)
        im2 = im.resize((max(1, w2 * side // s2), max(1, h2 * side // s2))) if s2 > side else im
        b = io.BytesIO()
        im2.save(b, "JPEG", quality=80 if side == ms else 55)
        data, fmt = b.getvalue(), "jpeg"
        if len(b.getvalue()) <= 700000:
            break
except Exception:
    pass
sys.stdout.write("OK %s %s" % (fmt, base64.b64encode(data).decode()))
'''

_IMAGE_NOTE = ("[harness] The gateway rejected the attached image(s), so they were removed from the"
               " conversation. Pictures cannot be shown on this run: analyze media with in-container"
               " tools instead (ffmpeg frame extraction and filters, PIL pixel statistics, any OCR/"
               "captioning service the task provides) and reason from their textual output.")

VIEW_RESULT = "view_image: image loaded from %s (%s, downscaled); it is attached below as a picture you can see."


def _image_user_message(imgs):
    """One user message carrying this turn's attached images (text part first)."""
    content = [{"type": "text", "text": "[harness] Attached image(s) you asked to view:"}]
    for fmt, b64 in imgs:
        content.append({"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (fmt, b64)}})
    return {"role": "user", "content": content}


def _image_parts(messages):
    """Yield (message, index) for every attached image part in the history."""
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for j, part in enumerate(c):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    yield m, j


def _replace_image_part(m, j, note):
    m["content"][j] = {"type": "text", "text": note}


def _prune_images(messages, keep=IMG_KEEP):
    parts = list(_image_parts(messages))
    for m, j in parts[:-keep] if keep else parts:
        _replace_image_part(m, j, "[earlier image removed to save context]")


def _strip_all_images(messages):
    for m, j in list(_image_parts(messages)):
        _replace_image_part(m, j, "[image could not be displayed]")


async def view_image(world, args):
    """Load a container-side image, downscale it, and hand it back for attachment."""
    path = args.get("path") if isinstance(args, dict) else None
    if not isinstance(path, str) or not path.strip():
        return "view_image error: no path given"
    try:
        ms = max(256, min(2048, int(args.get("max_side") or 1024)))
    except Exception:
        ms = 1024
    q = shlex.quote
    enc = "%s -c %s" % (PY_CMD, q(_IMAGE_SNIPPET))
    # PIL-first with an ffmpeg fallback (which also turns a stray video/PDF
    # argument into its first frame/page instead of failing outright).
    cmd = ("OUT=$(%s %s %s 2>/dev/null); case \"$OUT\" in 'OK '*) printf '%%s' \"$OUT\"; exit 0;; esac; "
           "ffmpeg -v error -y -i %s -vf scale='min(%d,iw)':-2 /tmp/.harness_view.jpg 2>/dev/null"
           " && %s /tmp/.harness_view.jpg %s 2>/dev/null" % (enc, q(path), ms, q(path), ms, enc, ms))
    try:
        r = await asyncio.wait_for(world.exec(cmd, timeout_s=90), timeout=90)
    except Exception as exc:
        return "view_image error: %s: %s" % (type(exc).__name__, exc)
    out = (r.stdout or "").strip()
    if not out.startswith("OK ") or len(out) < 100:
        return ("view_image error: could not read %r. If it is a video or PDF, extract frames/pages"
                " to image files first (ffmpeg -i v.mp4 -vf fps=1/2 f_%%03d.jpg; pdftoppm for PDFs),"
                " then view those." % path[:200])
    try:
        fmt, b64 = out[3:].split(" ", 1)
    except ValueError:
        return "view_image error: bad encoder output"
    if not fmt or len(b64) > 1_600_000:
        return "view_image error: image too large to attach; extract a smaller frame (scale=640:-1) and view that."
    return {"format": fmt, "b64": b64, "path": path}

SYSTEM_PROMPT = """You are an autonomous agent completing a task in a Linux container. Work through tool calls
until the task is done, then reply with a final message and no tool calls.

Working method:
1. Pin down the deliverable before acting: exactly what artifact, answer, or effect is required (file paths, formats, names, recipients), and any constraints or acceptance criteria the task states.
2. Inventory before you build: look at the working directory, the inputs, and what is already installed (interpreters, packages, CLI tools); prefer what exists over downloading or rebuilding; sample data formats before parsing. If inputs include images, video, or PDFs, inspect them early - extract frames/pages (ffmpeg, pdftoppm) and use the view_image tool to actually look at them, rather than inferring from names, metadata, or pixel arithmetic.
3. Work in small verified steps: run a command, read the result, adjust. When something fails, read the actual error and change something - do not repeat the same call expecting a different result.
4. Be economical with output: search and slice (grep -n, sed -n, head, tail) instead of dumping whole files; batch related commands into one call.

   Bash exec semantics that bite: stdin arrives closed (a command that waits
   for input gets EOF and exits rather than hanging), and an exec does not
   return until every process it started exits - so a backgrounded server
   holds the exec open. Either keep such a job inside one bounded exec (start,
   use, kill) or detach it (`setsid nohup CMD >log 2>&1 </dev/null &`) and
   poll the log or port in later calls; wrap anything that could block in
   `timeout Ns` so a hang costs seconds, never the run.
5. Checkpoint the deliverable: as soon as a required artifact exists in acceptable form, write it to the requested location; refine afterwards if time allows. Never leave a task without its deliverable in place. The deliverable may be a file, a sent message, or your final reply - find out which and bank a defensible version of it early, then improve it. If part of the task cannot be done here (a tool, file, or network resource is missing), do not burn the budget trying to obtain it: deliver the best answer the environment allows and say plainly what is missing.
6. Verify before finishing: re-read the task's requirements and check each against what you actually produced (exact path, exact format, the value actually asked for); if a checker or test suite exists, run it. When the deliverable is a specific value, re-derive it once by an independent method before answering: carry full precision through intermediate steps, watch units and rounding conventions, and round only at the final step, exactly as the task asks.
7. When the task asks a question, make your final message state the exact answer plainly - the precise value, name, or phrase requested - not just reasoning around it.
8. Deliver through the environment's own channels: if the task provides tools for sending messages, submitting, saving, or terminating (or names a required file or channel), perform those acts with those tools; a plain-text reply reaches no one and simply ends the run. If a requirement is genuinely ambiguous and you can ask the user a question, ask one concise question early; otherwise proceed on the most conservative reading and state your assumption. If the task describes an interactive protocol (a person or service that replies, a limited number of exchanges), the interaction itself is part of the work: use the channel as the task spells out instead of answering from assumptions. Before finishing, check whether the instructions define an ending protocol (a stop token, or an end/terminate/submit call) and perform it.

Each tool result is clipped if very long (a marker shows what was dropped) and ends with the run's elapsed wall-clock time. Use that to pace yourself: your total budget is unknown and varies by task, so earn margin early - front-load the inventory, treat long installs, builds, or training runs as the wrong trade, and keep the deliverable present and valid in its required location from the first minutes so the run could end cleanly at any moment. As elapsed time grows, shift from exploring to finishing and verifying. Keep reasoning bursts short: prefer a small tool call now over long uninterrupted deliberation - you learn more per token from a tool result than from speculation, and very long single calls waste wall clock and can be cut off. If a reply of yours is ever cut off, or a single call runs many minutes, treat that as a signal to shrink every later reply to one small action and to move computation into scripts the machine runs instead of working it out in your head.

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


def exec_cap(t0):
    """Cap for one exec: generous while the run is young (installs, builds),
    shrinking to EXEC_FLOOR_S as it ages. No flat cap is safe against a hung
    command: the run's total budget is unknown, so late in a run only a small
    cap still leaves time to recover and deliver."""
    return max(EXEC_FLOOR_S, EXEC_TIMEOUT_S
               - EXEC_SHRINK_PER_MIN * ((time.monotonic() - t0) / 60.0))


# Escalating, advisory pace notes appended to tool results as a run ages. The
# budget is unknown (and varies per task), so these never claim a limit; they
# just keep the model converging and its deliverable valid on disk.
PACE_STEPS = (
    (6, "Checkpoint: the deliverable must exist in its required location right now - if it does"
        " not exist yet, write a provisional version now (marked provisional) and refine it later."),
    (12, "Shift weight from exploring to finishing; keep the deliverable valid on disk from here on."),
    (18, "Prefer completing and verifying exactly what was asked over broadening the work."),
    (26, "A lot of time has gone by: finish the remaining essentials, verify the deliverable in its"
         " final place, and end the run cleanly."),
    (40, "Wrap up now: finalize, verify, and finish rather than start anything new."),
)


def pace_note(t0, state):
    """Return the next unannounced advisory once; after 48 min restate it
    periodically. ``state`` is a small dict carried across calls."""
    m = (time.monotonic() - t0) / 60.0
    for threshold, text in PACE_STEPS:
        if m >= threshold and threshold not in state:
            state[threshold] = True
            return " [harness pace] " + text
    if m >= 40 and m - state.get("_last_firm", -12.0) >= 12.0:
        state[40] = True
        state["_last_firm"] = m
        return " [harness pace] " + PACE_STEPS[-1][1]
    return ""


# Tool-driven environments often need an explicit action (send, submit, save,
# terminate, or a stop token defined by the task) for a reply or a finish to
# take effect; a bare text reply reaches no one. Round-3 evidence: in every
# failed tool-environment trial the model ended with an unsent bare-text final
# message, and a single soft nudge drew back an identical repeat - the escape
# hatch won. So the check escalates: nudge once, then firmer, with the tone of
# the follow-up chosen by whether any delivery-style tool has been called.
DELIVERY_STEMS = ("send", "submit", "deliver", "reply", "end", "finish",
                  "terminate", "complete", "stop", "save", "close", "checkout")


def _delivery_candidates(tool_names):
    return [n for n in tool_names if any(s in n.lower() for s in DELIVERY_STEMS)][:8]


def finish_check(stage, delivery_seen, tool_names):
    cands = _delivery_candidates(tool_names)
    listing = (" Among your available tools, these look like delivery or termination"
               " actions: " + ", ".join("`%s`" % c for c in cands) + ".") if cands else ""
    if stage == 1:
        return ("[harness] Your reply contained no tool call, so the run is about to"
                " end - and plain text reaches no application or user. Before it"
                " closes, do whatever outstanding step needs a tool: send or submit"
                " the final message through this environment's own tools, or perform"
                " the ending protocol the task instructions define (an"
                " end/terminate/stop action or stop token)." + listing +
                " Only if you are certain no tool action is required, reply with"
                " this same final answer unchanged.")
    if delivery_seen:
        return ("[harness] Still no tool call. If every delivery and termination"
                " step is already done, reply with the same final answer unchanged"
                " and the run closes. If anything is still unsent, or the"
                " environment has not been told you are finished, call the tool"
                " that does it now.")
    return ("[harness] Still no tool call, so nothing you have written has reached"
            " the application or user, and the run ends after this. Deliver now:"
            " send your final message through this environment's send/submit-style"
            " tool, or call its end/terminate action, as the task requires." +
            listing + " If and only if no such tool exists here, reply with the"
            " same final answer.")

# A bare reply that stops mid-sentence was cut off before it finished; it is not
# a final answer and not a delivery question - whatever it was about to do next
# simply never happened. One nudge to continue beats ending the run there.
TRUNCATED_NOTE = (
    "[harness] Your previous reply stops mid-sentence: it was cut off before you"
    " finished, so whatever it was about to do or say never happened. Continue from"
    " exactly where it stopped - do not restart or re-summarize. If the task is in"
    " fact already complete, reply with the short final answer and no tool call.")


def _looks_truncated(text):
    t = (text or "").rstrip()
    return len(t) >= 40 and (t[-1].isalnum() or t[-1] in ",;:-")


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
        r = await asyncio.wait_for(
            world.exec("%s %s/bridge.py list %s" % (PY_CMD, MCP_DIR, _target_args(server)),
                       timeout_s=MCP_LIST_TIMEOUT_S + 30),
            timeout=MCP_LIST_TIMEOUT_S + 30)
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
    r = await asyncio.wait_for(world.exec(install, timeout_s=120), timeout=120)
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
        # Same harness-side bound as exec_bash: the bridge has its own internal
        # timeout for the tool call, so this only has to catch a wedged
        # transport - but it must not hang the run past the unknown wall either.
        r = await asyncio.wait_for(
            world.exec(cmd, timeout_s=MCP_CALL_TIMEOUT_S + 90),
            timeout=MCP_CALL_TIMEOUT_S + 90)
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


async def exec_bash(world, args, t0, state):
    command = args.get("command") if isinstance(args, dict) else None
    if not isinstance(command, str) or not command.strip():
        return "bash error: no command given"
    limit = exec_cap(t0)
    try:
        # stdin is closed up front so a stdin-reader gets EOF instead of hanging,
        # and wait_for enforces the cap harness-side: some transports ignore
        # their own timeout_s, and only control returning here turns a hung
        # command into a recoverable error instead of a wall-clock death.
        r = await asyncio.wait_for(
            world.exec("exec 0</dev/null; " + command, timeout_s=limit),
            timeout=limit)
    except asyncio.TimeoutError:
        state["exec_hung"] = True
        return ("bash error: killed after %ds (the cap tightens as the run ages);"
                " run it detached instead: `setsid nohup ... >log 2>&1 </dev/null &`,"
                " then poll the log - or wrap the command in `timeout Ns`, or narrow"
                " it" % limit)
    except Exception as exc:
        return ("bash error: %s: %s (the command was killed after %.0fs; rerun it in"
                " the background with `setsid nohup ... >log 2>&1 </dev/null &`, poll"
                " the log, or narrow the command)" % (type(exc).__name__, exc, limit))
    parts = [r.stdout]
    if r.stderr:
        parts.append("[stderr]\n" + r.stderr)
    parts.append("[exit %s]" % r.return_code)
    text = "\n".join(p for p in parts if p)
    low = text.lower()
    cmd_low = command.lower()
    installer = any(t in cmd_low for t in ("apt-get", "apt install", "apt-get install", "pip install",
                                           "pip3 install", "conda install", "npm install", "gem install",
                                           "cargo install", "apk add", "dnf install", "yum install"))
    failed = any(s in low for s in ("unable to locate package", "failed to fetch", "temporary failure resolving",
                                    "could not resolve host", "no matching distribution", "could not find a version",
                                    "network is unreachable", "could not connect", "connection refused"))
    if installer and failed and not state.get("net_dead"):
        # Every failed install costs a minute or more, and trials have burned a
        # third of their budget re-probing dead package sources.
        text += (" [harness] That install failed at the package source. Probe the network once, briefly;"
                 " if it is unavailable, stop attempting installs - they cannot succeed here. Work with"
                 " what is already installed (python stdlib, preinstalled binaries, the task's own local"
                 " services); a missing capability is not a reason to stop.")
        state["net_dead"] = True
    elif installer and failed and state.get("net_dead"):
        text += " [no network: installs will not work here]"
    return text


def _is_context_error(text):
    text = text.lower()
    return ("context length" in text or "context_length" in text or "too large" in text
            or "too long" in text or "input tokens" in text)


def _trim_arguments(a, keep=800):
    """Shorten an old tool-call's argument string to a valid-JSON prefix, so a
    gateway that parses arguments still accepts the replayed item."""
    cut = a[:keep]
    p = cut.rfind("}")
    if p > 0:
        try:
            json.loads(cut[:p + 1])
            return cut[:p + 1]
        except Exception:
            pass
    return "{\"trimmed\": true}"


def _shrink_history(messages, keep_last=6, floor=300):
    """Trim the heavy, stale parts of old history in place (never delete
    messages: the replay chain pairs every tool result with its call). The
    last ``keep_last`` messages stay verbatim; older ones lose long tool
    outputs, old tool-call arguments (whole files written inline) and the
    replayed per-turn reasoning blobs — the true bulk of a long reasoning-
    heavy request, which trimming tool outputs alone never touched."""
    for m in messages[:-keep_last] if keep_last else messages:
        role = m.get("role")
        if role == "tool":
            c = m.get("content")
            if isinstance(c, str) and len(c) > floor + 100:
                m["content"] = c[:floor] + "...[older output trimmed]"
        elif role == "assistant":
            _drop_replay(m)
            for tc in m.get("tool_calls") or []:
                a = (tc.get("function") or {}).get("arguments")
                if isinstance(a, str) and len(a) > 1000:
                    tc["function"]["arguments"] = _trim_arguments(a)
        if isinstance(m.get("content"), list):
            # Old attached pictures are stale too: keep the text, drop the bytes.
            for j, part in list(enumerate(m["content"])):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    _replace_image_part(m, j, "[earlier image removed to save context]")


def _drop_replay(m):
    """Remove the per-turn reasoning replay fields an assistant message may
    carry. They exist only to help the model continue its own train of thought
    across a tool call; old turns' reasoning is dead weight in every later
    request, and on reasoning-heavy runs it is most of the request."""
    for k in ("_responses_items", "reasoning", "reasoning_details"):
        m.pop(k, None)


def _history_chars(messages):
    """Estimate what a request really weighs: every serialized message field,
    not just ``content``. Tool-call arguments (whole files written inline) and
    the replayed reasoning blobs are invisible to a content-only count yet
    they are what filled the context window in the trials that died. Attached
    images count at their token-equivalent weight, not their base64 length."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            total += len(json.dumps({k: v for k, v in m.items() if k != "content"}, default=str))
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    total += IMG_CHAR_COST
                else:
                    total += len(json.dumps(part, default=str))
        else:
            total += len(json.dumps(m, default=str))
    return total


def _maybe_compact(messages):
    """Proactively trim old tool outputs as history grows, so the request stays
    far below the gateway's input-token cap on long runs."""
    if _history_chars(messages) > COMPACT_AT_CHARS:
        _shrink_history(messages, keep_last=10)


def _is_cutoff(text):
    return ("generationcutoff" in text or "generation limit" in text
            or "cut at" in text or "watchdogtimeout" in text)


CUTOFF_NOTES = (
    "[harness] Your previous generation never arrived: the gateway cut every sample of it"
    " (each attempt is cut at ~5 minutes), so the tool layer received nothing. This reply"
    " must be tiny: the single smallest useful next tool call and nothing else. If the step"
    " needs computation, write a short script and run it instead of working it out in your"
    " reply.",
    "[harness] Another generation was lost the same way. Reply with one short tool call -"
    " or, if the task is done, one short sentence - and no prose.",
)

REBUILD_NOTE = (
    "[harness] The request exceeded the model's context window even after"
    " trimming, so older turns were dropped and replayed reasoning cleared."
    " The task instruction is at the top; the recent history shows where you"
    " are. Continue from the current state without redoing finished work, and"
    " keep replies small.")


# Once generations start being cut off (or succeeding only after ~15 min of
# sampling), that property holds for the rest of the run - so unlike the one-shot
# retry notes above, the adaptation is persistent: it rides on every tool result
# and survives compaction because fresh results keep carrying it.
STALL_NOTE = (" [harness stall-watch] Generations on this run are being cut off or sampled for"
              " many minutes (the gateway cuts each attempt at ~5 min and re-samples up to 3x,"
              " so one oversized reply can cost ~15 min even when it eventually gets through)."
              " From here on keep every reply to one small tool call; do nontrivial computation"
              " by writing a short script and running it, never by working it out in your reply.")


# Same rule for execs as the stall note for generations: one hang marks the
# run, and later tool results keep carrying the fix so it survives compaction.
HANG_NOTE = (" [harness hang-watch] A foreground command was just killed for running too"
             " long. From here on start anything that could block or run for minutes in"
             " the background detached (`setsid nohup CMD >log 2>&1 </dev/null &`) and"
             " poll the log; keep foreground commands to quick, bounded steps.")


def _rebuild_history(messages, note):
    """Last-resort recovery from a context error that survives shrinking: keep
    the system prompt, the original instruction, and a clean tail of recent
    turns (starting at a non-tool message, so no tool result is orphaned from
    its call), with all replayed reasoning dropped. Rewrites the list in
    place; the recorder sees a rewritten history and logs it in full."""
    tail = list(messages[-12:])
    while len(tail) > 1 and tail[0].get("role") != "assistant":
        tail.pop(0)
    for m in tail:
        _drop_replay(m)
    messages[:] = [messages[0], messages[1], {"role": "user", "content": note}] + tail
    _shrink_history(messages, keep_last=2)


CTX_PRESSURE_50 = (" [harness context] The conversation is filling up: as it grows, older tool outputs and old per-turn reasoning are trimmed automatically. Persist anything you may need later (key values, paths, decisions, what has been ruled out) into a notes file in the container and re-read it if context is cut back; query narrowly (grep, head, tail) instead of dumping files into the transcript.")
CTX_PRESSURE_80 = (" [harness context] The request is nearing the compaction threshold: keep replies small, write large content to files in chunks, and move toward finalizing and verifying the deliverable.")


async def call_model(model, messages, tools, state):
    """model.chat with retries for transient provider failures.

    Every call sits under a workspace watchdog: a hung call would otherwise
    burn the rest of the run silently (the kernel's own timeout is longer).
    Generation cut-offs and watchdog kills are handled separately from
    ordinary transient errors: each costs minutes of gateway sampling, so the
    retry count is capped, and each retry first shrinks the history and
    injects a short behavioral demand as a user message. Injected notes are
    dropped again once a reply gets through, so they do not shape later turns;
    the persistent ``state["stalled"]`` note takes over that role instead.
    """
    attempt = 0
    cutoffs = 0
    while True:
        try:
            started = time.monotonic()
            wd = WATCHDOG_S if state.get("tool_calls") else FIRST_CALL_WATCHDOG_S
            try:
                reply = await asyncio.wait_for(model.chat(messages, tools=tools), timeout=wd)
            except asyncio.TimeoutError:
                raise _WatchdogTimeout("watchdogtimeout: no generation within %ds" % wd) from None
            if time.monotonic() - started > 600.0:
                # Succeeded, but only after ~2 wasted gateway samples: the run
                # will keep paying that price unless the replies get smaller.
                state["stalled"] = True
            # One-shot notes are dropped once a reply gets through (matched by
            # their exact harness-written text, not by index: a rebuild may
            # have rewritten the list in between; no foreign key is added to
            # any message, so the request stays within what the client sends).
            messages[:] = [m for m in messages if m.get("content") not in CUTOFF_NOTES]
            return reply
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            text = ("%s: %s" % (type(exc).__name__, exc)).lower()
            if _is_context_error(text) and attempt <= 4:
                attempt += 1
                if attempt >= 3:
                    # A plain shrink already ran and did not clear it: rebuild
                    # to a guaranteed-small, structurally clean request rather
                    # than resending the same oversized one.
                    _rebuild_history(messages, REBUILD_NOTE)
                else:
                    _shrink_history(messages)
                continue
            if list(_image_parts(messages)) and any(
                    s in text for s in ("image", "media", "multimodal", "content part", "data url",
                                        "invalid", "unsupported", "not supported", "reject",
                                        "400", "413", "415", "422")):
                # The request that just failed carried picture parts the
                # gateway would not take (or malformed ones). Strip them once
                # and carry on text-only instead of dying here; transient
                # failures (5xx, timeouts, rate limits) do not match and keep
                # the images.
                attempt += 1
                _strip_all_images(messages)
                messages.append({"role": "user", "content": _IMAGE_NOTE})
                continue
            if _is_cutoff(text):
                state["stalled"] = True
                if cutoffs < 2:
                    attempt += 1
                    _shrink_history(messages, keep_last=2 if cutoffs else 6)
                    messages.append({"role": "user", "content": CUTOFF_NOTES[cutoffs]})
                    cutoffs += 1
                    continue
                raise
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
    tools = [BASH_TOOL, VIEW_IMAGE_TOOL] + [
        {"type": "function", "function": {"name": t["tool_name"], "description": t["description"],
                                          "parameters": t["parameters"]}} for t in mcp_tools]

    messages = [{"role": "system", "content": system}, {"role": "user", "content": instruction}]
    repeat = {"keys": []}
    pace = {}
    state = {"stalled": False,   # generations being cut off / sampled for many minutes
             "tool_calls": 0,    # completed tool calls; selects the watchdog deadline
             "exec_hung": False} # an exec was killed at the cap; ride the hang-watch note
    pending_images = []   # (format, b64) loaded by view_image this turn; attached after tool results
    nudges = 0            # finish-check nudges spent over the whole run
    bare_streak = 0       # consecutive replies without a tool call
    delivery_seen = False # a delivery/termination-style tool has been called
    degraded = False
    emergency_used = False
    while True:
        try:
            reply = await call_model(model, messages, tools, state)
        except Exception as exc:
            made_progress = len(messages) > 2  # more than system+user: a call got through
            if made_progress and not emergency_used:
                # A fatal error escaped mid-run (stalled cut-offs, dead
                # connection). Everything so far would be lost; one bounded
                # retry on a changed request is strictly better than dying.
                # Tools are kept: the failure is provider-side, not theirs.
                emergency_used = True
                _shrink_history(messages, keep_last=2)
                messages.append({"role": "user", "content":
                    "[harness] The previous request failed at the provider (%s), so"
                    " nothing was lost except time. Continue the task: reply with"
                    " the single next tool call, kept short." % type(exc).__name__})
                reply = await call_model(model, messages, tools, state)
            elif mcp_tools and not degraded:
                # Nothing has happened yet in this run: if task tools look like
                # the problem (e.g. the gateway rejects their schemas), fall
                # back to bash-only once instead of ending with zero progress.
                degraded = True
                mcp_notes.append("task tools dropped after %s: %s" % (type(exc).__name__, exc))
                messages[0]["content"] += ("\n\nMCP status: " + " ".join(mcp_notes)
                                           + "\nIf the task depends on these servers, connect to"
                                             " them yourself via bash; the task text names their"
                                             " endpoints and transports.")
                tools = [BASH_TOOL, VIEW_IMAGE_TOOL]
                mcp_tools, mcp_by_name = [], {}
                reply = await call_model(model, messages, tools, state)
            else:
                raise
        messages.append(reply.message)
        if not reply.tool_calls:
            # A reply cut off mid-sentence is not a final message at all: nudge
            # to continue regardless of environment (bash-only runs included,
            # which the delivery check below deliberately does not cover).
            if _looks_truncated(reply.text) and nudges < 3:
                nudges += 1
                messages.append({"role": "user", "content": TRUNCATED_NOTE})
                continue
            # Bare text in a tool environment reaches no one. Nudge, escalate,
            # and nudge once more if needed; whatever follows the last nudge
            # (or a third bare reply) ends the run.
            bare_streak += 1
            if mcp_tools and bare_streak < 3 and nudges < 3:
                nudges += 1
                messages.append({"role": "user", "content": finish_check(
                    1 if nudges == 1 else 2, delivery_seen, sorted(mcp_by_name))})
                continue
            return
        bare_streak = 0
        for call in reply.tool_calls:
            args = call.arguments if isinstance(call.arguments, dict) else {}
            if call.name != "bash" and any(s in call.name.lower() for s in DELIVERY_STEMS):
                delivery_seen = True
            if call.name == "bash":
                out = await exec_bash(world, args, t0, state)
            elif call.name == "view_image":
                out = await view_image(world, args)
                if isinstance(out, dict):
                    pending_images.append((out["format"], out["b64"]))
                    out = VIEW_RESULT % (out["path"], out["format"])
            elif call.name in mcp_by_name:
                out = await call_mcp(world, mcp_by_name[call.name], args)
            else:
                out = "unknown tool %r; available: bash, MCP tools: %s" % (
                    call.name, ", ".join(sorted(mcp_by_name)) or "none")
            out = clip_output(out)
            state["tool_calls"] += 1
            key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=True)[:400], out[:400])
            repeat["keys"] = (repeat["keys"] + [key])[-3:]
            if len(repeat["keys"]) == 3 and len(set(repeat["keys"])) == 1:
                out += ("\n[loop guard] This call and its result are nearly identical to the"
                        " previous ones. If you are polling for a change, poll differently (or"
                        " wait first); otherwise stop repeating: change something in how you"
                        " attack this, or finish the task.")
            pressure = (_history_chars(messages) + len(out)) / COMPACT_AT_CHARS
            if "ctx80" not in pace and pressure >= 0.8:
                pace["ctx80"] = True
                out += CTX_PRESSURE_80
            elif "ctx50" not in pace and pressure >= 0.5:
                pace["ctx50"] = True
                out += CTX_PRESSURE_50
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": out + pace_note(t0, pace)
                             + (STALL_NOTE if state["stalled"] else "")
                             + (HANG_NOTE if state["exec_hung"] else "")
                             + "\n" + time_tag(t0)})
            _maybe_compact(messages)
        if pending_images:
            # Attach after the tool results, so the replay chain stays intact:
            # assistant(tool_calls) -> tool results -> one user message with pictures.
            messages.append(_image_user_message(pending_images))
            pending_images = []
            _prune_images(messages)
            _maybe_compact(messages)
