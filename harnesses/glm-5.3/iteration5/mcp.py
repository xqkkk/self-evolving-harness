"""Bridge the task's MCP servers into ordinary model tools.

The kernel appends to the instruction a note naming the MCP servers the task
declared — a URL, or a stdio command — reachable from inside the task
container, not from here. This module finds them, speaks JSON-RPC through
`world.exec` (a helper script installed in the container once per run), and
re-publishes their tools as plain function tools, so the model calls them
directly instead of hand-rolling `curl` with shell-quoted JSON, which is where
a large share of turns used to go. Discovery stays loose on purpose: the
note's wording belongs to the kernel, so servers are read off the note's shape
and off any obvious endpoint URL in the instruction; one that will not
initialize is dropped and left for the model to reach by hand.
"""

import json
import shlex

HELPER_PATH = "/tmp/.harness_mcp.py"

HELPER = r'''
import json, os, select, subprocess, sys, time, urllib.error, urllib.request

def http(url, body, session, timeout):
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        h["mcp-session-id"] = session
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (r.headers.get("mcp-session-id") or session, r.status,
                    r.read().decode("utf-8", "replace"), "")
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:
            raw = ""
        return session, e.code, raw, ""
    except Exception as e:
        return session, 0, "", "%s: %s" % (type(e).__name__, e)

def msgs_of(raw):
    out = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if s.startswith("data:"):
            s = s[5:].strip()
        elif not s.startswith("{"):
            continue
        if s and s != "[DONE]":
            try:
                out.append(json.loads(s))
            except Exception:
                pass
    return out

def rpc(url, session, bodies, timeout):
    """Send each body in turn, keeping every reply; `want` picks ours out."""
    sid, status, msgs, err = session, 0, [], ""
    for b in bodies:
        sid, status, raw, err = http(url, b, sid, timeout)
        if err or status >= 400:
            break
        msgs += msgs_of(raw)
    return sid, status, msgs, err

def stdio(cmd, bodies, timeout):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.time() + timeout
    for b in bodies:
        p.stdin.write((json.dumps(b) + "\n").encode())
    p.stdin.flush()
    out = err = b""
    while time.time() < deadline:
        ready, _, _ = select.select([p.stdout, p.stderr], [], [], 0.5)
        if not ready and p.poll() is not None:
            break
        for f in ready:
            chunk = os.read(f.fileno(), 65536)
            if chunk:
                out, err = (out + chunk, err) if f is p.stdout else (out, err + chunk)
        if any(m.get("id") == bodies[-1]["id"] for m in msgs_of(out.decode("utf-8", "replace"))):
            break
    try:
        p.kill()
    except Exception:
        pass
    return msgs_of(out.decode("utf-8", "replace")), err.decode("utf-8", "replace")[-2000:]

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-03-26", "capabilities": {},
    "clientInfo": {"name": "harness", "version": "1"}}}

def main():
    a = sys.argv[1:]
    if a[0] in ("list", "call"):
        url, timeout = a[1], float(a[2])
        sid = a[3] if len(a) > 3 and a[3] != "-" else None
        bodies = [INIT, {"jsonrpc": "2.0", "method": "notifications/initialized"}]
        if a[0] == "list":
            bodies.append({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            want = 2
        else:
            bodies.append({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                           "params": {"name": a[4], "arguments": json.loads(a[5] or "{}")}})
            want = 3
        sid, status, msgs, err = rpc(url, sid, bodies, timeout)
        res = next((m.get("result") for m in msgs if m.get("id") == want), None)
        rerr = [m.get("error") for m in msgs if m.get("error")]
        out = {"ok": res is not None, "session": sid, "status": status,
               "error": err or (rerr[-1] if rerr else None)}
        out["tools" if want == 2 else "result"] = (res or {}).get("tools") or [] if want == 2 else res
        print(json.dumps(out))
    elif a[0] == "stdio":
        cmd, timeout, name = a[1], float(a[2]), (a[3] if len(a) > 3 else "")
        bodies = [INIT, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                  {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        if name:
            bodies.append({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                           "params": {"name": name, "arguments": json.loads(a[4] or "{}")}})
        msgs, err = stdio(cmd, bodies, timeout)
        res = next((m.get("result") for m in msgs if m.get("id") == 3), None)
        tools = next((m.get("result") for m in msgs if m.get("id") == 2), None)
        print(json.dumps({"ok": res is not None or tools is not None,
                          "tools": (tools or {}).get("tools") or [],
                          "result": res, "stderr": err}))
    else:
        print('{"ok": false, "error": "bad mode"}')

main()
'''


def _last_json(stdout):
    """The helper prints one JSON object; find it without trusting the rest."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"ok"' in line:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _safe(name):
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)


def _coerce(schema, arguments):
    """Nudge argument values to the types the tool's own schema declares.

    Models habitually quote numbers, booleans, lists and objects, and a typed
    server then answers with an error the model works around by repeating the
    call — which changes the world twice. Only lossless moves are made here; a
    value already the right shape, or one the schema is silent about, stays as
    the model sent it.
    """
    if not isinstance(arguments, dict) or not isinstance(schema, dict):
        return arguments, []
    out, changed = dict(arguments), []
    for name, spec in (schema.get("properties") or {}).items():
        if name not in out or not isinstance(spec, dict):
            continue
        v, nv = out[name], out[name]
        kind = spec.get("type")
        if isinstance(kind, list):        # e.g. ["integer", "null"]
            kind = next((x for x in kind if x and x != "null"), None)
        if kind in ("integer", "number") and isinstance(v, str):
            try:
                f = float(v)
            except ValueError:
                continue
            if kind == "integer" and (f != int(f) or "e" in v.lower()):
                continue                  # a lossy or exotic cast is not ours to make
            nv = int(f) if kind == "integer" else f
        elif kind == "boolean" and isinstance(v, str):
            low = v.strip().lower()
            nv = True if low in ("true", "yes") else False if low in ("false", "no") else v
        elif kind in ("array", "object") and isinstance(v, str):
            try:
                nv = json.loads(v)
            except ValueError:
                nv = [s.strip() for s in v.split(",")] if (kind == "array" and "," in v) else v
            if not isinstance(nv, list if kind == "array" else dict):
                nv = v
        elif kind == "string" and isinstance(v, int) and not isinstance(v, bool):
            nv = str(v)
        if type(nv) is not type(v):
            out[name] = nv
            changed.append(f"{name}: {json.dumps(v)[:60]} -> {json.dumps(nv)[:60]}")
    return out, changed


def find_servers(instruction):
    """`(name, url, command)` for every MCP server the task mentions."""
    out, seen = [], set()
    for line in instruction.splitlines():
        s = line.strip()
        if not (s.startswith("- ") and ": " in s):
            continue
        name, _, rest = s[2:].partition(": ")
        name, rest = name.strip(), rest.strip()
        if " at " in rest:
            transport, _, url = rest.rpartition(" at ")
            url = url.strip().strip("`\"'")
            if url.startswith(("http://", "https://")) and url not in seen:
                seen.add(url)
                out.append((name or _safe(transport), url, None))
        elif " via " in rest and "`" in rest:
            cmd = rest[rest.index("`") + 1:rest.rindex("`")]
            if cmd.strip() and cmd not in seen:
                seen.add(cmd)
                out.append((name or "stdio", None, cmd))
    if not out:
        for token in instruction.replace(")", " ").replace("`", " ").split():
            url = token.rstrip(".,") if token.startswith(("http://", "https://")) else ""
            if "/mcp" in url and url not in seen:
                seen.add(url)
                out.append((url.split("//")[1].split("/")[0], url, None))
    return out


class Bridge:
    """Installs the helper once, then lists tools and forwards calls."""

    def __init__(self, world):
        self.world = world
        self._installed = None
        self.sessions = {}
        self.tools = {}      # tool name -> ((name, url, command), real MCP name)
        self.schemas = {}    # tool name -> the parameters schema we published
        self.servers = []
        self._notes = 0      # coercion lessons handed out: a lesson, not a metronome

    async def _run(self, script, timeout_s):
        r = await self.world.exec(script, timeout_s=timeout_s)
        return _last_json(r.stdout) or {"ok": False,
                                        "error": (r.stderr or r.stdout or "no output")[-2000:]}

    async def _helper(self):
        """Install the helper script; False if the container has no python3."""
        if self._installed is None:
            self._installed = False
            script = (f"command -v python3 && mkdir -p /tmp && cat > {HELPER_PATH} <<'HMCP_PY_EOF'\n{HELPER}"
                      f"\nHMCP_PY_EOF\ntest -s {HELPER_PATH} && echo installed")
            try:
                r = await self.world.exec(script, timeout_s=30)
                self._installed = r.return_code == 0 and "installed" in (r.stdout or "")
            except Exception:
                pass
        return self._installed

    async def discover(self, instruction, timeout_s=45.0):
        servers = find_servers(instruction)
        if not servers or not await self._helper():
            return [], []
        defs = []
        for server in servers:
            name, url, command = server
            script = (f"python3 {HELPER_PATH} list {shlex.quote(url)} 30 -" if url else
                      f"python3 {HELPER_PATH} stdio {shlex.quote(command)} 30")
            try:
                r = await self._run(script, timeout_s)
            except Exception as e:
                r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            if not r.get("ok"):
                continue          # unusable: the model can still reach it by hand
            if url and r.get("session"):
                self.sessions[name] = r["session"]
            self.servers.append(server)
            for t in r.get("tools") or []:
                real = str(t.get("name") or "")
                alias = _safe(real) if real else ""
                if alias == "bash" or alias in self.tools:
                    alias = _safe(f"{name}__{real}")
                if not alias or alias in self.tools:
                    continue
                self.tools[alias] = (server, real)
                fn = {"name": alias, "parameters": self._schema(t.get("inputSchema")),
                      "description": str(t.get("description") or real)[:2000]}
                self.schemas[alias] = fn["parameters"]
                defs.append({"type": "function", "function": fn})
        return defs, list(self.tools)

    def _schema(self, schema):
        schema = dict(schema) if isinstance(schema, dict) else {}
        props = dict(schema.get("properties") or {})
        props["_timeout_s"] = {"type": "number",
                               "description": "optional: seconds to allow for this call"}
        schema["properties"] = props
        schema.setdefault("type", "object")
        return schema

    async def call(self, alias, arguments, default_timeout=180.0):
        server, real = self.tools[alias]
        name, url, command = server
        timeout = default_timeout
        changed = []
        if isinstance(arguments, dict):
            raw = arguments.pop("_timeout_s", None)
            if isinstance(raw, (int, float)) and raw > 0:
                timeout = min(float(raw), 900.0)
            arguments, changed = _coerce(self.schemas.get(alias), arguments)
        args = json.dumps(arguments if isinstance(arguments, dict) else {})
        bound = max(int(timeout), 5) + 15
        sid = shlex.quote(self.sessions.get(name, "-"))
        script = (f"python3 {HELPER_PATH} {'call ' + shlex.quote(url) + ' ' + str(bound) + ' ' + sid if url else 'stdio ' + shlex.quote(command) + ' ' + str(bound)} "
                  f"{shlex.quote(real)} {shlex.quote(args)}")
        try:
            r = await self.world.exec(script, timeout_s=timeout + 45)
        except Exception as e:
            return f"[mcp {real} did not complete: {type(e).__name__}: {e}]"
        obj = _last_json(r.stdout) or {"ok": False, "error": (r.stderr or r.stdout or "no output")[-2000:]}
        if obj.get("session") and url:
            self.sessions[name] = obj["session"]
        text = self._render(real, obj)
        if changed and self._notes < 2:
            self._notes += 1
            text += (f"\n[harness] coerced to the schema's types ({'; '.join(changed[:3])}). "
                     "Send numbers, booleans, lists and objects as JSON values, not quoted "
                     "strings; never repeat a world-changing call to work around an error.")
        return text

    def _render(self, real, obj):
        if not obj.get("ok"):
            return f"[mcp {real} error: {obj.get('error')}]"
        res = obj.get("result")
        parts = []
        if isinstance(res, dict):
            parts = [str(i["text"]) for i in res.get("content") or []
                     if isinstance(i, dict) and i.get("text") is not None]
            if not parts and res.get("structuredContent") is not None:
                parts.append(json.dumps(res["structuredContent"])[:24000])
            if res.get("isError"):
                parts.append("[the server reported an error for this call]")
        if not parts:
            parts = [json.dumps(res)[:24000] if res is not None else "[empty result]"]
        return "\n".join(parts) or "[empty result]"
