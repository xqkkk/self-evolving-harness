"""Small MCP-to-model bridge.

MCP endpoints are reachable from the task container, while this harness runs in
the agent container.  The bridge therefore performs protocol requests through
``world.exec`` and turns advertised MCP tools into ordinary model tools.
"""

import base64
import itertools
import json
import re
import shlex

# This is installed in /tmp in the task container.  Reading only through the
# matching JSON-RPC response is important: streamable HTTP responses may leave
# their SSE connection open indefinitely after delivering the result.
_REMOTE = r'''import json, sys, urllib.request, urllib.error
url, mode = sys.argv[1], sys.argv[2]
sid = None if sys.argv[3] == "-" else sys.argv[3]
payload = json.loads(sys.argv[4])
rpc_id = int(sys.argv[5])

def post(obj, want=True):
    global sid
    data = json.dumps(obj, separators=(",", ":")).encode()
    h = {"Content-Type":"application/json", "Accept":"application/json, text/event-stream"}
    if sid: h["Mcp-Session-Id"] = sid
    if mode != "init": h["MCP-Protocol-Version"] = "2025-03-26"
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    # This is an idle socket timeout, not a wall-clock limit. SSE keepalives can
    # legitimately arrive while a remote operation performs slow model work.
    with urllib.request.urlopen(req, timeout=900) as r:
        sid = r.headers.get("Mcp-Session-Id", sid)
        if not want:
            return None
        ctype = r.headers.get("Content-Type", "")
        if "text/event-stream" not in ctype:
            raw = r.read()
            return json.loads(raw) if raw else None
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try: value = json.loads(line[5:].strip())
            except Exception: continue
            if value.get("id") == obj.get("id"):
                return value
        raise RuntimeError("MCP stream ended without a matching response")

try:
    if mode == "init":
        ans = post({"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{
            "protocolVersion":"2025-03-26", "capabilities":{},
            "clientInfo":{"name":"agent-harness", "version":"1"}}})
        post({"jsonrpc":"2.0", "method":"notifications/initialized", "params":{}}, False)
        tools = post({"jsonrpc":"2.0", "id":2, "method":"tools/list", "params":{}})
        print(json.dumps({"ok":True, "session":sid, "initialize":ans, "tools":tools}))
    else:
        ans = post({"jsonrpc":"2.0", "id":rpc_id, "method":"tools/call", "params":payload})
        print(json.dumps({"ok":True, "response":ans}))
except Exception as e:
    print(json.dumps({"ok":False, "error":type(e).__name__+": "+str(e)}))
'''

_DECL = re.compile(
    r"^\s*-\s*([^:\n]+):\s*streamable-http\s+at\s+(https?://\S+)\s*$",
    re.MULTILINE,
)


class MCPBridge:
    def __init__(self, world):
        self.world = world
        self.servers = {}
        self.routes = {}
        self._path = "/tmp/agent_harness_mcp.py"
        self._ids = itertools.count(100)

    async def discover(self, instruction):
        """Initialize declared HTTP servers and return OpenAI-format tools."""
        declarations = _DECL.findall(instruction)
        if not declarations:
            return [], []
        encoded = base64.b64encode(_REMOTE.encode()).decode()
        install = f"echo {shlex.quote(encoded)} | base64 -d > {self._path}"
        result = await self.world.exec(install, timeout_s=30)
        if result.return_code:
            return [], ["Could not install the MCP transport helper: " + result.stderr]

        found = []
        errors = []
        for server_name, url in declarations:
            payload = json.dumps({})
            cmd = self._command(url, "init", "-", payload, 0)
            try:
                out = await self.world.exec(cmd, timeout_s=330)
                data = self._last_json(out.stdout)
                if not data.get("ok"):
                    raise RuntimeError(data.get("error", "unknown initialization error"))
                response = data.get("tools") or {}
                tool_items = ((response.get("result") or {}).get("tools") or [])
                session = data.get("session")
                self.servers[server_name] = (url, session)
                for item in tool_items:
                    if item.get("name"):
                        found.append((server_name, item))
            except Exception as exc:
                errors.append(f"MCP server {server_name!r} unavailable: {exc}")

        counts = {}
        for _, item in found:
            counts[item["name"]] = counts.get(item["name"], 0) + 1
        tools = []
        for server_name, item in found:
            remote = item["name"]
            public = remote if counts[remote] == 1 else f"{server_name}__{remote}"
            # Keep names legal for common model APIs.
            public = re.sub(r"[^A-Za-z0-9_-]", "_", public)[:64]
            base = public
            suffix = 2
            while public in self.routes:
                tag = f"_{suffix}"
                public = base[:64-len(tag)] + tag
                suffix += 1
            self.routes[public] = (server_name, remote)
            schema = item.get("inputSchema") or {"type": "object", "properties": {}}
            tools.append({"type":"function", "function":{
                "name":public,
                "description":item.get("description") or f"MCP tool {remote}",
                "parameters":schema,
            }})
        return tools, errors

    def _command(self, url, mode, sid, payload, rpc_id):
        args = ["python3", self._path, url, mode, sid or "-", payload, rpc_id]
        return " ".join(shlex.quote(str(x)) for x in args)

    @staticmethod
    def _last_json(text):
        # Shell startup noise is unusual but should not make the transport fail.
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
        raise ValueError("no JSON response from MCP transport: " + text[-500:])

    async def call(self, public_name, arguments):
        server_name, remote_name = self.routes[public_name]
        url, sid = self.servers[server_name]
        payload = json.dumps(
            {"name":remote_name, "arguments":arguments or {}},
            separators=(",", ":"), ensure_ascii=False,
        )
        try:
            # A tool may itself wait on a human or model.  The former 330s cap
            # repeatedly cancelled valid long-lived calls; allow one bounded
            # wait instead of encouraging ambiguous retries of side effects.
            out = await self.world.exec(
                self._command(url, "call", sid, payload, next(self._ids)),
                timeout_s=930,
            )
            if out.return_code:
                detail = (out.stderr or out.stdout or "no diagnostic output").strip()
                return f"MCP transport process exited {out.return_code}: {detail[-1000:]}"
            data = self._last_json(out.stdout)
            if not data.get("ok"):
                return "MCP transport error: " + str(data.get("error"))
            response = data.get("response") or {}
            if response.get("error"):
                return "MCP error: " + json.dumps(response["error"], ensure_ascii=False)
            result = response.get("result")
            # MCP SDKs normally present content blocks to the agent.  Returning
            # both content and structuredContent often duplicates a very large
            # payload verbatim, so prefer readable content while preserving
            # errors and non-text blocks.
            if isinstance(result, dict) and isinstance(result.get("content"), list):
                rendered = []
                for block in result["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        rendered.append(str(block.get("text", "")))
                    else:
                        rendered.append(json.dumps(block, ensure_ascii=False))
                text = "\n".join(rendered)
                if result.get("isError"):
                    text = "MCP tool reported an error:\n" + text
                return text
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"MCP transport error: {type(exc).__name__}: {exc}"
