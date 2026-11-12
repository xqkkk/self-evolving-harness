"""Small MCP-to-model bridge.

MCP endpoints are reachable from the task container, while this harness runs in
the agent container.  The bridge therefore performs protocol requests through
``world.exec`` and turns advertised MCP tools into ordinary model tools.

A call may ask its MCP client to perform model sampling.  Calls run as detached
workers in the task container so the harness can service those reverse JSON-RPC
requests without closing the original HTTP response stream.
"""

import asyncio
import base64
import itertools
import json
import re
import shlex
import time
import uuid

_REMOTE = r'''import json, os, sys, time, urllib.request
url, mode = sys.argv[1], sys.argv[2]
sid = None if sys.argv[3] == "-" else sys.argv[3]
payload = json.loads(sys.argv[4])
rpc_id = int(sys.argv[5])
job = sys.argv[6] if len(sys.argv) > 6 else None
request_no = 0

def atomic_write(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f)
    os.replace(tmp, path)

def reverse_request(value):
    global request_no
    method = value.get("method", "")
    ident = value.get("id")
    if ident is None:
        return
    if method == "roots/list":
        answer = {"jsonrpc":"2.0", "id":ident, "result":{"roots":[]}}
    elif job:
        request_no += 1
        token = str(request_no)
        atomic_write(os.path.join(job, "request_" + token + ".json"), value)
        response_path = os.path.join(job, "response_" + token + ".json")
        deadline = time.monotonic() + 600
        while not os.path.exists(response_path):
            if time.monotonic() >= deadline:
                answer = {"jsonrpc":"2.0", "id":ident,
                          "error":{"code":-32000,"message":"client response timed out"}}
                break
            time.sleep(.1)
        else:
            with open(response_path) as f:
                answer = json.load(f)
            try: os.unlink(response_path)
            except OSError: pass
            try: os.unlink(os.path.join(job, "request_" + token + ".json"))
            except OSError: pass
    else:
        answer = {"jsonrpc":"2.0", "id":ident,
                  "error":{"code":-32601,"message":"client capability unavailable"}}
    post(answer, False)

def post(obj, want=True):
    global sid
    data = json.dumps(obj, separators=(",", ":")).encode()
    h = {"Content-Type":"application/json", "Accept":"application/json, text/event-stream"}
    if sid: h["Mcp-Session-Id"] = sid
    if mode != "init": h["MCP-Protocol-Version"] = "2025-03-26"
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
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
            if value.get("id") == obj.get("id") and ("result" in value or "error" in value):
                return value
            if value.get("method") and "id" in value:
                reverse_request(value)
        raise RuntimeError("MCP stream ended without a matching response")

def main():
    if mode == "init":
        ans = post({"jsonrpc":"2.0", "id":1, "method":"initialize", "params":{
            "protocolVersion":"2025-03-26",
            "capabilities":{"sampling":{}, "roots":{"listChanged":False}},
            "clientInfo":{"name":"agent-harness", "version":"1"}}})
        post({"jsonrpc":"2.0", "method":"notifications/initialized", "params":{}}, False)
        tools = post({"jsonrpc":"2.0", "id":2, "method":"tools/list", "params":{}})
        return {"ok":True, "session":sid, "initialize":ans, "tools":tools}
    ans = post({"jsonrpc":"2.0", "id":rpc_id, "method":"tools/call", "params":payload})
    return {"ok":True, "response":ans}

try:
    result = main()
except Exception as e:
    result = {"ok":False, "error":type(e).__name__+": "+str(e)}
if job:
    atomic_write(os.path.join(job, "result.json"), result)
else:
    print(json.dumps(result))
'''

_DECL = re.compile(
    r"^\s*-\s*([^:\n]+):\s*streamable-http\s+at\s+(https?://\S+)\s*$",
    re.MULTILINE,
)


class MCPBridge:
    def __init__(self, world, model=None):
        self.world = world
        self.model = model
        self.servers = {}
        self.routes = {}
        self._path = "/tmp/agent_harness_mcp.py"
        self._ids = itertools.count(100)
        self._tag = uuid.uuid4().hex[:10]

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
            cmd = self._command(url, "init", "-", json.dumps({}), 0)
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

    def _command(self, url, mode, sid, payload, rpc_id, job=None):
        args = ["python3", self._path, url, mode, sid or "-", payload, rpc_id]
        if job:
            args.append(job)
        return " ".join(shlex.quote(str(x)) for x in args)

    @staticmethod
    def _last_json(text):
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
        raise ValueError("no JSON response from MCP transport: " + text[-500:])

    async def _sample(self, request):
        """Fulfil an MCP sampling request using the harness model."""
        ident = request.get("id")
        params = request.get("params") or {}
        method = request.get("method")
        if method != "sampling/createMessage" or self.model is None:
            return {"jsonrpc":"2.0", "id":ident, "error":{
                "code":-32601, "message":f"Unsupported client request: {method}"}}

        messages = []
        if params.get("systemPrompt"):
            messages.append({"role":"system", "content":str(params["systemPrompt"])})
        for item in params.get("messages") or []:
            content = item.get("content", "")
            if isinstance(content, dict):
                if content.get("type") == "text":
                    content = content.get("text", "")
                else:
                    content = json.dumps(content, ensure_ascii=False)
            elif isinstance(content, list):
                content = "\n".join(
                    str(x.get("text", "")) if isinstance(x, dict) else str(x)
                    for x in content
                )
            role = item.get("role")
            messages.append({"role": role if role in ("user", "assistant") else "user",
                             "content": str(content)})
        if not messages:
            messages = [{"role":"user", "content":"Provide the requested sample."}]
        reply = await self.model.chat(messages, tools=None)
        text = reply.text or reply.message.get("content") or ""
        return {"jsonrpc":"2.0", "id":ident, "result":{
            "model":str(getattr(self.model, "model", "harness-model")),
            "stopReason":"endTurn", "role":"assistant",
            "content":{"type":"text", "text":text},
        }}

    async def _worker_call(self, url, sid, payload, rpc_id):
        job = f"/tmp/agent_mcp_{self._tag}_{rpc_id}"
        command = self._command(url, "call", sid, payload, rpc_id, job)
        launch = (
            f"rm -rf {shlex.quote(job)}; mkdir -p {shlex.quote(job)}; "
            f"({command} >/dev/null 2>{shlex.quote(job + '/stderr')}; "
            f"echo $? > {shlex.quote(job + '/exit')}) "
            f"</dev/null >/dev/null 2>&1 &"
        )
        out = await self.world.exec(launch, timeout_s=30)
        if out.return_code:
            raise RuntimeError((out.stderr or "could not launch MCP worker").strip())

        deadline = time.monotonic() + 600
        handled = set()
        while time.monotonic() < deadline:
            # Wait in the task container to avoid a recorded exec every 100 ms.
            poll = (
                f"for i in $(seq 1 40); do "
                f"[ -f {shlex.quote(job + '/result.json')} ] && "
                f"{{ echo RESULT; cat {shlex.quote(job + '/result.json')}; exit; }}; "
                f"[ -f {shlex.quote(job + '/exit')} ] && "
                f"{{ echo FAILED; cat {shlex.quote(job + '/stderr')}; exit; }}; "
                f"f=$(find {shlex.quote(job)} -maxdepth 1 -name 'request_*.json' "
                f"-type f | sort | head -1); "
                f"[ -n \"$f\" ] && {{ echo REQUEST:$f; cat \"$f\"; exit; }}; "
                f"sleep .5; done; echo WAIT"
            )
            state = await self.world.exec(poll, timeout_s=30)
            lines = state.stdout.splitlines()
            if not lines:
                continue
            if lines[0] == "RESULT":
                return json.loads("\n".join(lines[1:]))
            if lines[0] == "FAILED":
                raise RuntimeError("MCP worker exited: " + "\n".join(lines[1:])[-1000:])
            if lines[0].startswith("REQUEST:"):
                path = lines[0][8:]
                if path in handled:
                    await asyncio.sleep(.2)
                    continue
                handled.add(path)
                request = json.loads("\n".join(lines[1:]))
                try:
                    response = await asyncio.wait_for(self._sample(request), timeout=540)
                except Exception as exc:
                    response = {"jsonrpc":"2.0", "id":request.get("id"), "error":{
                        "code":-32000, "message":f"client sampling failed: {exc}"}}
                token = re.search(r"request_(\d+)\.json$", path)
                if not token:
                    raise RuntimeError("invalid MCP reverse-request path")
                response_path = f"{job}/response_{token.group(1)}.json"
                encoded = base64.b64encode(json.dumps(response).encode()).decode()
                temporary = response_path + ".tmp"
                write = (f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(temporary)}; "
                         f"mv {shlex.quote(temporary)} {shlex.quote(response_path)}")
                wr = await self.world.exec(write, timeout_s=30)
                if wr.return_code:
                    raise RuntimeError("could not answer MCP reverse request: " + wr.stderr)
        raise TimeoutError("MCP call did not finish within 600 seconds")

    async def call(self, public_name, arguments):
        server_name, remote_name = self.routes[public_name]
        url, sid = self.servers[server_name]
        payload = json.dumps(
            {"name":remote_name, "arguments":arguments or {}},
            separators=(",", ":"), ensure_ascii=False,
        )
        try:
            data = await self._worker_call(url, sid, payload, next(self._ids))
            if not data.get("ok"):
                return "MCP transport error: " + str(data.get("error"))
            response = data.get("response") or {}
            if response.get("error"):
                return "MCP error: " + json.dumps(response["error"], ensure_ascii=False)
            result = response.get("result")
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
