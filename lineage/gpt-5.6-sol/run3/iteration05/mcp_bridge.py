"""Small, dependency-free bridge from the model tool API to HTTP MCP servers.

The kernel deliberately exposes only a command-execution world.  MCP servers live in
that world, so requests are made with the task container's Python standard library through ``world.exec`` rather than from the harness process.  Discovery is best effort: ordinary terminal tasks never notice it.
"""

from __future__ import annotations

import base64
import json
import re
import shlex
from dataclasses import dataclass


@dataclass
class MCPTool:
    public_name: str
    remote_name: str
    server: "MCPServer"
    description: str
    schema: dict
    read_only_hint: bool | None = None


class MCPServer:
    def __init__(self, name: str, url: str, world):
        self.name = name
        self.url = url
        self.world = world
        self.session_id: str | None = None

    def _http_command(self, payload: dict, *, headers: bool = False, max_time: int = 240) -> str:
        """Build a one-shot stdlib HTTP client command for the remote container."""
        encoded = base64.b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        ).decode()
        # urllib is available anywhere the agent can reasonably run Python, unlike
        # curl/requests which vary between task images.  FastMCP closes each response
        # after delivering its SSE message, so reading the body to EOF is appropriate.
        program = (
            "import base64,sys,urllib.request;"
            "u,p,s,t,h=sys.argv[1:];"
            "d=base64.b64decode(p);"
            "x={'Content-Type':'application/json','Accept':'application/json, text/event-stream'};"
            "x.update({'mcp-session-id':s} if s else {});"
            "r=urllib.request.urlopen(urllib.request.Request(u,data=d,headers=x,method='POST'),timeout=int(t));"
            "print('__MCP_STATUS__',r.status);"
            "print('\\n'.join(str(k)+': '+str(v) for k,v in r.headers.items()) if h=='1' else '');"
            "print(r.read().decode('utf-8','replace'))"
        )
        py = 'PY=$(command -v python3 || command -v python); "$PY"'
        return " ".join(
            [
                py,
                "-c",
                shlex.quote(program),
                shlex.quote(self.url),
                shlex.quote(encoded),
                shlex.quote(self.session_id or ""),
                str(max_time),
                "1" if headers else "0",
            ]
        )

    async def _request(self, payload: dict, *, headers: bool = False, max_time: int = 240):
        outcome = await self.world.exec(
            self._http_command(payload, headers=headers, max_time=max_time),
            timeout_s=max_time + 15,
        )
        if outcome.return_code:
            raise RuntimeError(outcome.stderr or f"HTTP helper exited {outcome.return_code}")
        return outcome.stdout

    async def initialize(self) -> list[dict]:
        raw = await self._request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "general-agent-harness", "version": "1"},
                },
            },
            headers=True,
            max_time=45,
        )
        match = re.search(r"(?im)^mcp-session-id:\s*([^\r\n]+)", raw)
        if not match:
            raise RuntimeError("MCP initialize response had no session id")
        self.session_id = match.group(1).strip()

        # This notification commonly has an empty 202 response.  Failure is harmless:
        # several compatible servers accept tools/list immediately after initialize.
        try:
            await self._request(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                max_time=15,
            )
        except Exception:
            pass
        tools: list[dict] = []
        cursor = None
        # MCP permits paginated tool lists. A small safety bound avoids looping on a
        # broken server while covering realistic registries.
        for page in range(8):
            params = {"cursor": cursor} if cursor else {}
            listed = await self._request(
                {"jsonrpc": "2.0", "id": 2 + page, "method": "tools/list", "params": params},
                max_time=60,
            )
            message = parse_wire_message(listed)
            result = message.get("result") or {}
            batch = result.get("tools") or []
            if not isinstance(batch, list):
                raise RuntimeError("invalid MCP tools/list response")
            tools.extend(batch)
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call(self, name: str, arguments: dict) -> str:
        raw = await self._request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            max_time=240,
        )
        return friendly_result(parse_wire_message(raw))


def parse_wire_message(raw: str) -> dict:
    """Extract the last JSON-RPC message from JSON or an SSE response."""
    candidates = []
    for line in raw.replace("\r\n", "\n").splitlines():
        if line.startswith("data:"):
            candidates.append(line[5:].strip())
    if not candidates:
        # When headers were requested, locate a JSON body even if the server did not
        # choose SSE.  Usually the final nonempty line is sufficient.
        candidates = [line.strip() for line in raw.splitlines() if line.strip().startswith("{")]
    for text in reversed(candidates):
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise RuntimeError("could not parse MCP response: " + raw[-1000:])


def _json_for_model(value) -> str:
    """Keep small payloads readable without bloating large structured collections."""
    pretty = json.dumps(value, ensure_ascii=False, indent=2)
    if len(pretty) <= 30000:
        return pretty
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _pretty_inner(text: str) -> str:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    # FastMCP often wraps the useful payload in exactly one ``result`` field.
    if isinstance(value, dict) and set(value) == {"result"}:
        value = value["result"]
    if isinstance(value, str):
        return value
    return _json_for_model(value)


def friendly_result(message: dict) -> str:
    if message.get("error") is not None:
        return "MCP error: " + json.dumps(message["error"], ensure_ascii=False)
    result = message.get("result")
    if not isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    chunks = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            chunks.append(_pretty_inner(item.get("text", "")))
    if chunks:
        prefix = "MCP tool reported an error:\n" if result.get("isError") else ""
        return prefix + "\n".join(chunks)
    structured = result.get("structuredContent")
    if structured is not None:
        return _json_for_model(structured)
    return _json_for_model(result)


def declared_http_servers(instruction: str) -> list[tuple[str, str]]:
    """Read only the server declarations appended by the fixed kernel."""
    marker = "MCP servers reachable from inside the task container:"
    if marker not in instruction:
        return []
    tail = instruction.split(marker, 1)[1]
    found = []
    for line in tail.splitlines():
        m = re.match(r"\s*-\s*([^:]+):.*?\bat\s+(https?://\S+)\s*$", line)
        if m:
            found.append((m.group(1).strip(), m.group(2).rstrip(".,;")))
    return found


def _is_runner_control(raw: dict) -> bool:
    """Keep benchmark/runtime control-plane methods out of the task capability set.

    Some application servers colocate evaluator setup endpoints with user-facing
    methods. Exposing knobs for seeds and run/error limits violates least privilege and
    invites an agent to alter the scenario it is meant to solve. The classification is
    based on schema/description semantics rather than a benchmark-specific tool name.
    """
    schema = raw.get("inputSchema")
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    prop_words = {str(key).lower().replace("-", "_") for key in properties}
    runner_knobs = {"seed", "max_steps", "max_errors", "user_llm_args_json"}
    if len(prop_words & runner_knobs) >= 2:
        return True

    description = str(raw.get("description") or "").lower()
    control_context = any(word in description for word in (
        "benchmark", "evaluation", "termination metadata", "run parameters",
    ))
    runtime_state = any(word in description for word in (
        "seed", "max steps", "max errors", "runtime step",
    ))
    return control_context and runtime_state


async def discover(instruction: str, world) -> tuple[list[dict], dict[str, MCPTool]]:
    """Initialize declared HTTP servers and return model schemas plus dispatch map."""
    discovered: list[MCPTool] = []
    for server_name, url in declared_http_servers(instruction):
        server = MCPServer(server_name, url, world)
        try:
            remote_tools = await server.initialize()
        except Exception:
            # Bash remains available, so an unusual MCP transport can still be used
            # manually.  Discovery must never break unrelated work.
            continue
        for raw in remote_tools:
            if not isinstance(raw, dict) or _is_runner_control(raw):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                continue
            schema = raw.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            schema.setdefault("type", "object")
            annotations = raw.get("annotations")
            read_only_hint = None
            if isinstance(annotations, dict) and isinstance(annotations.get("readOnlyHint"), bool):
                read_only_hint = annotations["readOnlyHint"]
            discovered.append(
                MCPTool(
                    name,
                    name,
                    server,
                    str(raw.get("description") or ""),
                    schema,
                    read_only_hint,
                )
            )

    # Preserve the advertised remote name whenever possible.  Only collisions need a
    # server-qualified alias.
    counts: dict[str, int] = {}
    for tool in discovered:
        counts[tool.public_name] = counts.get(tool.public_name, 0) + 1
    schemas = []
    dispatch: dict[str, MCPTool] = {}
    for tool in discovered:
        if counts[tool.public_name] > 1 or tool.public_name == "bash":
            safe_server = re.sub(r"[^A-Za-z0-9_-]", "_", tool.server.name)
            tool.public_name = f"{safe_server}__{tool.remote_name}"
        dispatch[tool.public_name] = tool
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.public_name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
        )
    return schemas, dispatch
