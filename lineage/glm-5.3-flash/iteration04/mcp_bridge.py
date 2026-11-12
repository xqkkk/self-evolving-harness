"""Minimal MCP client bridge (stdlib only). Runs INSIDE the task container; the
agent harness writes it there and invokes it so the task's MCP servers can be
exposed to the model as ordinary tools. Supports the streamable-HTTP transport
and, as a fallback, stdio.

Output discipline: exactly one line on stdout, a JSON object, ASCII-escaped so
no encoding can mangle it in transit: {"ok": true, ...} or
{"ok": false, "error": "..."}. Exit status is 0 even on tool errors (the error
is data for the caller); 2 only for malformed CLI usage. Long strings are
truncated so one result cannot flood the agent's context.
"""

import argparse
import json
import select
import subprocess
import sys
import time
import urllib.request

MAX_STR = 60000
LIST_RETRIES = 3
CALL_RETRIES = 2


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def clamp(value):
    """Recursively cap string length so a huge blob cannot flood the context."""
    if isinstance(value, str) and len(value) > MAX_STR:
        return value[:MAX_STR] + "...[truncated %d chars]" % (len(value) - MAX_STR)
    if isinstance(value, list):
        return [clamp(v) for v in value]
    if isinstance(value, dict):
        return {k: clamp(v) for k, v in value.items()}
    return value


class HttpConn:
    """JSON-RPC over MCP streamable HTTP (POST only; sessions via header)."""

    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.session = None
        self.next_id = 0

    def rpc(self, method, params=None, notify=False, retries=LIST_RETRIES):
        self.next_id += 1
        payload = {"jsonrpc": "2.0", "method": method}
        if not notify:
            payload["id"] = self.next_id
        if params is not None:
            payload["params"] = params
        last = "no response"
        for attempt in range(max(1, retries)):
            try:
                req = urllib.request.Request(
                    self.url, data=json.dumps(payload).encode("utf-8"), method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Accept", "application/json, text/event-stream")
                if self.session:
                    req.add_header("mcp-session-id", self.session)
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                self.session = resp.headers.get("mcp-session-id") or self.session
                raw = resp.read().decode("utf-8", "replace")
                ctype = resp.headers.get("Content-Type") or ""
                if "text/event-stream" in ctype:  # last parseable data: line
                    for sse_line in raw.splitlines():
                        sse_line = sse_line.strip()
                        if sse_line.startswith("data:"):
                            try:
                                data = json.loads(sse_line[5:].strip())
                            except Exception:
                                pass
                else:
                    data = json.loads(raw) if raw.strip() else None
                if data is not None:
                    if isinstance(data, dict) and data.get("error"):
                        return {"ok": False, "error": data["error"]}
                    return {"ok": True, "result": data.get("result")}
                if notify:
                    return {"ok": True, "result": None}
                last = "empty response (http status %s)" % getattr(resp, "status", "?")
            except Exception as exc:
                last = "%s: %s" % (type(exc).__name__, exc)
            if attempt + 1 < max(1, retries):
                time.sleep(1.5 * (attempt + 1))
        return {"ok": False, "error": last}


class StdioConn:
    """JSON-RPC over a child process's stdin/stdout (newline-delimited)."""

    def __init__(self, command, timeout):
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.timeout = timeout
        self.next_id = 0

    def _read(self, want_id, deadline):
        while time.time() < deadline:
            ready, _, _ = select.select([self.proc.stdout], [], [],
                                        max(0.05, deadline - time.time()))
            if not ready:
                if self.proc.poll() is not None:
                    return None
                continue
            line = self.proc.stdout.readline()
            if not line:
                return None
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue
            if isinstance(msg, dict) and msg.get("id") == want_id:
                return msg
        return None

    def rpc(self, method, params=None, notify=False, retries=1):
        if self.proc.poll() is not None:
            return {"ok": False, "error": "server process already exited"}
        self.next_id += 1
        payload = {"jsonrpc": "2.0", "method": method}
        if not notify:
            payload["id"] = self.next_id
        if params is not None:
            payload["params"] = params
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except Exception as exc:
            return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        if notify:
            return {"ok": True, "result": None}
        msg = self._read(self.next_id, time.time() + self.timeout)
        if msg is None:
            return {"ok": False, "error": "no response before timeout"}
        if msg.get("error"):
            return {"ok": False, "error": msg["error"]}
        return {"ok": True, "result": msg.get("result")}


def handshake(conn):
    r = conn.rpc("initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "harness-mcp-bridge", "version": "1.0"}})
    if not r.get("ok"):
        return r
    conn.rpc("notifications/initialized", {}, notify=True)
    return r


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, default_timeout in (("list", 120.0), ("call", 600.0)):
        sp = sub.add_parser(name)
        group = sp.add_mutually_exclusive_group(required=True)
        group.add_argument("--url")
        group.add_argument("--stdio", nargs="+", metavar="ARG")
        sp.add_argument("--timeout", type=float, default=default_timeout)
        if name == "call":
            sp.add_argument("--name", required=True)
            sp.add_argument("--args", default="{}")

    a = parser.parse_args()
    try:
        args = json.loads(getattr(a, "args", None) or "{}")
    except Exception as exc:
        emit({"ok": False, "error": "invalid --args JSON: %s" % exc})
        return 0

    conn = StdioConn(a.stdio, a.timeout) if a.stdio else HttpConn(a.url, a.timeout)
    r = handshake(conn)
    if not r.get("ok"):
        emit({"ok": False, "error": "initialize failed: %s" % r.get("error")})
        return 0

    if a.cmd == "list":
        tools, cursor = [], None
        for _ in range(30):
            r = conn.rpc("tools/list", {} if not cursor else {"cursor": cursor})
            if not r.get("ok"):
                emit({"ok": False, "error": "tools/list failed: %s" % r.get("error")})
                return 0
            result = r.get("result") or {}
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        emit({"ok": True, "tools": [
            {"name": t.get("name"), "description": (t.get("description") or "")[:4000],
             "inputSchema": t.get("inputSchema") or {"type": "object", "properties": {}}}
            for t in tools if t.get("name")]})
        return 0

    r = conn.rpc("tools/call", {"name": a.name, "arguments": args}, retries=CALL_RETRIES)
    if not r.get("ok"):
        emit({"ok": False, "error": r.get("error")})
        return 0
    result = r.get("result") or {}
    emit({"ok": True, "isError": bool(result.get("isError")),
          "content": clamp(result.get("content") or []),
          "structuredContent": clamp(result.get("structuredContent"))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
