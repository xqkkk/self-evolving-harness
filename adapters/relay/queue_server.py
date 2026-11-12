"""In-sandbox relay: a local endpoint whose requests are answered by the host (``adapters.relay.host``).

Task containers have no network (or only what their task needs) and never hold API keys. Services inside them that
need a model (the tau3 user simulator and NL-assertion judge, the Claw-Eval user simulator) or a web proxy (Claw-Eval
``web_real``) talk to this server instead. It parks each request; the host daemon reaches into the sandbox, takes the
pending requests, performs them with the host's credentials, and posts the replies back, which unblocks the callers.

Modes (stdlib only, runs on any python:3 image):

    python3 queue_server.py --port 8088                # LLM: POST /v1/chat/completions, POST /v1/responses
    python3 queue_server.py --port 8089 --http-get     # also: GET /?<params> is parked as a web request
                                                       # (its api_key parameter is dropped)

Relay protocol (used by the host daemon):

    GET  /relay/pending        unclaimed requests [{id, kind, path, body | params}]; claims them
                               (re-offered if no reply within CLAIM_TIMEOUT_S)
    POST /relay/reply/<id>     llm:      {"status": int, "body": <json>}
                               http_get: {"status": int, "content_type": str, "body_b64": str}
    GET  /health

Streaming chat requests are forwarded non-streaming and the reply is re-encoded as a server-sent event stream.
"""

import argparse
import base64
import json
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WAIT_TIMEOUT_S = 900.0  # a blocked caller gives up (SDKs retry on 504)
CLAIM_TIMEOUT_S = 120.0  # claimed but unanswered: offered again
LLM_PATHS = ("/v1/chat/completions", "/v1/responses")
HTTP_GET = False

_lock = threading.Condition()
_requests: dict[str, dict] = {}  # id -> {kind, path, body|params, claimed_at, reply}


def _park(entry: dict, timeout: float):
    rid = uuid.uuid4().hex
    with _lock:
        _requests[rid] = {**entry, "claimed_at": None, "reply": None}
        _lock.notify_all()
        deadline = time.monotonic() + timeout
        while _requests[rid]["reply"] is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _lock.wait(timeout=min(remaining, 5.0))
        return _requests.pop(rid)["reply"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, payload, content_type: str = "application/json") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _send_sse_chat(self, response: dict) -> None:
        """Replay a completed chat response as one SSE chunk stream (clients accumulate deltas)."""
        head = {"id": response.get("id", "relay"), "object": "chat.completion.chunk",
                "created": response.get("created", int(time.time())), "model": response.get("model", "")}
        chunks = []
        for choice in response.get("choices") or [{}]:
            idx = choice.get("index", 0)
            message = choice.get("message") or {}
            chunks.append({**head, "choices": [{"index": idx, "delta": {"role": "assistant"}, "finish_reason": None}]})
            if message.get("content") is not None:
                chunks.append({**head, "choices": [{"index": idx, "delta": {"content": message["content"]},
                                                    "finish_reason": None}]})
            for i, call in enumerate(message.get("tool_calls") or []):
                delta = {"index": i, "id": call.get("id"), "type": call.get("type", "function"),
                         "function": call.get("function", {})}
                chunks.append({**head, "choices": [{"index": idx, "delta": {"tool_calls": [delta]},
                                                    "finish_reason": None}]})
            chunks.append({**head, "choices": [{"index": idx, "delta": {},
                                                "finish_reason": choice.get("finish_reason", "stop")}]})
        if response.get("usage"):
            chunks.append({**head, "choices": [], "usage": response["usage"]})
        data = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        self._send(200, data, "text/event-stream")

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path == "/health":
            self._send(200, {"ok": True})
        elif path == "/relay/pending":
            now, out = time.monotonic(), []
            with _lock:
                for rid, req in _requests.items():
                    if req["reply"] is None and (req["claimed_at"] is None or now - req["claimed_at"] > CLAIM_TIMEOUT_S):
                        req["claimed_at"] = now
                        out.append({"id": rid, **{k: req[k] for k in ("kind", "path", "body", "params") if k in req}})
            self._send(200, out)
        elif HTTP_GET and path == "/":  # ScraperAPI-style calls: GET /?url=...&...
            params = [(k, v) for k, v in urllib.parse.parse_qsl(query, keep_blank_values=True) if k != "api_key"]
            reply = _park({"kind": "http_get", "path": path, "params": params}, timeout=150.0)
            if reply is None:
                self._send(504, {"error": "host relay timeout"})
            else:
                self._send(int(reply.get("status", 502)), base64.b64decode(reply.get("body_b64") or b""),
                           reply.get("content_type") or "text/html")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path in LLM_PATHS:
            try:
                body = json.loads(self._body())
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "invalid json"})
                return
            want_stream = bool(body.pop("stream", False))
            body.pop("stream_options", None)
            reply = _park({"kind": "llm", "path": self.path, "body": body}, timeout=WAIT_TIMEOUT_S)
            if reply is None:
                self._send(504, {"error": {"message": "relay timeout", "type": "relay_timeout"}})
            elif want_stream and reply["status"] == 200 and self.path == "/v1/chat/completions":
                self._send_sse_chat(reply["body"])
            else:
                self._send(int(reply["status"]), reply["body"])
        elif self.path.startswith("/relay/reply/"):
            rid = self.path.rsplit("/", 1)[-1]
            try:
                payload = json.loads(self._body())
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "invalid json"})
                return
            with _lock:
                req = _requests.get(rid)
                if req is None or req["reply"] is not None:
                    self._send(410, {"error": "unknown or completed id"})
                    return
                req["reply"] = payload
                _lock.notify_all()
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})


def main() -> None:
    global HTTP_GET
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--http-get", action="store_true", help="park GET /?<params> as a web request")
    args = ap.parse_args()
    HTTP_GET = args.http_get
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
