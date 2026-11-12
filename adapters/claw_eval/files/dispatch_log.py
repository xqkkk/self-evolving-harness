"""Record every mock-service request as a Claw-Eval ToolDispatch event.

Claw-Eval's own runner builds ``ToolDispatch`` records because *it* dispatches
the named tools over HTTP. Our kernel gives the agent a shell instead, so the
agent calls the same endpoints with curl/httpx and there is no dispatcher to
record anything. The graders need those records — ``tool_called`` checks,
safety gates ("must not call gmail_send_message"), and ``compute_robustness``
all read ``dispatches``.

So we log on the *service* side: this middleware sits in every mock service and
appends one JSON line per request to ``$CLAW_DISPATCH_LOG``. The judge maps
(port, path) back to the task's tool name host-side via ``tool_endpoints``.

Installed by appending one call to ``mock_services/_base.add_error_injection``,
which every mock service already calls, so a single edit covers all of them.
"""

from __future__ import annotations

import json
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware

# Endpoints the graders never treat as agent tool calls.
_SKIP = ("/audit", "/reset", "/health", "/docs", "/openapi.json", "/favicon.ico")
_MAX_BODY = 200_000  # a runaway response must not blow up the log


def _truncate(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= _MAX_BODY:
        return value
    return {"_truncated": True, "preview": text[:_MAX_BODY]}


class DispatchLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if any(path.endswith(s) for s in _SKIP):
            return await call_next(request)

        raw = await request.body()
        # Starlette consumes the stream; hand it back so the route still sees it.
        async def _receive():
            return {"type": "http.request", "body": raw, "more_body": False}
        request._receive = _receive  # noqa: SLF001

        t0 = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - t0) * 1000.0

        # BaseHTTPMiddleware hands back a streaming response; drain it so the
        # body can be logged, then rebuild an equivalent response for the client.
        chunks = [chunk async for chunk in response.body_iterator]
        body = b"".join(chunks)
        from starlette.responses import Response as _Response
        out = _Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        log_path = os.environ.get("CLAW_DISPATCH_LOG")
        if log_path:
            def _decode(b):
                if not b:
                    return None
                try:
                    return json.loads(b)
                except Exception:
                    return b.decode("utf-8", "replace")[:_MAX_BODY]
            record = {
                "port": int(os.environ.get("CLAW_SERVICE_PORT", "0")),
                "service": os.environ.get("CLAW_SERVICE_NAME", ""),
                "method": request.method,
                "path": path,
                "request_body": _truncate(_decode(raw)),
                "response_status": out.status_code,
                "response_body": _truncate(_decode(body)),
                "latency_ms": round(latency_ms, 2),
                "t": time.time(),
            }
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass  # logging must never break the service
        return out


def add_dispatch_log(app) -> None:
    app.add_middleware(DispatchLogMiddleware)
