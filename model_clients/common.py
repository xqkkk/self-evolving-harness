"""Shared pieces of the host-only clients: the synthetic overflow error, cost receipts, token counting."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

# The provider's own response to an over-long request. Sending it back for an over-ceiling request
# makes the ceiling indistinguishable from a real context window.
CONTEXT_LENGTH_ERROR = {
    "message": ("code: context_length_exceeded; message: Your input exceeds the context window of this model. "
                "Please adjust your input and try again."),
    "type": "invalid_request_error",
    "param": "input",
    "code": "context_length_exceeded",
}


class CapTransport(httpx.AsyncBaseTransport):
    """Answer an over-ceiling request with the provider's own 400, without sending it.

    The SDK then raises the error itself, so the exception and traceback are the same as for a real
    overflow. Nothing leaves the machine, and a 400 is not on the SDK's retry list.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, over_cap):
        self._inner = inner
        self._over_cap = over_cap

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._over_cap():
            return httpx.Response(400, headers={"content-type": "application/json; charset=utf-8"},
                                  json={"error": CONTEXT_LENGTH_ERROR}, request=request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def log_cost(entry: dict) -> None:
    """Append a cost receipt to ``$HARNESS_COST_LOG`` when that variable is set. Never logs prompts."""
    path = os.environ.get("HARNESS_COST_LOG")
    if not path:
        return
    entry = {"recorded_at": datetime.now(timezone.utc).isoformat(), **entry}
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def count_tokens(encoder, payload) -> int:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return len(encoder.encode(text, disallowed_special=()))
