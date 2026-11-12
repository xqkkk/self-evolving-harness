"""OpenAI-compatible Chat Completions client for OpenRouter (used for GLM-5.3 and GLM-5.3-Flash).

Settings used in the paper: ``reasoning_effort`` "max" with ``thinking`` on, ``max_tokens`` 65,536,
``max_input_tokens`` 258,400, and a provider policy restricted to fp8 endpoints, sorted by price.

Behaviour, in the order a call meets it:

* the input ceiling: a request whose estimated size (tiktoken o200k_base over the JSON body) reaches
  ``max_input_tokens`` fails locally with the provider's ``context_length_exceeded`` 400;
* transient HTTP errors (408, 409, 429, 5xx) and 200 replies without a ``choices`` envelope are retried
  with backoff, as an API SDK would; any other error reaches the harness like a normal API error;
* reasoning blocks returned by the provider are attached to the assistant message, because OpenRouter
  requires them to survive tool-call round trips.
"""

import asyncio
import random
import time

import httpx
import tiktoken

from harness_kernel.model import Reply, _normalize

from .common import CONTEXT_LENGTH_ERROR, count_tokens, log_cost

BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 8
_RETRY_STATUS = {408, 409, 429}
_REASONING_KEYS = ("reasoning_details", "reasoning_content", "reasoning")


def _retryable(code: int) -> bool:
    return code in _RETRY_STATUS or code >= 500


def _wire_message(m: dict) -> dict:
    """Drop client-private keys (``_responses_items`` etc.) before a message goes over the wire."""
    return {k: v for k, v in m.items() if not k.startswith("_")}


class OpenRouterChatModel:
    def __init__(
        self,
        recorder,
        *,
        model: str,
        api_key: str,
        base_url: str = BASE_URL,
        reasoning_effort: str | None = None,
        thinking: bool = False,
        max_tokens: int = 65536,
        max_input_tokens: int | None = None,
        provider: dict | None = None,
        price_sort: bool = True,
        timeout_s: float = 600.0,
        **_ignored,
    ):
        self.recorder = recorder
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_output_tokens = max_tokens
        self.max_input_tokens = int(max_input_tokens) if max_input_tokens else None
        self.calls = 0
        self.input_tokens = 0
        self.cache_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self._provider = dict(provider or {})
        if price_sort:
            self._provider["sort"] = "price"
        self._enc = tiktoken.get_encoding("o200k_base") if self.max_input_tokens else None
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30.0))

    def _body(self, messages, tools) -> dict:
        body: dict = {"model": self.model, "messages": [_wire_message(m) for m in messages],
                      "max_tokens": self.max_output_tokens, "stream": False}
        if tools:
            body["tools"] = tools
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        elif self.thinking:
            body["reasoning"] = {"enabled": True}
        if self._provider:
            body["provider"] = self._provider
        return body

    @staticmethod
    def _delay(attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(2.0 ** attempt + random.random(), 60.0)

    async def _post(self, body: dict) -> dict:
        """One request with HTTP-level retries; returns the parsed payload or raises."""
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await self._client.post(self._url, headers=self._headers, json=body)
            except httpx.TimeoutException:
                raise  # a request that timed out is not resent
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt >= MAX_RETRIES:
                    raise
                await asyncio.sleep(self._delay(attempt))
                continue
            if _retryable(resp.status_code) and attempt < MAX_RETRIES:
                await asyncio.sleep(self._delay(attempt, resp.headers.get("retry-after")))
                continue
            resp.raise_for_status()
            payload = resp.json()
            if "choices" not in payload:
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(self._delay(attempt))
                    continue
                raise RuntimeError(f"provider returned 200 without choices: {resp.text[:300]}")
            payload["_generation_id"] = resp.headers.get("x-generation-id")
            return payload
        raise last_exc or RuntimeError("retries exhausted with no response")

    def _account(self, payload: dict, purpose: str) -> None:
        usage = payload.get("usage") or {}
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            self.cost_usd += float(cost)
        log_cost({"id": payload.get("_generation_id") or payload.get("id"), "model": payload.get("model"),
                  "provider": payload.get("provider"), "purpose": purpose, "cost_usd": cost,
                  "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        body = self._body(messages, tools)
        request = self.recorder.model_request(messages, tools)
        started = time.monotonic()
        try:
            if self.max_input_tokens is not None and count_tokens(self._enc, body) >= self.max_input_tokens:
                req = httpx.Request("POST", self._url)
                resp = httpx.Response(400, request=req, json={"error": CONTEXT_LENGTH_ERROR})
                raise httpx.HTTPStatusError(f"Error code: 400 - {{'error': {CONTEXT_LENGTH_ERROR!r}}}",
                                            request=req, response=resp)
            payload = await self._post(body)
        except BaseException as e:
            self.recorder.record("model_call", call=self.calls + 1, request=request,
                                 error=f"{type(e).__name__}: {e}", duration_s=round(time.monotonic() - started, 3))
            raise
        duration_s = round(time.monotonic() - started, 3)

        texts: list[str] = []
        raw_calls: list[dict] = []
        finish = None
        for ch in payload.get("choices") or []:
            msg = ch.get("message") or {}
            if msg.get("content"):
                texts.append(msg["content"])
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw_calls.append({"id": tc.get("id", ""),
                                  "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")}})
            finish = ch.get("finish_reason") or finish
        message, text, calls = _normalize("\n".join(texts), raw_calls)

        self.calls += 1
        usage = payload.get("usage") or {}
        self.input_tokens += usage.get("prompt_tokens", 0) or 0
        self.output_tokens += usage.get("completion_tokens", 0) or 0
        details = usage.get("prompt_tokens_details") or {}
        self.cache_tokens += (details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
        self._account(payload, purpose="solver")

        self.recorder.record(
            "model_call",
            call=self.calls,
            request=request,
            response={
                "items": [ch.get("message") or {} for ch in payload.get("choices") or []],
                "text": text,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                "status": finish,
                "provider": payload.get("provider"),
                "incomplete_details": None,
                "usage": usage or None,
            },
            duration_s=duration_s,
        )
        first = ((payload.get("choices") or [{}])[0] or {}).get("message") or {}
        for key in _REASONING_KEYS:
            if first.get(key) is not None:
                message[key] = first[key]
        return Reply(message=message, text=text, tool_calls=calls)
