"""Responses API client with a client-side input-token ceiling (used for GPT-5.6 Sol and GPT-5.6 Luna).

Input-size estimation: the first call is counted locally with tiktoken (o200k_base) over the
serialized Responses payload; later calls anchor on the provider's reported usage, i.e. last input
plus last output (the output approximates the reasoning items replayed next turn), plus the local
token delta since the previous call (new tool results, or a negative delta if the harness compacted
its history). When the estimate reaches ``max_input_tokens`` the request is answered locally with the
provider's own ``context_length_exceeded`` 400 (see ``common.CapTransport``).

For OpenRouter (``base_url`` = https://openrouter.ai/api/v1), ``extra_body`` carries the provider
routing, e.g. ``{"provider": {"sort": "price"}}``. Transient API errors are retried by the OpenAI SDK
itself; any other error reaches the harness like a normal API error.
"""

import json

import httpx
import tiktoken

from harness_kernel.responses_model import ResponsesModel, _chat_tools_to_responses, _messages_to_input

from .common import CapTransport, count_tokens, log_cost


class CappedResponsesModel(ResponsesModel):
    def __init__(
        self,
        recorder,
        *,
        max_input_tokens: int,
        extra_body: dict | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        if max_tokens:  # the kernel passes --ak max_tokens as max_tokens
            kwargs["max_output_tokens"] = max_tokens
        self._over_cap = False  # read by the transport; must exist before the client is built
        super().__init__(recorder, **kwargs)
        self.max_input_tokens = int(max_input_tokens)
        self._extra_body = dict(extra_body or {})
        self._enc = tiktoken.get_encoding("o200k_base")
        self._last_call: tuple[int, int] | None = None  # (input, output) tokens of the previous call
        self._local_at_last: int | None = None

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=CapTransport(httpx.AsyncHTTPTransport(), lambda: self._over_cap),
                                 timeout=httpx.Timeout(self._timeout_s, connect=30.0))

    def _request_kwargs(self, messages, tools):
        kwargs = super()._request_kwargs(messages, tools)
        if self._extra_body:
            kwargs["extra_body"] = dict(self._extra_body)
        return kwargs

    def _local_count(self, messages, tools) -> int:
        payload = json.dumps(_messages_to_input(messages), ensure_ascii=False)
        resp_tools = _chat_tools_to_responses(tools)
        if resp_tools:
            payload += json.dumps(resp_tools, ensure_ascii=False)
        return count_tokens(self._enc, payload)

    def _estimate(self, local_now: int) -> int:
        if self._last_call is None or self._local_at_last is None:
            return local_now
        last_in, last_out = self._last_call
        return last_in + last_out + (local_now - self._local_at_last)

    async def _create(self, kwargs: dict):
        resp = await self._client.responses.create(**kwargs)
        usage = getattr(resp, "usage", None)
        log_cost({"id": getattr(resp, "id", None), "model": self.model, "purpose": "solver",
                  "cost_usd": (usage.model_dump().get("cost") if usage is not None else None)})
        return resp

    async def chat(self, messages: list[dict], tools: list[dict] | None = None):
        local_now = self._local_count(messages, tools)
        self._over_cap = self._estimate(local_now) >= self.max_input_tokens
        before = (self.input_tokens, self.output_tokens)
        reply = await super().chat(messages, tools)
        self._last_call = (self.input_tokens - before[0], self.output_tokens - before[1])
        self._local_at_last = local_now
        return reply
