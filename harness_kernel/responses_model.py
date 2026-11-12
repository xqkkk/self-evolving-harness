"""Model client for an OpenAI Responses API endpoint.

``chat(messages, tools)`` re-sends the full history on each call. The raw output items of a reply
(reasoning items included, with their encrypted content) are stashed on the assistant message under
``_responses_items`` and replayed verbatim on the next call, so the model keeps its chain of thought
across tool calls. Every call is recorded in full (request, response, usage, timing, or the error);
see ``recorder``.
"""

import json
import time

import httpx


def _chat_tools_to_responses(tools: list[dict] | None) -> list[dict] | None:
    """Chat-style nested tool schemas -> flat Responses tool schemas."""
    if not tools:
        return None
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            f = t["function"]
            out.append({"type": "function", "name": f["name"], "description": f.get("description", ""),
                        "parameters": f.get("parameters", {})})
        else:
            out.append(t)
    return out


def _messages_to_input(messages: list[dict]) -> list[dict]:
    """Chat-format history -> Responses input item list."""
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user"):
            items.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            raw = m.get("_responses_items")
            if raw:
                items.extend(raw)
                continue
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": m.get("tool_call_id", ""),
                          "output": m.get("content") or ""})
    return items


class ResponsesModel:
    def __init__(
        self,
        recorder,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 65536,
        timeout_s: float = 600.0,
        azure_api_version: str | None = None,
    ):
        self.recorder = recorder
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.calls = 0
        self.input_tokens = 0
        self.cache_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._timeout_s = timeout_s
        self._azure_api_version = azure_api_version
        self._client = self._make_client(base_url, api_key)

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=httpx.Timeout(self._timeout_s, connect=30.0))

    def _make_client(self, base_url: str, api_key: str):
        if self._azure_api_version:
            from openai import AsyncAzureOpenAI

            return AsyncAzureOpenAI(base_url=base_url, api_key=api_key, api_version=self._azure_api_version,
                                    http_client=self._http_client(), max_retries=5)
        from openai import AsyncOpenAI

        return AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=self._http_client(), max_retries=5)

    def _request_kwargs(self, messages: list[dict], tools: list[dict] | None) -> dict:
        kwargs: dict = {
            "model": self.model,
            "input": _messages_to_input(messages),
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        resp_tools = _chat_tools_to_responses(tools)
        if resp_tools:
            kwargs["tools"] = resp_tools
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        return kwargs

    def _usage_cost(self, usage_raw: dict | None) -> float:
        """Cost reported by the endpoint, if any (OpenRouter reports ``usage.cost``)."""
        cost = (usage_raw or {}).get("cost")
        return float(cost) if isinstance(cost, (int, float)) else 0.0

    async def _create(self, kwargs: dict):
        """The request itself; host-only subclasses wrap it with retries and fault handling."""
        return await self._client.responses.create(**kwargs)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None):
        from .model import Reply, ToolCall
        from .recorder import strip_encrypted

        kwargs = self._request_kwargs(messages, tools)
        request = self.recorder.model_request(messages, tools)
        started = time.monotonic()
        try:
            resp = await self._create(kwargs)
        except BaseException as e:
            self.recorder.record("model_call", call=self.calls + 1, request=request,
                                 error=f"{type(e).__name__}: {e}", duration_s=round(time.monotonic() - started, 3))
            raise
        duration_s = round(time.monotonic() - started, 3)

        raw_items = [item.model_dump(exclude_none=True, mode="json") for item in resp.output]
        text = resp.output_text or ""
        calls: list[ToolCall] = []
        tool_calls_field = []
        bad_arguments: str | None = None
        for item in raw_items:
            if item.get("type") == "function_call":
                args_str = item.get("arguments") or "{}"
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    bad_arguments = args_str
                    break
                call_id = item.get("call_id") or item.get("id", "")
                calls.append(ToolCall(id=call_id, name=item.get("name", ""), arguments=args))
                tool_calls_field.append({"id": call_id, "type": "function",
                                         "function": {"name": item.get("name", ""), "arguments": args_str}})

        self.calls += 1
        usage = getattr(resp, "usage", None)
        usage_raw = None
        if usage is not None:
            usage_raw = usage.model_dump(exclude_none=True, mode="json")
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            details = getattr(usage, "input_tokens_details", None)
            self.cache_tokens += getattr(details, "cached_tokens", 0) or 0
            self.cost_usd += self._usage_cost(usage_raw)

        incomplete = getattr(resp, "incomplete_details", None)
        response = {
            "items": strip_encrypted(raw_items),
            "text": text,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
            "status": getattr(resp, "status", None),
            "incomplete_details": (incomplete.model_dump(exclude_none=True, mode="json")
                                   if incomplete is not None else None),
            "usage": usage_raw,
        }
        event = {"call": self.calls, "request": request, "response": response, "duration_s": duration_s}
        if bad_arguments is not None:
            event["error"] = f"unparseable tool arguments: {bad_arguments}"
        self.recorder.record("model_call", **event)
        if bad_arguments is not None:
            raise json.JSONDecodeError("unparseable tool arguments", bad_arguments, 0)

        message: dict = {"role": "assistant", "content": text, "_responses_items": raw_items}
        if tool_calls_field:
            message["tool_calls"] = tool_calls_field
        return Reply(message=message, text=text, tool_calls=calls)
