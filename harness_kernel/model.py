"""Contract types (``Reply``, ``ToolCall``) plus the smoke-check ``FakeModel``.

Messages use the OpenAI chat format: ``reply.message`` is appended to the history unchanged, and
tool results follow as ``{"role": "tool", "tool_call_id": ..., "content": ...}``. A client may stash
provider-specific items on the assistant message (``_responses_items`` for the Responses API,
``reasoning_details`` for OpenRouter chat) so that they are replayed on the next call.
"""

import asyncio
import json
from dataclasses import dataclass

MAX_FAKE_CALLS = 200


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    message: dict
    text: str
    tool_calls: list[ToolCall]


def _normalize(content: str | None, raw_tool_calls) -> tuple[dict, str, list[ToolCall]]:
    text = content or ""
    message: dict = {"role": "assistant", "content": text}
    calls: list[ToolCall] = []
    if raw_tool_calls:
        message["tool_calls"] = []
        for tc in raw_tool_calls:
            args_str = tc["function"]["arguments"]
            message["tool_calls"].append(
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": args_str}}
            )
            args = json.loads(args_str) if args_str else {}
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
    return message, text, calls


def _tool_name_and_schema(tool: dict) -> tuple[str, dict]:
    """Accept both chat-style ``{"function": {...}}`` and flat tool schemas."""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name") or ""), fn.get("parameters") or {}


def _harmless_arguments(schema: dict) -> dict:
    """Fill a tool's required parameters with the most inert values possible.

    A string that looks like a shell command becomes ``true`` (a no-op that exits 0); any other
    string is the word ``smoke``. The point is not to exercise the tool but to hand the harness a
    well-formed call and see that it copes.
    """
    props = schema.get("properties") or {}
    args = {}
    for name in schema.get("required") or []:
        spec = props.get(name) or {}
        kind = spec.get("type")
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), None)
        if kind == "string":
            args[name] = "true" if any(w in name.lower() for w in ("cmd", "command", "shell", "script")) else "smoke"
        elif kind in ("integer", "number"):
            args[name] = 1
        elif kind == "boolean":
            args[name] = False
        elif kind == "array":
            args[name] = []
        elif kind == "object":
            args[name] = {}
        else:
            args[name] = "smoke"
    return args


class FakeModel:
    """Model stand-in for the offline smoke check.

    First call: if the harness declares any tools, call the first one with harmless arguments (so the
    tool-handling path runs); otherwise plain text. Every later call: ``"Done."`` with no tool calls.
    Exposes the same public surface as the real clients.
    """

    def __init__(self, recorder):
        self.recorder = recorder
        self.model = "fake"
        self.reasoning_effort = None
        self.max_output_tokens = 65536
        self.calls = 0
        self.input_tokens = 0
        self.cache_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    async def chat(self, messages: list[dict], tools: list[dict] | None = None):
        # Yield to the event loop like a real call does, so the smoke timeout can fire.
        await asyncio.sleep(0)
        self.calls += 1
        if self.calls > MAX_FAKE_CALLS:
            raise RuntimeError(
                f"model called {self.calls} times and the harness still has not "
                "finished, though every reply after the first said it was done"
            )
        raw_tool_calls = []
        text = "Done."
        if self.calls == 1 and tools:
            name, schema = _tool_name_and_schema(tools[0])
            if name:
                text = f"Trying the {name} tool."
                raw_tool_calls = [{
                    "id": f"fake_{self.calls}_0",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(_harmless_arguments(schema))},
                }]
        message, text, calls = _normalize(text, raw_tool_calls)
        self.recorder.record(
            "model_call",
            call=self.calls,
            request=self.recorder.model_request(messages, tools),
            response={
                "items": [],
                "text": text,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                "status": "fake",
                "incomplete_details": None,
                "usage": None,
            },
            duration_s=0.0,
        )
        return Reply(message=message, text=text, tool_calls=calls)
