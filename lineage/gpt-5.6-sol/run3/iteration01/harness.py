"""General autonomous agent loop with bounded context and native MCP bridging."""

from __future__ import annotations

import asyncio

from .mcp_bridge import discover


SYSTEM_PROMPT = """You are a capable autonomous agent working inside a Linux task container. Complete the user's task, not merely describe how to do it.

Operating method:
- First identify the required deliverables, constraints, and likely acceptance checks. Inspect the environment and relevant inputs before choosing an implementation.
- Work from evidence. Check file formats, dimensions, units, interfaces, and existing project conventions rather than guessing. For repository changes, trace the call path and read nearby tests and analogous implementations before editing.
- Make focused changes, then validate them with the strongest affordable checks. Exercise edge cases implied by the request and independently inspect final artifacts. Do not modify tests or evaluation machinery.
- Budget the run. Prioritize a correct deliverable over exhaustive exploration. Bound commands that may wait on services, searches, builds, or training; redirect noisy output to files and inspect selected portions. Use the bash timeout_s argument explicitly for genuinely long jobs. Never allow one diagnostic probe to consume the whole task.
- Bash output is clipped when very large. Use targeted grep/find, sed ranges, summaries, or redirect output to a file instead of dumping entire trees, generated files, binaries, or logs.
- If application/MCP tools are present, call those tools directly. Use them to obtain current state before acting, respect confirmation and authorization requirements in the task, and verify state-changing operations. Do not build a second MCP client in bash when direct tools are available.
- When uncertainty is resolvable from local evidence or tools, resolve it. Ask the user only when a consequential ambiguity truly cannot be resolved and the environment supports interaction.
- Before finishing, re-read the request, check every deliverable path and constraint, and run a final concise verification. In your final response state what was completed and any material limitation; do not claim checks you did not run.

Never cheat: do not obtain an answer from an upstream fix, published solution, hidden tests, evaluation artifacts, or other source that already carries the task's answer. Ordinary documentation and installed library source are allowed. Solve the task yourself.
"""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command in the task container. Prefer focused output. "
            "The default timeout is 900 seconds; set timeout_s for a deliberately "
            "long build/training job or a shorter potentially blocking probe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_s": {
                    "type": "integer",
                    "description": "Command timeout in seconds (1-2400).",
                    "minimum": 1,
                    "maximum": 2400,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

MAX_RESULT_CHARS = 60000
CONTEXT_CHAR_TARGET = 450000


def clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 300) // 2
    omitted = len(text) - 2 * half
    return (
        text[:half]
        + f"\n\n[... harness omitted {omitted} characters of excessive output; "
          "rerun a targeted command or inspect a redirected file ...]\n\n"
        + text[-half:]
    )


def _message_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def compact_history(messages: list[dict], *, aggressive: bool = False) -> None:
    """Bound old observations while preserving tool-call/result protocol pairs."""
    assistant_indexes = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    keep_assistant = 3 if aggressive else 8
    for i in assistant_indexes[:-keep_assistant]:
        # Responses API reasoning items can be large. The ordinary content and
        # tool_calls fields are sufficient to reconstruct old turns.
        messages[i].pop("_responses_items", None)

    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep_tools = 3 if aggressive else 7
    old = tool_indexes[:-keep_tools]
    old_limit = 900 if aggressive else 3000
    for i in old:
        content = str(messages[i].get("content") or "")
        if len(content) > old_limit:
            half = old_limit // 2
            messages[i]["content"] = (
                content[:half]
                + "\n[... old observation compacted; rerun if needed ...]\n"
                + content[-half:]
            )

    # A long initial policy plus several medium results can still exceed the target.
    if not aggressive and _message_chars(messages) > CONTEXT_CHAR_TARGET:
        for i in tool_indexes[:-4]:
            content = str(messages[i].get("content") or "")
            if len(content) > 1200:
                messages[i]["content"] = (
                    content[:550] + "\n[... old observation compacted ...]\n" + content[-550:]
                )


def _looks_like_context_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "context_length" in text or "context window" in text or "too many tokens" in text


async def run(instruction, model, world):
    mcp_schemas, mcp_dispatch = await discover(instruction, world)
    system = SYSTEM_PROMPT
    if mcp_schemas:
        system += (
            f"\nThis run exposes {len(mcp_schemas)} application tools discovered from "
            "the declared MCP server(s). Call them directly by name; transport setup "
            "has already been completed.\n"
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    tools = [BASH_TOOL, *mcp_schemas]

    for _turn in range(120):
        compact_history(messages)
        try:
            reply = await model.chat(messages, tools=tools)
        except Exception as exc:
            if not _looks_like_context_error(exc):
                raise
            # A provider may tokenize source/log output much more densely than the
            # character heuristic predicts. Compact once substantially and retry.
            compact_history(messages, aggressive=True)
            reply = await model.chat(messages, tools=tools)

        messages.append(reply.message)
        if not reply.tool_calls:
            return

        for call in reply.tool_calls:
            if call.name == "bash":
                command = str(call.arguments.get("command", ""))
                try:
                    timeout_s = int(call.arguments.get("timeout_s", 900))
                except (TypeError, ValueError):
                    timeout_s = 900
                timeout_s = max(1, min(timeout_s, 2400))
                try:
                    result = await world.exec(command, timeout_s=timeout_s)
                    parts = [result.stdout]
                    if result.stderr:
                        parts.append("[stderr]\n" + result.stderr)
                    parts.append(f"[exit {result.return_code}]")
                    output = "\n".join(part for part in parts if part)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    output = f"bash execution failed or timed out: {type(exc).__name__}: {exc}"
            elif call.name in mcp_dispatch:
                tool = mcp_dispatch[call.name]
                try:
                    output = await tool.server.call(tool.remote_name, call.arguments)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    output = f"MCP tool transport failed: {type(exc).__name__}: {exc}"
            else:
                output = f"unknown tool: {call.name}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": clip(str(output))}
            )

    # Avoid an unbounded pathological loop while still allowing long interactive tasks.
    messages.append(
        {
            "role": "user",
            "content": "Tool-call budget is exhausted. Finalize now with the best completed result.",
        }
    )
    await model.chat(messages, tools=tools)
