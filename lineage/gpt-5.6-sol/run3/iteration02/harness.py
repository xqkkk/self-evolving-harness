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
- If application/MCP tools are present, call those tools directly. Use them to obtain current state before acting, respect confirmation and authorization requirements in the task, and verify state-changing operations. Do not build a second MCP client in bash when direct tools are available. Tool availability is not permission: never invoke setup, evaluation, administrative, or destructive operations merely because they are exposed.
- Treat explicit task protocols as hard constraints. Keep an observe-decide-act loop: inspect each result before choosing the next consequential action. Parallelize only independent read-only queries; perform writes, messages, purchases, bookings, deletions, and other state changes one at a time so failures and changed state can be observed.
- When uncertainty is resolvable from local evidence or tools, resolve it. Ask the user only when a consequential ambiguity truly cannot be resolved and the environment supports interaction.
- For software changes, prefer the smallest patch that satisfies the stated behavior and fits existing abstractions. Do not broaden the redesign speculatively. Run the closest existing tests, inspect the final diff for accidental files and unrelated changes, and distinguish tests actually run from checks that were unavailable.
- Before finishing, re-read the request, check every deliverable path and constraint, and run a final concise verification. In your final response state what was completed and any material limitation; do not claim checks you did not run.

Never cheat: do not obtain an answer from an upstream fix, published solution, hidden tests, evaluation artifacts, or other source that already carries the task's answer. Ordinary documentation and installed library source are allowed. Solve the task yourself.
"""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command in the task container. Prefer focused output. "
            "The default timeout is 300 seconds; set timeout_s for a deliberately "
            "long build or a shorter potentially blocking probe. No single command "
            "may consume most of the overall run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_s": {
                    "type": "integer",
                    "description": "Command timeout in seconds (1-600).",
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

MAX_RESULT_CHARS = 60000
CONTEXT_CHAR_TARGET = 450000
DEFAULT_BASH_TIMEOUT = 300
MAX_BASH_TIMEOUT = 600


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


def _looks_like_policy_interruption(exc: Exception) -> bool:
    """Recognize provider-side screening, which should not crash an otherwise valid run."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        "content policy", "content_policy", "cyber_policy", "safety policy", "was flagged",
    ))


_READ_ONLY_VERBS = {
    "get", "list", "search", "find", "read", "fetch", "lookup", "query",
    "check", "view", "inspect", "retrieve", "show", "count", "preview",
    "calculate", "validate", "verify", "status", "time",
}


def _mcp_is_read_only(tool) -> bool:
    """Conservatively identify queries so only independent reads may be batched."""
    # Names may be qualified as App__method. Split all separators and inspect the
    # method's leading verb. Unknown verbs are deliberately treated as consequential.
    method = tool.remote_name.rsplit("__", 1)[-1].lower()
    words = [part for part in method.replace("-", "_").split("_") if part]
    if words and words[0] in _READ_ONLY_VERBS:
        return True
    # A few APIs put a harmless namespace before the verb (for example, clock_get_*).
    return len(words) > 1 and words[1] in _READ_ONLY_VERBS


MCP_METHOD = """
Application-tool method:
- Start by using the application tools rather than merely announcing a plan. Follow any required conversation or workflow entry point exactly once.
- Read tool results as authoritative new observations. In an interactive workflow, a returned user message is a new turn to answer; do not invent the user's response or end while a material request is unresolved.
- Prefer narrow searches, filters, and pagination over fetching entire collections. Establish the exact record identifiers and current state before changing anything.
- Do not call environment configuration, benchmark-control, seed, runtime, or evaluation utilities unless the user's requested task itself is to administer them.
- Before a consequential action, check prerequisites and scope. After it, verify the returned status before proceeding. Never batch consequential calls that depend on one another.
"""


async def run(instruction, model, world):
    mcp_schemas, mcp_dispatch = await discover(instruction, world)
    system = SYSTEM_PROMPT
    if mcp_schemas:
        system += (
            f"\nThis run exposes {len(mcp_schemas)} application tools discovered from "
            "the declared MCP server(s). Call them directly by name; transport setup "
            "has already been completed.\n" + MCP_METHOD
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    tools = [BASH_TOOL, *mcp_schemas]
    used_application_tool = False
    premature_completion_retried = False
    policy_retries = 0

    for _turn in range(120):
        compact_history(messages)
        try:
            reply = await model.chat(messages, tools=tools)
        except Exception as exc:
            if _looks_like_context_error(exc):
                # A provider may tokenize source/log output much more densely than the
                # character heuristic predicts. Compact once substantially and retry.
                compact_history(messages, aggressive=True)
                reply = await model.chat(messages, tools=tools)
            elif _looks_like_policy_interruption(exc) and policy_retries < 1:
                # Screening can interrupt a benign local research or maintenance task
                # mid-run. Give the model one chance to continue within policy rather
                # than terminating the harness. This does not authorize restricted work.
                policy_retries += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "The provider interrupted the previous attempt before returning "
                        "a response. Reassess the request and continue only in a benign, "
                        "authorized, policy-compliant way. Prefer local analysis and "
                        "high-level explanation; if it is not permissible, state that "
                        "limitation instead of attempting to bypass it."
                    ),
                })
                continue
            else:
                raise

        messages.append(reply.message)
        if not reply.tool_calls:
            # Models occasionally emit only a future-tense plan on the first turn. In
            # an application-backed task that silently ends the run without doing any
            # work. Require one grounded interaction, but only retry once so refusals
            # and genuinely answerable requests cannot loop forever.
            if mcp_schemas and not used_application_tool and not premature_completion_retried:
                premature_completion_retried = True
                messages.append({
                    "role": "user",
                    "content": (
                        "No application tool has been used yet, so the task has not been "
                        "grounded in the available state. Continue by performing the "
                        "appropriate first workflow step rather than only describing a plan."
                    ),
                })
                continue
            return

        consequential_mcp_seen = False
        for call in reply.tool_calls:
            deferred = False
            if call.name in mcp_dispatch and not _mcp_is_read_only(mcp_dispatch[call.name]):
                if consequential_mcp_seen:
                    deferred = True
                consequential_mcp_seen = True

            if deferred:
                output = (
                    "Not executed: another consequential application action was already "
                    "performed in this turn. Inspect its result and current state, then "
                    "decide whether this action is still appropriate and issue it separately."
                )
            elif call.name == "bash":
                command = str(call.arguments.get("command", ""))
                try:
                    timeout_s = int(call.arguments.get("timeout_s", DEFAULT_BASH_TIMEOUT))
                except (TypeError, ValueError):
                    timeout_s = DEFAULT_BASH_TIMEOUT
                timeout_s = max(1, min(timeout_s, MAX_BASH_TIMEOUT))
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
                used_application_tool = True
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
