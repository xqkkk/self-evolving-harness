"""General autonomous agent loop with bounded context and native MCP bridging."""

from __future__ import annotations

import asyncio

from .mcp_bridge import discover


SYSTEM_PROMPT = """You are a capable autonomous agent working inside a Linux task container. Complete the user's task, not merely describe how to do it.

Operating method:
- First turn the request into a compact contract: required outputs and paths, observable behaviors, constraints, explicit exclusions, and likely acceptance checks. Preserve exact identifiers, field names and types, quoted strings, dates, units, enum values, and protocol ordering. Inspect the environment and relevant inputs before choosing an implementation.
- Work from evidence. Check file formats, dimensions, units, interfaces, and existing project conventions rather than guessing. For repository changes, trace the call path and read nearby tests and analogous implementations before editing. For generated interfaces or schemas, inspect the generated descriptor/API as an independent check that every requested name and type survived generation.
- For search, ranking, allocation, or filtering work, make the objective and tie-breakers explicit. Enumerate the complete eligible candidate set before changing state; distinguish constraints from preferences and "maximize count" from "maximize value". Compute and sanity-check the intended selection before executing it.
- Make focused changes, then validate them with the strongest affordable checks. Exercise edge cases implied by the request and independently inspect final artifacts. Do not modify tests or evaluation machinery.
- Budget the run. Prioritize a correct deliverable over exhaustive exploration. Bound commands that may wait on services, searches, builds, or training; redirect noisy output to files and inspect selected portions. Use the bash timeout_s argument explicitly for genuinely long jobs. Never allow one diagnostic probe to consume the whole task.
- Bash output is clipped when very large. Use targeted grep/find, sed ranges, summaries, or redirect output to a file instead of dumping entire trees, generated files, binaries, or logs.
- If application/MCP tools are present, call those tools directly. Use them to obtain current state before acting, respect confirmation and authorization requirements in the task, and verify state-changing operations. Do not build a second MCP client in bash when direct tools are available. Tool availability is not permission: never invoke setup, evaluation, administrative, or destructive operations merely because they are exposed.
- Treat explicit task protocols as hard constraints. Keep an observe-decide-act loop: inspect each result before choosing the next consequential action. Parallelize only independent read-only queries; perform writes, messages, purchases, bookings, deletions, and other state changes one at a time so failures and changed state can be observed.
- When uncertainty is resolvable from local evidence or tools, resolve it. Ask the user only when a consequential ambiguity truly cannot be resolved and the environment supports interaction; ask the smallest direct question that identifies the missing fact, without unrelated claims or speculative alternatives. When authoritative lookup yields multiple plausible records, include only the verified options needed to disambiguate. Resolve identity, destination, or purchase ambiguity before external side effects rather than doing partial mutations while waiting. Do not silently choose among materially different interpretations.
- Maintain a short mental ledger of the requested outcomes, exact constraints, unresolved facts, and completed actions. Reconcile it whenever a tool result or new user turn changes the state instead of following a stale plan.
- For software changes, translate each requirement into an observable behavior or interface assertion before editing. Build a small behavior matrix covering the requested normal case, boundaries, and failure paths; do not substitute a nearby behavior just because existing tests pass. Prefer the smallest patch that satisfies the whole contract and fits existing abstractions; do not broaden the redesign or switch to a different suspected defect speculatively. Run a focused reproducer or closest existing tests, inspect the final diff and untracked files, and distinguish tests actually run from checks that were unavailable.
- For generated, recovered, or transformed artifacts, validate with an independent reader or invariant when possible rather than trusting the producing script's own success message.
- For quantitative analysis, write down the governing formula, source values, units/scales, date or category mapping, and required rounding before calculating. Trace every requested output to its source cells or records, distinguish missing data from zero, and independently recompute high-risk values rather than relying only on a polished workbook or a nearby approximate result.
- Before finishing, re-read the original request and reconcile every contract item against concrete evidence. Check every deliverable path and constraint, and run a final concise verification. A green broad test suite is not evidence for an untested new edge case. Do not replace a well-supported solution during review merely because another interpretation is imaginable. In your final response state what was completed and any material limitation; do not claim checks you did not run.

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
MAX_MCP_RESULT_CHARS = 160000
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
    """Bound history only when needed, preserving evidence on ordinary runs.

    Earlier versions shortened every observation as soon as seven newer tools had
    run. That saved tokens but silently discarded candidate records, test failures,
    and user constraints even when the request was nowhere near the context limit.
    Reasoning envelopes are cheap to remove; authoritative tool results are retained
    until the whole visible history actually crosses the budget.
    """
    assistant_indexes = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    keep_assistant = 3 if aggressive else 8
    for i in assistant_indexes[:-keep_assistant]:
        # The ordinary content and tool_calls fields preserve protocol continuity.
        messages[i].pop("_responses_items", None)

    if not aggressive and _message_chars(messages) <= CONTEXT_CHAR_TARGET:
        return

    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    keep_tools = 3 if aggressive else 8
    old_limit = 900 if aggressive else 3000
    for i in tool_indexes[:-keep_tools]:
        content = str(messages[i].get("content") or "")
        if len(content) > old_limit:
            half = old_limit // 2
            messages[i]["content"] = (
                content[:half]
                + "\n[... old observation compacted; rerun if needed ...]\n"
                + content[-half:]
            )

    # Tighten only as many older observations as necessary to reach the target.
    # Two most-recent results remain exact even in the worst case (each individual
    # result is separately clipped), preserving current-state evidence.
    if _message_chars(messages) > CONTEXT_CHAR_TARGET:
        for i in tool_indexes[:-2]:
            content = str(messages[i].get("content") or "")
            if len(content) > 1200:
                messages[i]["content"] = (
                    content[:550] + "\n[... old observation compacted ...]\n" + content[-550:]
                )
            if _message_chars(messages) <= CONTEXT_CHAR_TARGET:
                break

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
    # MCP annotations are stronger evidence than naming conventions when supplied.
    if tool.read_only_hint is not None:
        return tool.read_only_hint
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
- Read every result and notification as authoritative new state. In an interactive workflow, a returned user message is a new turn: reconcile it with the outstanding request, do not invent a response, and do not end while a material goal remains unresolved.
- Before acting, decompose the request into an outcome ledger with exact scope, constraints, ordering, literal content, recipients, destinations, and any required deadline. Map every write-tool argument back to that ledger. Preserve user-supplied strings and generic location labels unless the application schema or authoritative record requires a canonical value; do not embellish, paraphrase, or substitute a more specific value when exactness matters. If a notification or later message changes the request, revise the ledger instead of completing a stale action.
- Prefer narrow searches, filters, and pagination over fetching entire collections. Establish exact records and current state before changing anything. For multi-record selection, finish discovery, deduplicate and compare candidates against explicit criteria before mutating state. If a result says it was clipped, narrow or paginate rather than assuming omitted records are irrelevant.
- Keep the user's requested scope separate from optional adjacent work. Do not turn a suggestion into authorization, and do not claim a prerequisite, success, or final state that a tool result did not establish.
- Do not call environment configuration, benchmark-control, seed, runtime, or evaluation utilities unless the user's requested task itself is to administer them.
- Before a consequential action, check identity, parameters, prerequisites, authorization, and whether ambiguity or confirmation remains. Compare names, addresses, dates, titles, message text, and other tool arguments against the source records and request rather than relying on memory. After the action, verify the returned status before proceeding. Never batch consequential calls that depend on one another.
- When communicating through a user-interface tool, match the requested purpose directly. Clarifications should contain only the necessary question and any verified choices needed to answer it; simple completion notices should be concise and should not add conditions, speculative alternatives, or unsupported details. Use a later user message or notification as new state, not as implicit permission for adjacent work.
- For simulated-time requests, use the authoritative clock, compute the exact requested boundary, preserve enough pre-change state to reconstruct deleted or modified records, and re-check state after waiting; do not substitute wall-clock time or an approximate number of polling calls.
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
    used_bash = False
    premature_completion_retried = False
    final_reconciliation_requested = False
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
            # Give shell-backed work one final contract reconciliation in the existing
            # context. A previous clean-context reviewer often lost the evidence and
            # rationale behind a correct implementation, then replaced it with a
            # speculative alternative. Keeping the same context still catches forgotten
            # deliverables while preserving the implementer's observations and tests.
            if not mcp_schemas and used_bash and not final_reconciliation_requested:
                final_reconciliation_requested = True
                messages.append({
                    "role": "user",
                    "content": (
                        "Before finalizing, reconcile the original request item by item "
                        "against the actual workspace and the evidence already gathered. "
                        "Check exact paths, identifiers, field names/types, failure "
                        "behavior, untracked files, and the final diff or artifacts. Test "
                        "a high-risk requested behavior if it is not yet covered. Fix only "
                        "a concrete, reproducible mismatch; do not replace a supported "
                        "solution with a speculative alternate interpretation. Then give "
                        "the final report."
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
                used_bash = True
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
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": clip(
                        str(output),
                        MAX_MCP_RESULT_CHARS if call.name in mcp_dispatch else MAX_RESULT_CHARS,
                    ),
                }
            )

    # Avoid an unbounded pathological loop while still allowing long interactive tasks.
    messages.append(
        {
            "role": "user",
            "content": "Tool-call budget is exhausted. Finalize now with the best completed result.",
        }
    )
    await model.chat(messages, tools=tools)
