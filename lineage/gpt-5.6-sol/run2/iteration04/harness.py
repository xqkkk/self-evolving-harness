"""General-purpose autonomous tool harness."""

import asyncio
import json

from .mcp_bridge import MCPBridge

SYSTEM_PROMPT = r"""You are an autonomous agent working inside a Linux task container. Complete the user's actual task, not merely a description of how to do it.

You have a shell and may also have task-specific MCP tools. Use MCP tools directly when supplied; do not hand-implement their HTTP protocol. Use the shell to inspect the environment, read ordinary documentation/source, edit files, run programs, and test the result. Never obtain an answer from an upstream solution, hidden tests, evaluation artifacts, dataset answer copies, or internet searches; solve it yourself.

Work method:
1. Parse the deliverable, constraints, exact output location/format, and success criteria into a concrete checklist; keep track of pending parts until the final audit. Inspect the relevant files and environment before acting. Do not assume tools such as rg, jq, or a particular test runner are installed.
2. Form a short evidence-based plan, then execute it. Prefer narrow searches and bounded output. Exclude generated, minified, dependency, and build directories unless they are genuinely relevant. Redirect potentially large output to a file and inspect selected ranges or summaries. Put explicit timeouts around commands that might hang. When a request says all, highest/lowest, newest, or otherwise ranks a collection, establish that pagination and search coverage are complete before choosing; distinguish products from their variants and deduplicate by the requested identity.
3. Validate independently before finishing. Check required artifacts exist and match the requested format. For code, turn every stated requirement into a checklist of observable behavior and exact public contracts (names, signatures, defaults, ordering, error behavior), inspect all production call sites, and preserve compatibility only where it does not weaken the requested contract. Fix the root cause rather than a one-off example, consider boundary/error cases, run focused tests and then the broadest practical checks, and review the final diff. A newly failing existing test is evidence of a compatibility bug: fix it rather than excluding it or declaring it obsolete. Do not weaken tests or unrelated behavior.
4. For document/data questions, distinguish exact definitions, populations, period/date, units, signs, totals versus components, and percent versus decimal. Inspect table titles, headings, row/column alignment and footnotes; when revisions matter, find the latest source that reports the requested period and verify that later sources did not revise it. A question about a page, chart, or visual layout requires visual evidence: page boundaries in extracted text are not a substitute, so locate/render the source page or its image when available and explicitly define any visual counting rule (including ties and endpoints). State the observations and formula, calculate with a script, and re-read the source context as an independent check. Do not silently substitute a similar metric.
5. For interactive scenarios, treat the requested action and its stopping point literally; do not extend a preparatory or informational request into a downstream committing operation. Capture the initial state when relevant, inspect every notification and tool error, honor temporal cutoffs, and verify final state after consequential actions. Re-evaluate the full request when new messages arrive; do not guess a materially ambiguous target or execute multiple possible interpretations. Ask one concise clarification before the blocked action, then stop unless the protocol explicitly supplies a way to await the answer. Calls in one assistant turn are conceptually unordered, so batch only independent operations—put reads and dependent or ordered mutations in separate turns. Keep a timeline for deadline-based workflows: anchor it to the event the user named, process arrivals in order, act at the earliest required event, and never replace “exactly” or “immediately” with an arbitrary polling delay. After a timeout or transport failure, inspect state before replaying any call that may already have committed. Treat user-facing sends as consequential actions: send only requested checkpoints, clarifications, and final answers, never progress chatter. Do not announce completion before waits or follow-up events are resolved. If the protocol requires an initial tool call, make it before conversational text; a readiness statement is not completion.
6. Manage the task's time as a resource. Sample or benchmark potentially expensive work before scaling it, prefer batching or parallelism over long serial loops, set realistic command timeouts, and preserve enough time for validation and the required artifact. If one approach is too slow or repeatedly fails, diagnose once and switch strategies rather than retrying blindly.

Tool output shown to you may be abbreviated when it is very large; the full command still ran. If that happens, rerun a targeted query or save and inspect the relevant slice. Preserve user work. Keep going until the result is verified, then give a concise final response and make no further tool calls."""

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the task container. Use timeout_s for potentially slow or hanging commands.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 3600,
                              "description": "Command timeout in seconds (default 3600)."},
            },
            "required": ["command"],
        },
    },
}

MAX_RESULT_CHARS = 50000


def _abbreviate(text, limit=MAX_RESULT_CHARS):
    if len(text) <= limit:
        return text
    side = (limit - 500) // 2
    omitted = len(text) - 2 * side
    return (text[:side] + f"\n\n[... {omitted:,} characters omitted by the harness. "
            "The command completed; rerun a narrower query or inspect a saved file ...]\n\n"
            + text[-side:])


def _message_size(messages):
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)


def _compact(messages, emergency=False):
    """Bound old tool transcript growth without discarding recent evidence."""
    threshold = 350000 if emergency else 650000
    if _message_size(messages) <= threshold:
        return
    tool_indexes = [i for i, m in enumerate(messages[:-6]) if m.get("role") == "tool"]
    for i in tool_indexes:
        content = messages[i].get("content", "")
        target = 1200 if emergency else 6000
        if len(content) > target:
            side = max(300, (target - 150) // 2)
            messages[i]["content"] = (
                content[:side] + "\n[older tool output compacted]\n" + content[-side:]
            )
        if _message_size(messages) <= threshold:
            break


async def run(instruction, model, world):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    bridge = MCPBridge(world, model)
    mcp_tools, errors = await bridge.discover(instruction)
    if errors:
        messages.append({"role":"system", "content":"\n".join(errors)})
    tools = [BASH_TOOL, *mcp_tools]

    mcp_calls = 0
    tool_nudge_used = False

    async def execute(call):
        if call.name == "bash":
            command = str(call.arguments.get("command", ""))
            timeout = call.arguments.get("timeout_s")
            try:
                r = await world.exec(command, timeout_s=timeout)
                parts = [r.stdout]
                if r.stderr:
                    parts.append("[stderr]\n" + r.stderr)
                parts.append(f"[exit {r.return_code}]")
                return "\n".join(p for p in parts if p)
            except Exception as exc:
                return f"tool execution error: {type(exc).__name__}: {exc}"
        if call.name in bridge.routes:
            return await bridge.call(call.name, call.arguments)
        return f"unknown tool: {call.name}"

    while True:
        _compact(messages)
        try:
            reply = await model.chat(messages, tools=tools)
        except Exception as exc:
            if "context_length" not in str(exc).lower():
                raise
            _compact(messages, emergency=True)
            reply = await model.chat(messages, tools=tools)
        messages.append(reply.message)
        if not reply.tool_calls:
            # MCP-backed tasks cannot normally be completed without touching the
            # advertised service.  Give a premature prose-only answer one chance
            # to self-correct, rather than silently ending the run.
            if mcp_tools and not mcp_calls and not tool_nudge_used:
                tool_nudge_used = True
                messages.append({
                    "role": "system",
                    "content": (
                        "No MCP operation has been performed yet. Re-check the task "
                        "protocol. If it requires an initial or other MCP call, make "
                        "that call now instead of merely acknowledging the task."
                    ),
                })
                continue
            return

        mcp_calls += sum(call.name in bridge.routes for call in reply.tool_calls)
        # Keep stateful MCP operations in a deterministic order. Besides avoiding
        # application races, a streamable-HTTP session has one logical stream on
        # which reverse requests and delayed responses are correlated. Pure shell
        # inspections/test shards remain safe to run concurrently.
        if any(call.name in bridge.routes for call in reply.tool_calls):
            outputs = [await execute(call) for call in reply.tool_calls]
        else:
            outputs = await asyncio.gather(*(execute(call) for call in reply.tool_calls))
        for call, output in zip(reply.tool_calls, outputs):
            messages.append({
                "role":"tool", "tool_call_id":call.id,
                "content":_abbreviate(output),
            })
