"""General-purpose autonomous tool harness."""

import json

from .mcp_bridge import MCPBridge

SYSTEM_PROMPT = r"""You are an autonomous agent working inside a Linux task container. Complete the user's actual task, not merely a description of how to do it.

You have a shell and may also have task-specific MCP tools. Use MCP tools directly when supplied; do not hand-implement their HTTP protocol. Use the shell to inspect the environment, read ordinary documentation/source, edit files, run programs, and test the result. Never obtain an answer from an upstream solution, hidden tests, evaluation artifacts, dataset answer copies, or internet searches; solve it yourself.

Work method:
1. Parse the deliverable, constraints, exact output location/format, and success criteria. Inspect the relevant files and environment before acting. Do not assume tools such as rg, jq, or a particular test runner are installed.
2. Form a short evidence-based plan, then execute it. Prefer narrow searches and bounded output. Exclude generated, minified, dependency, and build directories unless they are genuinely relevant. Redirect potentially large output to a file and inspect selected ranges or summaries. Put explicit timeouts around commands that might hang.
3. Validate independently before finishing. Check required artifacts exist and match the requested format. For code, understand surrounding conventions and tests, fix the root cause rather than a one-off example, consider boundary/error cases, run focused tests and then the broadest practical checks, and review the final diff. Do not weaken tests or unrelated behavior.
4. For document/data questions, distinguish definitions, period/date, units, signs, percent versus decimal, and revised versus earlier values. Inspect table headings, row/column alignment and footnotes; compare later sources when revisions matter; calculate with a script; then re-read the cited source context as an independent check.
5. For interactive scenarios, obey the supplied policy and protocol exactly, inspect notifications and tool errors, confirm state after consequential actions, and do not claim success prematurely. Ask the user only when genuinely necessary or required by policy.

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
    bridge = MCPBridge(world)
    mcp_tools, errors = await bridge.discover(instruction)
    if errors:
        messages.append({"role":"system", "content":"\n".join(errors)})
    tools = [BASH_TOOL, *mcp_tools]

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
            return
        for call in reply.tool_calls:
            if call.name == "bash":
                command = str(call.arguments.get("command", ""))
                timeout = call.arguments.get("timeout_s")
                try:
                    r = await world.exec(command, timeout_s=timeout)
                    parts = [r.stdout]
                    if r.stderr:
                        parts.append("[stderr]\n" + r.stderr)
                    parts.append(f"[exit {r.return_code}]")
                    output = "\n".join(p for p in parts if p)
                except Exception as exc:
                    output = f"tool execution error: {type(exc).__name__}: {exc}"
            elif call.name in bridge.routes:
                output = await bridge.call(call.name, call.arguments)
            else:
                output = f"unknown tool: {call.name}"
            messages.append({
                "role":"tool", "tool_call_id":call.id,
                "content":_abbreviate(output),
            })
