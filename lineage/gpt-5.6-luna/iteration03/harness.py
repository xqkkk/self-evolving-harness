"""A bounded, tool-using agent loop.

The model is allowed to use the container as its working memory.  In
particular, the conversation is deliberately kept bounded: command output can
be enormous and replaying every old response eventually makes otherwise good
runs fail at the model context limit.
"""

SYSTEM_PROMPT = (
    "You are an autonomous agent working inside a Linux container.\n"
    "Use the bash tool to inspect the environment and complete the task.\n"
    "Never cheat: do not obtain the answer by searching the internet or any "
    "similar shortcut; solve the task yourself.\n\n"
    "Work methodically and adapt to the task rather than assuming a particular "
    "benchmark. First identify the requested deliverable, constraints, and the "
    "way success can be checked. Inspect relevant files and available local "
    "documentation before changing anything. Prefer small, targeted commands "
    "over dumping whole directories or large files. Treat command output as "
    "evidence: distinguish facts from guesses, follow promising leads, and "
    "re-check important assumptions. For implementation tasks, make the change "
    "in the expected location and run the most relevant affordable checks; for "
    "research or data tasks, gather enough primary evidence and calculate or "
    "cross-check the result. Do not stop merely because a command exited zero.\n"
    "Keep expensive work bounded and recoverable: prefer staged experiments, "
    "bounded searches, and checkpoints over one unbounded computation. For "
    "long tasks, leave a small useful checkpoint in the task workspace so "
    "you can resume after context is pruned. If a "
    "command fails or is cut short, use the diagnostic to revise the approach "
    "rather than repeating the same command blindly.\n"
    "If details are genuinely missing, make a clearly stated reasonable choice "
    "when possible instead of stalling. Before finishing, verify that the "
    "deliverable is present, correct, and in the requested format.\n"
    "When the task is done, reply with a concise final message and no tool calls."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the task container and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."}
                },
                "required": ["command"],
            },
        },
    }
]

# These are character budgets, intentionally well below the model's token
# limit.  They leave room for tokenization differences and model output.
_CONTEXT_BUDGET = 180_000
_RETRY_BUDGETS = (_CONTEXT_BUDGET, 110_000, 70_000)
_TOOL_OUTPUT_LIMIT = 16_000
# A model can accidentally spend the whole trial in one build, search, or
# package installation.  This is long enough for normal checks but leaves time
# to inspect a failure and choose a smaller or better staged approach.
_COMMAND_TIMEOUT_S = 240


def _clip(text: str, limit: int = _TOOL_OUTPUT_LIMIT) -> str:
    """Keep useful boundaries of a large result while retaining a warning."""
    if len(text) <= limit:
        return text
    head = (limit * 2) // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n[... {omitted} characters omitted; inspect the file or query a smaller range ...]\n"
        + text[-tail:]
    )


def _compact(messages: list[dict], budget: int) -> list[dict]:
    """Return a valid-enough recent transcript under a character budget.

    A turn is an assistant message followed by all of its tool results.  Turns
    are removed as units so that a function result is never left without its
    call.  The initial system prompt and task are always retained.  Responses
    API reasoning items are useful for the immediately preceding turn, but are
    both large and not useful as old memory, so they are retained only there.
    """
    if len(messages) <= 2:
        return messages[:]
    base = [messages[0], messages[1]]
    groups: list[list[dict]] = []
    group: list[dict] = []
    for message in messages[2:]:
        if message.get("role") == "assistant" and group:
            groups.append(group)
            group = []
        group.append(message)
    if group:
        groups.append(group)

    # Normalize outputs before measuring: the complete result remains in the
    # recorder, while the model only needs the bounded display transcript.
    normalized: list[list[dict]] = []
    for gi, old_group in enumerate(groups):
        new_group = []
        for message in old_group:
            m = dict(message)
            if m.get("role") == "tool":
                m["content"] = _clip(str(m.get("content") or ""))
            # Encrypted reasoning is replayable but can dominate the prompt.
            # Keep it only on the newest assistant turn (the one whose tool
            # outputs are about to be consumed).
            if m.get("role") == "assistant" and gi != len(groups) - 1:
                m.pop("_responses_items", None)
            new_group.append(m)
        normalized.append(new_group)

    result = base[:]
    used = sum(len(str(m)) for m in result)
    selected: list[list[dict]] = []
    for candidate in reversed(normalized):
        size = sum(len(str(m)) for m in candidate)
        if selected and used + size > budget:
            break
        if not selected and used + size > budget:
            # Keep the current turn even if one unusually large argument is
            # unavoidable; tool output itself has already been clipped.
            pass
        selected.append(candidate)
        used += size
    selected.reverse()
    if len(selected) < len(normalized):
        result.append({
            "role": "user",
            "content": (
                "Some earlier tool turns were omitted to stay within context. "
                "Use the filesystem and fresh, targeted commands to recover any "
                "detail you need; do not assume omitted work was successful."
            ),
        })
    for candidate in selected:
        result.extend(candidate)
    return result


async def _chat(model, messages):
    """Retry only context-limit failures with progressively smaller history."""
    last = None
    for budget in _RETRY_BUDGETS:
        try:
            return await model.chat(_compact(messages, budget), tools=TOOLS)
        except Exception as exc:
            last = exc
            text = str(exc).lower()
            # A provider-side prompt safety classifier can also reject a
            # long history containing arbitrary task data.  A smaller,
            # recent-only retry often removes the offending stale transcript;
            # do not turn this into an unbounded retry loop.
            retryable = (
                "context" in text
                or "maximum" in text
                or "invalid_prompt" in text
            )
            if not retryable:
                raise
    raise last


async def run(instruction, model, world):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    while True:
        reply = await _chat(model, messages)
        messages.append(reply.message)
        if not reply.tool_calls:
            return
        for call in reply.tool_calls:
            if call.name == "bash":
                command = str(call.arguments.get("command", ""))
                try:
                    r = await world.exec(command, timeout_s=_COMMAND_TIMEOUT_S)
                    # Keep the complete outcome in the world recorder, but do
                    # not let an accidental `cat` consume the next prompt.
                    parts = [_clip(r.stdout)]
                    if r.stderr:
                        parts.append("[stderr]\n" + _clip(r.stderr))
                    parts.append(f"[exit {r.return_code}]")
                    output = "\n".join(p for p in parts if p)
                except Exception as exc:
                    # A tool timeout or transport failure is evidence for the
                    # model, not a reason to lose the entire agent run.  Do not
                    # catch cancellation: the kernel must still end promptly
                    # when the trial itself times out.
                    output = (
                        f"[tool error: {type(exc).__name__}: {exc}]\n"
                        "The command did not complete; inspect any partial "
                        "effects and choose a bounded recovery step."
                    )
            else:
                output = f"unknown tool: {call.name}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )
