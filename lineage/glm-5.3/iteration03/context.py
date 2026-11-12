"""Keep the conversation inside the model's window.

Two things grow a history: tool output, and the reasoning items the model
client replays on every call. `clip` bounds each tool result as it arrives, so
no single output can overflow the window on its own — and the bound it uses
depends on how full the history already is, because a result read in half is
a result the model has to ask for again. `compact` stubs the text
of old tool results — pure information loss, but the history's shape is
untouched, so the request stays well formed; in hard mode it also drops the
replayed reasoning of old turns, which the client rebuilds from their text and
tool calls. The window size is not guessed at: the harness measures the real
prompt length from the client's usage counters and compacts on that.
"""

# Whatever the client stashes on an assistant message to replay its reasoning.
# Dropping it is the one lever that really shrinks a long history; the client
# rebuilds the message from its text and tool calls, so the request stays
# well formed, and the harness keeps a backup in case the call rejects it.
STASH = ("_responses_items", "reasoning_details", "reasoning", "reasoning_content")

CLIP_LIMIT = 24000
CLIP_HEAD = 16000
CLIP_TAIL = 5000

TRIM_NOTE = (
    "[harness] The conversation grew past the context budget, so tool output, "
    "replayed reasoning and old narration were dropped; the recent turns are "
    "intact. The task statement is at /tmp/.harness_task.md and your plan at "
    "/tmp/.harness_plan.md — re-read them from there rather than trusting "
    "memory, and keep facts you will need again in a file in the container."
)


def clip_budget(hist_chars):
    """How much of one tool result the window can still afford.

    A result is worth more whole than halved: the fact the task needs is as
    likely in the middle as at either end. So a result that arrives while the
    history is still small is allowed more than the default, and once the
    history is heavy the allowance shrinks rather than the window filling and
    the whole turn being replayed for nothing.
    """
    if hist_chars < 50_000:
        return 34_000
    if hist_chars < 120_000:
        return CLIP_LIMIT
    if hist_chars < 220_000:
        return 16_000
    return 10_000


def clip(text, limit=None, head=None, tail=None):
    """Shorten a tool result, keeping both ends and saying what was cut."""
    text = text or ""
    limit = CLIP_LIMIT if limit is None else limit
    head = int(limit * 0.66) if head is None else head
    tail = int(limit * 0.2) if tail is None else tail
    if len(text) <= limit:
        return text
    skipped = len(text) - head - tail
    return (
        text[:head]
        + f"\n... [{skipped} of {len(text)} characters skipped; use a narrower "
        "command (grep / sed -n / head / tail) or write the output to a file and "
        "read the part you need] ...\n"
        + text[-tail:]
    )


def size(messages):
    """Rough character size of a history, reasoning items included."""
    n = 0
    for m in messages:
        n += len(str(m.get("content") or "")) + 120
        n += sum(len(str(m.get(k) or "")) for k in STASH)
        for tc in m.get("tool_calls") or []:
            n += len(str((tc.get("function") or {}).get("arguments") or ""))
    return n


def _stub(content, keep=300):
    """What a dropped block leaves behind: its size, then its opening."""
    return f"[dropped for context, {len(content)} chars] " + content[:keep]


def compact(messages, *, keep_last=8, hard=False, target_chars=160000):
    """Trim history in place until it really is under `target_chars`.

    A trim that cannot reach its target is worse than none: the history sits
    at the trigger, every later turn trims again, and the window still fills.
    So `hard` runs a schedule of passes that protect less and less — the last
    few results, then fewer, then one — and each pass also drops the replayed
    reasoning of older turns and stubs the model's own long narration, which
    is where the bulk of a chatty history actually is. The very last tool
    result is never stubbed: it is the one the model has not answered yet.
    `size()` decides when to stop, so what leaves the window fits it.
    """
    touched = 0
    schedule = (keep_last,) if not hard else (
        keep_last, max(keep_last // 2, 2), 2, 1)
    for keep in schedule:
        for i, m in enumerate(messages):
            if any(m.get(k) for k in STASH) and i < len(messages) - keep - 1:
                for k in STASH:
                    m.pop(k, None)
                touched += 1
        tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        for i in tool_idx[:-keep] if keep else tool_idx[:-1]:
            content = str(messages[i].get("content") or "")
            if len(content) >= 600:
                messages[i]["content"] = _stub(content)
                touched += 1
        if hard:
            for m in messages:
                c = m.get("content")
                if m.get("role") == "assistant" and isinstance(c, str) and len(c) > 1500:
                    m["content"] = _stub(c, 600)
                    touched += 1
        if size(messages) <= target_chars:
            break
    return touched
