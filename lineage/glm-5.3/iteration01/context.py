"""Keep the conversation inside the model's window.

Two things grow a history: tool output, and the reasoning items the model
client replays on every call. `clip` bounds each tool result as it arrives, so
no single output can overflow the window on its own. `compact` stubs the text
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
    "[harness] The conversation grew past the context budget, so tool output and "
    "replayed reasoning from older turns were dropped. The commands and their "
    "outcomes are still listed; for anything you need verbatim, keep it in a file "
    "in the container and re-read it from there."
)


def clip(text, limit=CLIP_LIMIT, head=CLIP_HEAD, tail=CLIP_TAIL):
    """Shorten a tool result, keeping both ends and saying what was cut."""
    text = text or ""
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


def compact(messages, *, keep_last=8, hard=False, target_chars=200000):
    """Trim history in place, keeping the last `keep_last` tool results verbatim."""
    touched = 0
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    protected = set(tool_idx[-keep_last:])
    if hard:
        asst = [i for i, m in enumerate(messages) if any(m.get(k) for k in STASH)]
        for i in asst[: max(len(asst) - keep_last, 0)]:
            for k in STASH:
                messages[i].pop(k, None)
            touched += 1
    for i in tool_idx:
        if size(messages) <= target_chars:
            break
        if i in protected:
            continue
        content = str(messages[i].get("content") or "")
        if len(content) < 600:
            continue
        messages[i]["content"] = (
            f"[older tool result, {len(content)} chars, dropped for context]" + content[:200]
        )
        touched += 1
    return touched
