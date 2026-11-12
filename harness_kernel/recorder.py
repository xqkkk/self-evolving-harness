"""Kernel-owned trajectory recorder: an append-only, raw event log.

Everything the kernel sees goes in, unabridged and as it happened, one JSON object per line in
``kernel_trajectory.jsonl``:

- ``run_start``     the task instruction as handed to the harness, the model, the workspace hash,
                    and the MCP servers the task declared.
- ``model_call``    one per model call: the request as the harness passed it, the full response (raw
                    output items, text, parsed tool calls, status, usage) and how long it took, or,
                    if the call raised, the request plus the error.
- ``exec``          one per command run in the task container: the command, cwd/env/timeout as
                    requested, exit code, complete stdout and stderr and how long it took, or the error.
- ``harness_crash`` the exception that ended the harness, with traceback.
- ``run_end``       the outcome (error or none), token totals, wall time.

Nothing is truncated or summarized, but nothing is written twice either. Each call re-sends the
whole history, so a request is recorded as a delta against the previous one:

- ``request.messages_appended``  the previous call's messages were passed again unchanged, and these
  were added after them (the usual turn);
- ``request.messages``           the whole list, written out on the first call and whenever the
  history was rewritten rather than only appended to;
- ``request.tools``              the tool schemas, written only when they differ from the previous
  call's; absent means the same tools as last time.

Two things are dropped from recorded messages because they duplicate what is already on file: the
opaque ``encrypted_content`` of reasoning items and the ``_responses_items`` a client stashes on
assistant messages to replay them (their content is the ``response.items`` of the call that
produced them).

Each line is written and flushed the moment the event happens, so a run that is killed still leaves
everything up to that point on disk. Every event carries ``t`` (seconds since the run began) and
``at`` (UTC wall clock).
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

FILENAME = "kernel_trajectory.jsonl"

_DROP_KEYS = {"encrypted_content", "_responses_items"}


def strip_encrypted(obj):
    """Deep-copy ``obj`` without ciphertext or the client's replay stash."""
    if isinstance(obj, dict):
        return {k: strip_encrypted(v) for k, v in obj.items() if k not in _DROP_KEYS}
    if isinstance(obj, list):
        return [strip_encrypted(x) for x in obj]
    return obj


def _credentials() -> list[tuple[str, str]]:
    """``(placeholder, value)`` for every credential in the host environment.

    Matching is by value, not by pattern: only the exact strings that really are credentials get
    masked. A task container may receive such values as environment variables, and an agent that runs
    ``env`` while exploring would otherwise put them into its own stdout, which the trajectory records
    verbatim. Run records are later shown to the proposer and may be published, so the value is
    replaced at the moment the line is written and the surrounding output stays intact.
    """
    out = []
    for k, v in os.environ.items():
        if len(v) >= 20 and any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET")):
            out.append((f"[REDACTED:{k}]", v))
    return sorted(out, key=lambda kv: -len(kv[1]))


class Recorder:
    def __init__(self, path: Path | str | None = None) -> None:
        self.events: list[dict] = []
        self._t0 = time.monotonic()
        self._fh = None
        self._last_request: tuple[list, list] | None = None
        self._credentials = _credentials()
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def elapsed(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def model_request(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """The request field of a ``model_call``: in full the first time and whenever the history was
        rewritten, as a delta when it only grew."""
        msgs = strip_encrypted(list(messages))
        tls = strip_encrypted(list(tools or []))
        prev = self._last_request
        request: dict = {}
        if prev is not None and len(msgs) >= len(prev[0]) and msgs[: len(prev[0])] == prev[0]:
            request["messages_appended"] = msgs[len(prev[0]):]
        else:
            request["messages"] = msgs
        if prev is None or tls != prev[1]:
            request["tools"] = tls
        self._last_request = (msgs, tls)
        return request

    def record(self, type_: str, **data) -> None:
        event = {"type": type_, "t": self.elapsed(),
                 "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **data}
        self.events.append(event)
        if self._fh is not None:
            line = json.dumps(event, ensure_ascii=False, default=str)
            for placeholder, value in self._credentials:
                if value in line:
                    line = line.replace(value, placeholder)
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
