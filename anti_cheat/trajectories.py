"""The commands a run executed and what they printed, read from the three kinds of run record.

* our kernel: ``agent/kernel_trajectory.jsonl`` (``exec`` events; ``model_call`` events are checked for
  provider-side web search, which bypasses the sandbox's network isolation);
* Codex: ``agent/codex.txt`` (JSON events; ``command_execution`` and ``web_search`` items);
* Claude Code: ``agent/claude-code.txt`` (stream-json; ``tool_use`` blocks paired with their ``tool_result``).

Calls to the task's own environment tools (MCP) are not steps: they are the task's interface, not a way around it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step:
    command: str
    output: str
    web_search: bool = False  # a provider- or agent-side web search, outside the sandbox


def _json_lines(path: Path):
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


def kernel_steps(trial: Path):
    path = trial / "agent" / "kernel_trajectory.jsonl"
    if not path.is_file():
        return
    for e in _json_lines(path):
        if e.get("type") == "model_call":
            # Only what the harness DECLARED and what the provider actually RAN: grepping the whole event once
            # matched a file name from an `ls` of the SDK in a tool result.
            req, resp = e.get("request") or {}, e.get("response") or {}
            declared = any("web_search" in str(t.get("type", "")) or "web_search" in str(t.get("name", ""))
                           for t in (req.get("tools") or []) if isinstance(t, dict))
            used = any("web_search_call" in str(i.get("type", ""))
                       for i in (resp.get("items") or []) + (resp.get("output") or []) if isinstance(i, dict))
            if declared or used:
                yield Step("model declared/used provider-side web_search",
                           json.dumps({"declared": declared, "used": used}), web_search=True)
        elif e.get("type") == "exec":
            yield Step(e.get("command") or "", (e.get("stdout") or "") + (e.get("stderr") or ""))


def codex_steps(trial: Path):
    path = trial / "agent" / "codex.txt"
    if not path.is_file():
        return
    for e in _json_lines(path):
        item = e.get("item") if e.get("type") == "item.completed" else None
        if not isinstance(item, dict):
            continue
        if item.get("type") == "command_execution":
            yield Step(str(item.get("command") or ""), str(item.get("aggregated_output") or ""))
        elif item.get("type") in ("web_search", "web_search_call"):
            action = item.get("action") if isinstance(item.get("action"), dict) else {}
            query = item.get("query") or action.get("query") or ""
            yield Step(f"web_search {query}", "", web_search=True)


def _render_tool(name: str, args: dict) -> str:
    if name == "Bash":
        return str(args.get("command") or "")
    if name in ("Read", "Write", "Edit"):
        return f"{name} {args.get('file_path', '')}"
    if name in ("Grep", "Glob"):
        return f"{name} {args.get('pattern', '')} {args.get('path', '')}".strip()
    if name == "WebFetch":
        return f"WebFetch {args.get('url', '')}"
    if name == "WebSearch":
        return f"WebSearch {args.get('query', '')}"
    return f"{name} {json.dumps(args, ensure_ascii=False)[:600]}"


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


def _is_environment_tool(name: str) -> bool:
    """The task's MCP tools and Claude Code's MCP resource tools: the task's interface (as Codex's MCP calls)."""
    return name.startswith("mcp__") or name in ("ListMcpResourcesTool", "ReadMcpResourceTool")


def claude_code_steps(trial: Path):
    path = trial / "agent" / "claude-code.txt"
    if not path.is_file():
        return
    calls: dict[str, tuple[str, dict]] = {}
    for e in _json_lines(path):
        message = e.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if e.get("type") == "assistant" and block.get("type") == "tool_use":
                calls[block.get("id")] = (str(block.get("name") or ""), block.get("input") or {})
            elif e.get("type") == "user" and block.get("type") == "tool_result":
                name, args = calls.pop(block.get("tool_use_id"), ("", {}))
                if name and not _is_environment_tool(name):
                    yield Step(_render_tool(name, args), _result_text(block.get("content")),
                               web_search=name == "WebSearch")
    for name, args in calls.values():  # calls without a result (the run ended first)
        if _is_environment_tool(name):
            continue
        yield Step(_render_tool(name, args), "", web_search=name == "WebSearch")


def steps(trial: Path) -> list[Step]:
    """All steps of a trial, whichever harness produced it."""
    for reader in (kernel_steps, codex_steps, claude_code_steps):
        out = list(reader(trial))
        if out:
            return out
    return []
