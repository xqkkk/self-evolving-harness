"""Offline check that a harness workspace is a working harness.

    python -m harness_kernel.smoke <workspace>

It asks only what the kernel itself needs: every Python file compiles, the workspace imports, ``run``
is there, and one scripted run (a fake model that calls whichever tool the harness declares first,
then says it is done) gets through without raising. It makes no assumption about what tools the
harness offers, what they are called, or what they do.

When ``<workspace>/../added_lines_budget.txt`` exists (it does inside an update task), the check also
counts the lines added in this session and fails if they exceed the budget. This is the same count
the acceptance gate applies, so the proposer can run the exact check itself.
"""

import asyncio
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

from .loader import load_workspace
from .model import FakeModel
from .recorder import Recorder
from .world import LocalWorld

_INSTRUCTION = "Smoke check: do nothing of consequence and finish."
_RUN_TIMEOUT_S = 60


def _compile_all(workspace: Path) -> None:
    for p in sorted(workspace.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        compile(p.read_text(), str(p), "exec")  # a SyntaxError names the file


def run_smoke(workspace: Path, sandbox: Path | None = None, world=None) -> dict:
    """``world`` defaults to a ``LocalWorld`` in a temporary directory."""
    workspace = Path(workspace)
    tmp = None
    if sandbox is None:
        tmp = tempfile.TemporaryDirectory(prefix="harness_kernel_smoke_")
        sandbox = Path(tmp.name)
    try:
        _compile_all(workspace)
        module = load_workspace(workspace)
        run = getattr(module, "run", None)
        assert callable(run), "harness.py defines no run(instruction, model, world)"
        assert inspect.iscoroutinefunction(run), "run must be `async def`"

        recorder = Recorder()
        model = FakeModel(recorder)
        world = world or LocalWorld(sandbox, recorder)
        try:
            asyncio.run(asyncio.wait_for(run(_INSTRUCTION, model, world), timeout=_RUN_TIMEOUT_S))
        except TimeoutError:
            raise AssertionError(f"run did not finish within {_RUN_TIMEOUT_S}s of the model's final reply") from None
        assert model.calls >= 1, "run returned without ever calling the model"
        return {"model_calls": model.calls, "events": len(recorder.events)}
    finally:
        if tmp is not None:
            tmp.cleanup()


def count_added_lines(repo: Path, workspace: Path) -> int:
    """Lines added versus the repo's HEAD commit, counted only under ``workspace``: the added column of
    ``git diff HEAD --numstat`` plus the full line count of untracked files. Deleted lines,
    ``<workspace>/evolve_notes.md`` and bytecode do not count."""
    rel = Path(workspace).resolve().relative_to(Path(repo).resolve()).as_posix()
    notes = f"{rel}/evolve_notes.md"

    def counts(path: str) -> bool:
        return (path.startswith(f"{rel}/") and path != notes
                and "__pycache__" not in path and not path.endswith(".pyc"))

    added = 0
    out = subprocess.run(["git", "-C", str(repo), "diff", "HEAD", "--numstat"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] != "-" and counts(parts[2].strip().strip('"')):
            added += int(parts[0])
    out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "-uall"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("??"):
            path = line[3:].strip().strip('"')
            if counts(path) and (Path(repo) / path).is_file():
                added += len((Path(repo) / path).read_text(errors="ignore").splitlines())
    return added


def _budget_line(ws: Path) -> str | None:
    """When ``<ws>/../added_lines_budget.txt`` exists, count against it."""
    budget_file = ws.parent / "added_lines_budget.txt"
    if not budget_file.is_file():
        return None
    budget = int(budget_file.read_text().strip())
    added = count_added_lines(ws.parent, ws)
    line = f"ADDED LINES {added}/{budget} (deletions and evolve_notes.md not counted)"
    if added > budget:
        raise AssertionError(f"over the added-lines budget: {line}")
    return line


def main() -> int:
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("harnesses/seed")
    try:
        result = run_smoke(ws)
        budget_line = _budget_line(ws)
    except Exception as e:
        print(f"SMOKE FAIL [{ws}]: {type(e).__name__}: {e}")
        return 1
    if budget_line:
        print(budget_line)
    print(f"SMOKE OK [{ws}]: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
