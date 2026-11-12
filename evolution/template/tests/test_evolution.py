"""Acceptance gate for a harness edit (G_t in the paper), run inside the task container.

Engineering checks only: the edit stays inside the workspace, it is not empty, it stays within this
round's budget of new lines, and the harness still runs offline. There is deliberately no score-based
gate: a version that passes these checks is adopted as the next iteration.
"""

import subprocess
import sys
from pathlib import Path

APP = Path("/app")
WS = APP / "harness_workspace"


def _changed_paths() -> list[str]:
    r = subprocess.run(
        ["git", "-C", str(APP), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:  # rename: keep the destination
            path = path.split(" -> ", 1)[1].strip('"')
        paths.append(path)
    return paths


def test_diff_confined_to_workspace():
    outside = [p for p in _changed_paths() if not p.startswith("harness_workspace/")]
    assert not outside, f"changes outside harness_workspace/: {outside}"


NOTES = "harness_workspace/evolve_notes.md"


def test_diff_not_empty():
    # New, untracked files count too — the workspace is meant to grow — so this
    # goes by ``git status``, not ``git diff``. ``evolve_notes.md`` is updated
    # every round by instruction, so it alone does not make the edit non-empty.
    changed = [p for p in _changed_paths() if p != NOTES]
    assert changed, "no changes beyond evolve_notes.md: the harness must be edited"


def test_added_lines_within_budget():
    # The counting lives in the kernel's smoke module, so the model can run
    # the exact same check itself; the budget file sits in the baseline commit.
    budget_file = APP / "added_lines_budget.txt"
    if not budget_file.is_file():
        return  # no budget configured for this task
    sys.path.insert(0, str(APP))
    from harness_kernel.smoke import count_added_lines

    budget = int(budget_file.read_text().strip())
    added = count_added_lines(APP, WS)
    assert added <= budget, (
        f"added {added} lines, over this round's budget of {budget} "
        "(evolve_notes.md and deleted lines do not count)"
    )


def test_offline_smoke():
    r = subprocess.run(
        [sys.executable, "-m", "harness_kernel.smoke", str(WS)],
        capture_output=True,
        text=True,
        cwd=APP,
        timeout=180,
    )
    assert r.returncode == 0, f"smoke failed:\n{r.stdout}\n{r.stderr}"
