"""Assemble an update task (I_t in the paper) from the template.

The result is an ordinary Harbor task directory::

    <out>/
      instruction.md        the update prompt, with this iteration's budget and input ceiling filled in
      task.toml             agent time limit, collected artifacts (the edited workspace)
      tests/                the acceptance gate (G_t), run after the session
      environment/
        Dockerfile          python:3.12 + git; commits /app as the baseline the gate diffs against
        app/
          harness_kernel/          the fixed kernel, read-only by instruction
          harness_workspace/       the harness to edit (code C_t and notes N_t)
          added_lines_budget.txt   a_t, read by the smoke command and the gate
          eval_trials/<trial>/     the complete run records E_t, copied untouched

Run records are copied whole: every file of every trial (configuration, result, verifier output,
kernel trajectory), successes and failures alike, with nothing summarized, filtered or ranked. Two
bulky exports are left out because they repeat information already present in a readable form
(``are_trace.json`` of GAIA2, ``final_snapshot.tar.gz`` of APEX-Agents), and host-side absolute
paths are scrubbed so the container learns nothing about the host.
"""

import re
import shutil
from pathlib import Path

from .common import KERNEL_DIR, ROOT, TEMPLATE_DIR, WORKSPACE_IGNORE, trials

_TRIAL_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "are_trace.json", "final_snapshot.tar.gz")
_TIMEOUT = re.compile(r"^(timeout_sec = )\d+(\.\d+)?", re.M)


def _scrub_host_paths(dest: Path) -> None:
    root, home = str(ROOT), str(Path.home())
    for p in dest.rglob("*"):
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if root in text or home in text:
            p.write_text(text.replace(root, "").replace(home, "~"))


def build_update_task(
    out: Path,
    *,
    workspace: Path,
    evidence_jobs: list[Path],
    budget: int,
    max_input_tokens: int = 258_400,
    repair: bool = False,
    agent_timeout_sec: float | None = None,
) -> Path:
    """Build the task directory at ``out`` and return it.

    ``repair=True`` builds a repair session (instruction_repair.md): the evidence is then the failed
    update sessions of this iteration, and the session only fixes what made them fail.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    shutil.copytree(TEMPLATE_DIR, out, ignore=WORKSPACE_IGNORE)
    variant = out / "instruction_repair.md"
    if repair:
        variant.replace(out / "instruction.md")
    else:
        variant.unlink()

    instruction = out / "instruction.md"
    text = instruction.read_text()
    text = text.replace("{{ADDED_LINES_BUDGET}}", str(budget)).replace("{{MAX_INPUT_TOKENS}}", f"{max_input_tokens:,}")
    instruction.write_text(text)
    if agent_timeout_sec:
        toml = out / "task.toml"
        # the [agent] timeout is the last ``timeout_sec`` line of the template
        lines = toml.read_text().splitlines()
        idx = max(i for i, l in enumerate(lines) if _TIMEOUT.match(l))
        lines[idx] = _TIMEOUT.sub(rf"\g<1>{float(agent_timeout_sec)}", lines[idx])
        toml.write_text("\n".join(lines) + "\n")

    app = out / "environment" / "app"
    app.mkdir(parents=True, exist_ok=True)
    (app / "added_lines_budget.txt").write_text(f"{budget}\n")
    shutil.copytree(KERNEL_DIR, app / "harness_kernel", ignore=WORKSPACE_IGNORE)
    shutil.copytree(workspace, app / "harness_workspace", ignore=WORKSPACE_IGNORE)
    notes = app / "harness_workspace" / "evolve_notes.md"
    if not notes.exists():  # the released harnesses carry code only; the seed's notes were this bare title
        notes.write_text("# Evolve notes\n")

    n = 0
    for job in evidence_jobs:
        found = trials(job)
        if not found:
            raise ValueError(f"no trial directories in {job}")
        for trial in found:
            dest = app / "eval_trials" / trial.name
            k = 2
            while dest.exists():  # two sessions of one iteration may share a trial name
                dest = app / "eval_trials" / f"{trial.name}-{k}"
                k += 1
            shutil.copytree(trial, dest, ignore=_TRIAL_IGNORE)
            _scrub_host_paths(dest)
            n += 1
    if not n:
        raise ValueError("an update task needs at least one run record")
    return out
