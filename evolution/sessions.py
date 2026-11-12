"""Update sessions: run one, judge its outcome, adopt the result, and the repair pipeline.

An update session ends in one of two ways:

* ``ok``: it ended normally, changed the harness beyond the notes, and passed the acceptance gate
  (changes confined to the workspace, non-empty, within the budget of new lines, offline smoke OK);
* ``failed``: it ran out of time, overflowed its context, crashed, changed nothing, or did not pass the
  gate. Only a session that ended normally can be adopted.

Repair pipeline (first stage, up to ``repair_sessions`` repair sessions per iteration): when the update
session fails, a repair session gets this iteration's failed sessions as evidence and fixes only what
made them fail, editing the same workspace. The harness that runs the repair session falls back one
generation per failed repair (gen_{t-1}, gen_{t-2}, ... down to the seed), so the agent carrying the
repair is always one that is known to run. After a passed repair, the update session runs again on the
repaired workspace; if it fails again and repairs remain, the next repair sees every session so far.
"""

import difflib
import shutil
from pathlib import Path

import json

from .common import read_json, reward_of, trials, write_json
from .harbor import run_update_task
from .update_task import build_update_task

REPAIR_BUDGET = 100  # a repair session fixes; it is not an occasion to add features


def workspace_files(ws: Path) -> dict[str, str]:
    out = {}
    for p in sorted(Path(ws).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            out[p.relative_to(ws).as_posix()] = p.read_text(errors="replace")
    return out


def diff_md(base: Path, candidate: Path) -> str:
    old, new = workspace_files(base), workspace_files(candidate)
    chunks = []
    for name in sorted(set(old) | set(new)):
        d = list(difflib.unified_diff(old.get(name, "").splitlines(keepends=True), new.get(name, "").splitlines(keepends=True),
                                      fromfile=f"base/{name}", tofile=f"cand/{name}"))
        if d:
            chunks.append("```diff\n" + "".join(d) + "\n```")
    return "\n\n".join(chunks) or "(no file differences)"


def edited_workspace(job: Path) -> Path | None:
    ts = trials(job)
    src = ts[0] / "artifacts" / "app" / "harness_workspace" if ts else None
    return src if src is not None and src.is_dir() else None


def _gate_failures(trial: Path) -> str:
    tests = ((read_json(trial / "verifier" / "ctrf.json") or {}).get("results") or {}).get("tests") or []
    return ", ".join(t["name"].split("::")[-1] for t in tests if t.get("status") != "passed") or "?"


def _run_error(trial: Path) -> str:
    """The error the kernel recorded for the run (``run_end``, or a ``harness_crash``), empty if none."""
    error = ""
    path = trial / "agent" / "kernel_trajectory.jsonl"
    for line in (path.read_text(errors="replace").splitlines() if path.is_file() else []):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "run_end":
            error = e.get("error") or ""
        elif e.get("type") == "harness_crash" and not error:
            error = e.get("error") or ""
    return error


def session_outcome(job: Path, base_ws: Path) -> tuple[str, str]:
    """``("ok", "")`` or ``("failed", why)`` for one finished update session."""
    ts = trials(job)
    if not ts or not (ts[0] / "result.json").is_file():
        raise RuntimeError(f"the update session {job} produced no result; inspect the job, remove it, and rerun")
    trial = ts[0]
    exc = ((read_json(trial / "result.json") or {}).get("exception_info") or {}).get("exception_type")
    err = _run_error(trial)
    if exc == "AgentTimeoutError" or "cancelled" in err:
        return "failed", "the session ran out of time"
    if "context_length_exceeded" in err:
        return "failed", "the session overflowed its context"
    if err:
        return "failed", f"the harness crashed: {err[:160]}"
    if reward_of(trial) != 1.0:
        return "failed", f"did not pass the gate ({_gate_failures(trial)})"
    src = edited_workspace(job)
    if src is None or diff_md(base_ws, src) == "(no file differences)":
        return "failed", "no change to the harness"
    return "ok", ""


def run_session(*, task_dir: Path, model: str, runner_ws: Path, base_ws: Path, jobs_dir: Path, job_name: str,
                env: str | None) -> tuple[Path, str, str]:
    """Run one update session (or reuse it if it already finished) and judge its outcome."""
    job = Path(jobs_dir) / job_name
    if not (read_json(job / "result.json") or {}).get("finished_at"):
        if job.exists():
            shutil.rmtree(job)
        run_update_task(task_dir, model=model, runner_workspace=runner_ws, jobs_dir=jobs_dir, job_name=job_name, env=env)
    kind, reason = session_outcome(job, base_ws)
    return job, kind, reason


def adopt(job: Path, next_gen: Path, prev_ws: Path) -> dict:
    """Install the session's edited workspace as the next generation, with a DIFF.md against the parent."""
    src = edited_workspace(job)
    ws = Path(next_gen) / "harness_workspace"
    ws.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, ws, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    diff = diff_md(prev_ws, ws)
    (Path(next_gen) / "DIFF.md").write_text(diff)
    return {"adopted_from": Path(job).name, "changed": diff != "(no file differences)"}


def update_with_repairs(*, line_dir: Path, iteration: int, prev_ws: Path, evidence_jobs: list[Path], budget: int,
                        model: str, env: str | None, max_input_tokens: int, repair_sessions: int,
                        carry: list[Path]) -> tuple[Path | None, list[dict], str]:
    """The update step of a first-stage iteration, with the repair pipeline.

    Returns ``(adoptable session job or None, attempts, stop reason)``."""
    tasks_dir = line_dir / "update_tasks"
    attempts: list[dict] = []

    def session(name: str, target_ws: Path, runner_ws: Path, evidence: list[Path], session_budget: int, repair: bool):
        task = tasks_dir / name
        if not task.exists():
            build_update_task(task, workspace=target_ws, evidence_jobs=evidence, budget=session_budget,
                              max_input_tokens=max_input_tokens, repair=repair)
        job, kind, reason = run_session(task_dir=task, model=model, runner_ws=runner_ws, base_ws=target_ws,
                                        jobs_dir=line_dir, job_name=name, env=env)
        attempts.append({"session": name, "runner": str(runner_ws.relative_to(line_dir)), "outcome": kind, "reason": reason})
        return job, kind, reason

    prefix = f"iteration{iteration:02d}-update"
    job, kind, reason = session(prefix, prev_ws, prev_ws, evidence_jobs + carry, budget, repair=False)
    if kind == "ok":
        return job, attempts, ""
    print(f"    update session failed: {reason}", flush=True)

    failed = [job]
    k, rerun_no = 0, 2
    while True:
        fixed = None
        while k < repair_sessions:
            k += 1
            # repair k runs on gen_{t-k}: first the parent itself, then one generation further back each time
            runner = line_dir / "lineage" / f"gen{max(iteration - k, 0):02d}" / "harness_workspace"
            job, kind, reason = session(f"{prefix}-repair{k}", prev_ws, runner, list(failed), REPAIR_BUDGET, repair=True)
            if kind == "ok":
                fixed = job
                break
            failed.append(job)
        if fixed is None:
            return None, attempts, f"all {repair_sessions} repair sessions failed (last: {reason})"
        failed.append(fixed)
        fixed_ws = line_dir / "lineage" / f"iteration{iteration:02d}-repaired-{k}" / "harness_workspace"
        if not fixed_ws.exists():
            fixed_ws.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(edited_workspace(fixed), fixed_ws, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        job, kind, reason = session(f"{prefix}{rerun_no}", fixed_ws, fixed_ws, evidence_jobs + carry, budget, repair=False)
        if kind == "ok":
            return job, attempts, ""
        if k >= repair_sessions:
            return None, attempts, f"the update failed again after the last repair: {reason}"
        failed.append(job)
        rerun_no += 1


def record(line_dir: Path, entry: dict) -> None:
    path = Path(line_dir) / "lineage" / "lineage.json"
    data = read_json(path) or {"iterations": []}
    data["iterations"] = [e for e in data["iterations"] if e.get("iteration") != entry.get("iteration")] + [entry]
    write_json(path, data)
