"""Running Harbor jobs: one benchmark leg of a batch, or an update task.

Every job runs the same agent, ``harness_kernel.agent:KernelAgent``, with ``--ak workspace=<harness>``.
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .common import AGENT, ROOT, run_logged

HARBOR = shlex.split(os.environ.get("HARBOR_BIN", "harbor"))
AUTO_STOP_MINS = 120  # a Daytona sandbox left behind by a killed job stops (and is deleted) after 2 idle hours


@dataclass
class Leg:
    """One benchmark of a batch: its own Harbor job, because environment kwargs apply job-wide."""

    name: str
    dataset: str                       # registry name (``-d``) or a local task directory (``-p``)
    tasks: list[str]
    env_kwargs: dict = field(default_factory=dict)
    host_judge: str | None = None      # module run as ``python -m <module> <job>``, e.g. "judges.gaia2"
    attempts: int = 1


def _dataset_args(dataset: str) -> list[str]:
    local = (ROOT / dataset).is_dir() or Path(dataset).is_dir()
    return ["-p" if local else "-d", dataset]


def harbor_command(*, dataset: str, tasks: list[str], model: str, workspace: Path, jobs_dir: Path, job_name: str,
                   attempts: int = 1, concurrency: int | None = None, env: str | None = "daytona",
                   env_kwargs: dict | None = None) -> list[str]:
    cmd = [*HARBOR, "run", "-y", *_dataset_args(dataset),
           "-a", AGENT, "-m", model, "--ak", f"workspace={workspace}",
           "-k", str(attempts), "-n", str(concurrency or max(1, len(tasks) * attempts)),
           "-o", str(jobs_dir), "--job-name", job_name,
           # a burst of sandbox requests queues at the backend; give the queued builds room
           "--environment-build-timeout-multiplier", "3.0"]
    if env:
        cmd += ["--env", env]
    if env == "daytona":
        cmd += ["--ek", f"auto_stop_interval_mins={AUTO_STOP_MINS}"]
    for k, v in (env_kwargs or {}).items():
        # structured values (e.g. sandbox labels for the host relay) go as JSON: labels={"run": "sol-run1"}
        cmd += ["--ek", f"{k}={v if isinstance(v, str) else json.dumps(v)}"]
    for task in tasks:
        cmd += ["-i", task]
    return cmd


def run_leg(leg: Leg, *, model: str, workspace: Path, jobs_dir: Path, job_name: str, env: str | None,
            concurrency: int | None = None) -> Path:
    job = Path(jobs_dir) / job_name
    cmd = harbor_command(dataset=leg.dataset, tasks=leg.tasks, model=model, workspace=workspace, jobs_dir=jobs_dir,
                         job_name=job_name, attempts=leg.attempts, concurrency=concurrency, env=env,
                         env_kwargs=leg.env_kwargs)
    run_logged(cmd, Path(jobs_dir) / f"{job_name}.log")
    return job


def host_judge(leg: Leg, job: Path) -> None:
    """Score the trials that are graded on the host (GAIA2, Claw-Eval, APEX-Agents, BrowseComp-Plus).

    Must run before the trials become evidence: an unjudged trial would show the proposer a zero."""
    if leg.host_judge:
        import sys

        run_logged([sys.executable, "-m", leg.host_judge, str(job)], Path(job).parent / f"{Path(job).name}-judge.log")


def run_update_task(task_dir: Path, *, model: str, runner_workspace: Path, jobs_dir: Path, job_name: str,
                    env: str | None) -> Path:
    """Run one update session: the update task is an ordinary Harbor task for the same agent."""
    cmd = [*HARBOR, "run", "-y", "-p", str(task_dir), "-a", AGENT, "-m", model,
           "--ak", f"workspace={runner_workspace}", "-k", "1", "-n", "1",
           "-o", str(jobs_dir), "--job-name", job_name]
    if env:
        cmd += ["--env", env]
    if env == "daytona":
        cmd += ["--ek", f"auto_stop_interval_mins={AUTO_STOP_MINS}"]
    run_logged(cmd, Path(jobs_dir) / f"{job_name}.log")
    return Path(jobs_dir) / job_name
