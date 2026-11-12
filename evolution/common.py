"""Paths, JSON helpers and subprocess plumbing shared by the loop."""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "harness_kernel"
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
AGENT = "harness_kernel.agent:KernelAgent"
WORKSPACE_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def trials(job: Path) -> list[Path]:
    """Trial directories of a Harbor job (each carries a config.json)."""
    job = Path(job)
    if not job.is_dir():
        return []
    return sorted(p for p in job.iterdir() if p.is_dir() and (p / "config.json").is_file())


def job_finished(job: Path) -> bool:
    """Harbor writes result.json when a job starts and sets ``finished_at`` only at the end."""
    return bool((read_json(Path(job) / "result.json") or {}).get("finished_at"))


def reward_of(trial: Path):
    result = read_json(Path(trial) / "result.json") or {}
    return ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")


def task_of(trial: Path) -> str:
    """The task name as it was passed to ``harbor run -i``: the directory name for local datasets,
    the registry name otherwise."""
    task = (read_json(Path(trial) / "config.json") or {}).get("task") or {}
    path = task.get("path")
    return Path(path).name if path else task.get("name", "")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load ``KEY=VALUE`` lines into the environment without overriding variables already set."""
    if not Path(path).is_file():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def run_logged(cmd: list[str], log: Path, env: dict | None = None) -> int:
    print(f"    $ {' '.join(str(c) for c in cmd[:8])} ...", flush=True)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        fh.write(f"\n# {datetime.now().isoformat(timespec='seconds')} {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                              cwd=ROOT).returncode


def copy_workspace(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=WORKSPACE_IGNORE)
