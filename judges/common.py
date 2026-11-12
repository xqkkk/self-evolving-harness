"""Helpers shared by the host-side judges."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER = "https://openrouter.ai/api/v1"


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def job_trials(job: Path) -> list[Path]:
    return sorted(p for p in Path(job).iterdir() if p.is_dir() and not p.name.startswith(".")
                  and (p / "result.json").is_file())


def family_trials(job: Path, marker: str) -> list[Path]:
    """Trials of one benchmark: those whose local task path contains ``marker`` and that have an
    ``tests/expected.json``."""
    out = []
    for t in job_trials(job):
        tdir = task_dir(t)
        if tdir is not None and marker in str(tdir) and (tdir / "tests" / "expected.json").is_file():
            out.append(t)
    return out


def task_dir(trial: Path) -> Path | None:
    """The task directory a trial ran (local datasets only)."""
    for src in (trial / "config.json", trial / "result.json"):
        cfg = read_json(src) or {}
        task = cfg.get("task") or (cfg.get("config") or {}).get("task") or {}
        if task.get("path"):
            p = Path(task["path"])
            return p if p.is_absolute() else ROOT / p
    return None


def agent_ran(trial: Path) -> bool:
    """Whether the agent left any run record: a trial whose sandbox never came up has nothing to judge."""
    return (trial / "agent").is_dir() and any((trial / "agent").iterdir())


def write_reward(trial: Path, reward: float) -> None:
    (trial / "verifier").mkdir(exist_ok=True)
    (trial / "verifier" / "reward.txt").write_text(f"{reward}\n")
    result_path = trial / "result.json"
    result = read_json(result_path)
    if result is None:
        return
    verifier = result.setdefault("verifier_result", {}) or {}
    rewards = verifier.setdefault("rewards", {}) or {}
    rewards["reward"] = reward
    verifier["rewards"] = rewards
    result["verifier_result"] = verifier
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def judge_settings(prefix: str, *, model: str | None = None, base_url: str | None = None,
                   effort: str | None = None) -> dict:
    """Judge model settings from ``<prefix>_MODEL``, ``_BASE_URL``, ``_API_KEY`` and ``_REASONING_EFFORT``.

    The API key falls back to ``OPENROUTER_API_KEY`` when the endpoint is OpenRouter."""
    settings = {
        "model": os.environ.get(f"{prefix}_MODEL", model),
        "base_url": os.environ.get(f"{prefix}_BASE_URL", base_url or OPENROUTER),
        "api_key": os.environ.get(f"{prefix}_API_KEY"),
        "effort": os.environ.get(f"{prefix}_REASONING_EFFORT", effort or "") or None,
    }
    if not settings["api_key"] and settings["base_url"].startswith(OPENROUTER):
        settings["api_key"] = os.environ.get("OPENROUTER_API_KEY")
    missing = [k for k in ("model", "api_key") if not settings[k]]
    if missing:
        raise SystemExit(f"set {', '.join(f'{prefix}_{k.upper()}' for k in missing)} (see .env.example)")
    return settings
