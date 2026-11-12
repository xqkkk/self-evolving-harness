"""Score a finished GAIA2 job on the host with ARE's official judge.

    python -m judges.gaia2 <job-dir> [--force] [--workers 8]

In the task container, a sidecar MCP server exposes the ARE apps to the agent and exports an official
ARE trace after every tool call; the in-container verifier only collects that trace. This pass judges
each trace with ARE's ``GraphPerEventJudge``: the same oracle replay, turn segmentation and rubrics as
``are-benchmark judge``. Judging runs in a dedicated virtual environment pinned to the same ARE version
as the sidecar (created on first use, ``.gaia2-judge-venv/``), and the official GAIA2 filesystem snapshot
is downloaded once so that oracle replays can execute filesystem actions.

Reward (official semantics): 1.0 if every turn passes, else 0.0. The verdict lands in
``verifier/gaia2_official_result.json``. For training tasks the judge also writes
``verifier/oracle_debrief.json``: the pristine scenario script and, when exportable, the oracle-replay
world; it reaches the proposer with the run records (never the agent container).

Judge model (OpenAI-compatible chat endpoint): ``GAIA2_JUDGE_MODEL``, ``GAIA2_JUDGE_BASE_URL``,
``GAIA2_JUDGE_API_KEY``; ``GAIA2_JUDGE_REASONING_EFFORT`` (default ``max``), ``GAIA2_JUDGE_MAX_TOKENS``
(default 32768), ``GAIA2_JUDGE_NUM_VOTES`` (default 3). The paper used DeepSeek V4 Flash at max effort.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import ROOT, agent_ran, job_trials, judge_settings, load_dotenv, read_json, task_dir, write_reward

ARE_VERSION = "1.2.0"  # pinned to the sidecar image's version
JUDGE_VENV = ROOT / ".gaia2-judge-venv"
FILESYSTEM_DIR = Path(os.environ.get("GAIA2_FILESYSTEM_DIR", ROOT / "datasets" / ".gaia2-filesystem"))
WORKER = Path(__file__).resolve().parent / "gaia2_worker.py"


def ensure_judge_venv() -> Path:
    python = JUDGE_VENV / "bin" / "python"
    if not python.is_file():
        print(f"creating {JUDGE_VENV.name} (ARE {ARE_VERSION}) ...", flush=True)
        subprocess.run(["uv", "venv", str(JUDGE_VENV), "--python", "3.12"], check=True, capture_output=True)
        subprocess.run(["uv", "pip", "install", "-p", str(python), f"meta-agents-research-environments=={ARE_VERSION}"],
                       check=True, capture_output=True)
    if not (FILESYSTEM_DIR / "demo_filesystem").is_dir():
        print("downloading the official GAIA2 filesystem snapshot (once) ...", flush=True)
        subprocess.run([str(python), "-c", (
            "from huggingface_hub import snapshot_download\n"
            "snapshot_download(repo_id='meta-agents-research-environments/gaia2_filesystem', repo_type='dataset', "
            f"local_dir={str(FILESYSTEM_DIR)!r}, allow_patterns=['demo_filesystem/*', 'demo_filesystem/**'])")],
            check=True)
    return python


def judge_trial(python: Path, trial: Path, scenario: Path, settings: dict) -> tuple[float | None, str]:
    verifier = trial / "verifier"
    trace, out = verifier / "are_trace.json", verifier / "gaia2_official_result.json"
    if not trace.is_file():
        # no trace: the agent never made a single tool call
        out.write_text(json.dumps({"success": False, "rationale": "agent produced no ARE trace"}))
        return 0.0, "no trace"
    env = {**os.environ,
           "GAIA2_JUDGE_MODEL": settings["model"], "GAIA2_JUDGE_BASE_URL": settings["base_url"],
           "GAIA2_JUDGE_API_KEY": settings["api_key"], "GAIA2_JUDGE_REASONING_EFFORT": settings["effort"] or "",
           "DEMO_FS_PATH": str(FILESYSTEM_DIR / "demo_filesystem")}
    env.setdefault("GAIA2_JUDGE_MAX_TOKENS", "32768")
    proc = subprocess.run([str(python), str(WORKER), str(scenario), str(trace), str(out)],
                          env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return None, f"judge exited with {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"
    verdict = read_json(out) or {}
    if verdict.get("exception") is not None or verdict.get("rationale") == "Exception":
        # the judge itself raised: no verdict; keep the record aside, leave the trial unjudged
        out.replace(out.with_name("gaia2_official_result.judge_error.json"))
        return None, f"judge raised, not scored: {str(verdict.get('exception'))[:200]}"
    return (1.0 if verdict.get("success") is True else 0.0), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--force", action="store_true", help="re-judge trials that already have a verdict")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv()
    settings = judge_settings("GAIA2_JUDGE", effort="max")
    os.environ.setdefault("GAIA2_JUDGE_NUM_VOTES", "3")
    python = ensure_judge_venv()

    work = []
    for trial in job_trials(args.job):
        tdir = task_dir(trial)
        scenario = tdir / "environment" / "scenario.json" if tdir else None
        if scenario is None or not scenario.is_file():
            continue  # not a GAIA2 trial
        if not agent_ran(trial):
            continue  # nothing to judge
        if not args.force and (trial / "verifier" / "gaia2_official_result.json").is_file():
            continue
        work.append((trial, scenario))

    judged = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for trial, (reward, note) in zip([w[0] for w in work], pool.map(lambda w: judge_trial(python, w[0], w[1], settings), work)):
            if reward is None:
                failed += 1
                print(f"  {trial.name:<48} {note}", flush=True)
                continue
            write_reward(trial, reward)
            judged += 1
            print(f"  {trial.name:<48} reward={reward}" + (f" ({note})" if note else ""), flush=True)
    print(f"judged {judged}, not judged {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
