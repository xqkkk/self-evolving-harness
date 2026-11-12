"""Score a finished APEX-Agents job on the host with Archipelago's official grading.

    python -m judges.apex_agents <job-dir> [--force] [--workers 2]

The in-container verifier only collects the final environment snapshot and ``/app/answer.txt``. This pass
runs Archipelago's grading runner (a clone at ``datasets/.apex-src/archipelago``,
https://github.com/Mercor-Intelligence/archipelago, commit 6419e09) per trial:

* initial snapshot: the task's world zip, from the HF dataset ``mercor/apex-agents``;
* final snapshot: the trial's collected export;
* trajectory: the task prompt and the final answer (the rubric judges the output and the artifact diff);
* verifiers: the task's rubric, verbatim, in the official example's shape.

reward = the official ``final_score`` (rubric-weighted partial credit in [0, 1]); the full grading output
lands in ``verifier/apex_official_result.json``. A rubric item whose judge never answered makes the
official runner report 0; such a trial is left unjudged instead of scored, and a re-run picks it up.

Judge: ``APEX_JUDGE_MODEL`` is a LiteLLM model name (default ``openrouter/google/gemini-3.5-flash``, the
model used in the paper), with ``APEX_JUDGE_API_KEY`` (falls back to ``OPENROUTER_API_KEY``) and, for an
OpenAI-compatible endpoint, ``APEX_JUDGE_BASE_URL``. Reasoning effort high, up to 32,768 output tokens.
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import ROOT, family_trials, load_dotenv, read_json, task_dir, write_reward

SRC = ROOT / "datasets" / ".apex-src"
GRADING = SRC / "archipelago" / "grading"
EXAMPLE = SRC / "archipelago" / "examples" / "hugging_face_task"
JUDGE_MAX_TOKENS = 32768
JUDGE_REASONING_EFFORT = "high"
_STATE_SUFFIXES = {".json", ".ics", ".txt", ".md", ".csv", ".yaml", ".yml", ".log"}
_STATE_FILE_CAP, _STATE_TOTAL_CAP = 64 * 1024, 512 * 1024


def tar_gz_to_zip(tgz: Path, zip_path: Path) -> Path:
    with tarfile.open(tgz, "r:gz") as tar, zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member in tar.getmembers():
            if member.isfile() and (f := tar.extractfile(member)) is not None:
                zf.writestr(member.name, f.read())
    return zip_path


def distill_final_state(trial: Path, tgz: Path) -> None:
    """Write ``verifier/apex_final_state.json``: the evolution-readable part of the final snapshot.

    The snapshot itself is 250-330 MB, almost all of it the world's document corpus, identical on every
    run; it is left out of the update task. What carries signal stays: the full file manifest (an agent's
    deliverable is visible by name and size) and the apps' small text state files, verbatim."""
    manifest, state, total = [], {}, 0
    try:
        with tarfile.open(tgz, "r:gz") as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                manifest.append({"path": m.name, "size": m.size})
                if (m.name.startswith(".apps_data/") and Path(m.name).suffix.lower() in _STATE_SUFFIXES
                        and m.size <= _STATE_FILE_CAP and total + m.size <= _STATE_TOTAL_CAP
                        and (f := tar.extractfile(m)) is not None):
                    state[m.name] = f.read().decode("utf-8", errors="replace")
                    total += m.size
    except (tarfile.TarError, OSError) as e:
        (trial / "verifier" / "apex_final_state.json").write_text(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return
    (trial / "verifier" / "apex_final_state.json").write_text(json.dumps({
        "note": "distilled final snapshot: the full manifest and the apps' small text state files",
        "n_files": len(manifest), "total_bytes": sum(m["size"] for m in manifest), "manifest": manifest,
        "apps_data": state}, ensure_ascii=False, indent=1))


def world_zip(world_id: str) -> Path:
    local = SRC / "data" / "world_files_zipped" / f"{world_id}.zip"
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("mercor/apex-agents", f"world_files_zipped/{world_id}.zip", repo_type="dataset"))


_TASKS = None


def task_meta(task_id: str) -> dict:
    global _TASKS
    if _TASKS is None:
        _TASKS = {t["task_id"]: t for t in json.loads((SRC / "data" / "tasks_and_rubrics.json").read_text())}
    return _TASKS[task_id]


def trajectory(trial: Path, prompt: str) -> dict:
    from datetime import datetime

    answer_file = trial / "verifier" / "answer.txt"
    answer = answer_file.read_text(errors="ignore")[:60000] if answer_file.is_file() else ""
    elapsed = 0.0
    run = (read_json(trial / "result.json") or {}).get("agent_execution") or {}
    try:
        elapsed = (datetime.fromisoformat(run["finished_at"].replace("Z", "+00:00"))
                   - datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))).total_seconds()
    except (KeyError, TypeError, ValueError):
        pass
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer or "(no answer produced)"}],
            "output": {"final_answer": answer}, "status": "completed", "time_elapsed": elapsed}


def judge_trial(trial: Path, force: bool, judge: dict) -> str:
    out = trial / "verifier" / "apex_official_result.json"
    if out.is_file() and not force:
        return f"  {trial.name:<44} already judged"
    tdir = task_dir(trial)
    exp = read_json(tdir / "tests" / "expected.json") if tdir else None
    if not exp:
        return f"  {trial.name:<44} no task directory, skipped"
    meta = task_meta(exp["task_id"])
    final_tgz = trial / "verifier" / "final_snapshot.tar.gz"
    if not final_tgz.is_file():
        out.write_text(json.dumps({"error": "no final snapshot collected", "final_score": 0.0}))
        write_reward(trial, 0.0)
        return f"  {trial.name:<44} reward=0.0 (no final snapshot)"
    distill_final_state(trial, final_tgz)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        (tmpd / "trajectory.json").write_text(json.dumps(trajectory(trial, meta["prompt"]), ensure_ascii=False))
        (tmpd / "verifiers.json").write_text(json.dumps([
            {"verifier_id": c["verifier_id"], "verifier_version": 1, "world_id": meta["world_id"], "task_id": meta["task_id"],
             "eval_config_id": "ec_output_llm",
             "verifier_values": {"criteria": c["criteria"], "is_primary_objective": i == 0},
             "verifier_index": i, "verifier_dependencies": None}
            for i, c in enumerate(meta.get("rubric") or [])]))
        (tmpd / "grading_settings.json").write_text(json.dumps({
            "llm_judge_model": judge["model"],
            "llm_judge_extra_args": {"max_tokens": JUDGE_MAX_TOKENS, "reasoning_effort": JUDGE_REASONING_EFFORT,
                                     "allowed_openai_params": ["reasoning_effort"]}}))
        venv_python = GRADING / ".venv" / "bin" / "python"
        cmd = [str(venv_python) if venv_python.exists() else "python3", "-m", "runner.main",
               "--grading-run-id", f"gr_{trial.name[-8:]}", "--trajectory-id", f"traj_{trial.name[-8:]}",
               "--initial-snapshot", str(world_zip(meta["world_id"])),
               "--final-snapshot", str(tar_gz_to_zip(final_tgz, tmpd / "final.zip")),
               "--trajectory", str(tmpd / "trajectory.json"), "--grading-settings", str(tmpd / "grading_settings.json"),
               "--verifiers", str(tmpd / "verifiers.json"), "--eval-configs", str(EXAMPLE / "eval_configs.json"),
               "--scoring-config", str(EXAMPLE / "scoring_config.json"), "--output", str(tmpd / "grades.json")]
        env = {**os.environ, "OPENROUTER_API_KEY": judge["api_key"], "OPENAI_API_KEY": judge["api_key"]}
        if judge.get("base_url"):
            env.update(OPENAI_API_BASE=judge["base_url"], OPENAI_BASE_URL=judge["base_url"])
        proc = subprocess.run(cmd, cwd=GRADING, env=env, capture_output=True, text=True, timeout=1800)
        grades = read_json(tmpd / "grades.json")
    if grades is None:
        (trial / "verifier" / "apex_grading_stderr.log").write_text((proc.stdout or "")[-4000:] + "\n----\n" + (proc.stderr or "")[-8000:])
        return f"  {trial.name:<44} grading failed (rc={proc.returncode}, stderr saved)"
    errs = [v for v in grades.get("verifier_results") or []
            if v.get("status") == "error" or (v.get("verifier_result_values") or {}).get("verifier_crashed")]
    if grades.get("grading_run_status") == "error" or errs:
        (trial / "verifier" / "apex_grading_error.json").write_text(json.dumps(grades, ensure_ascii=False, indent=1))
        return f"  {trial.name:<44} grading did not finish (not scored)"
    reward = float((grades.get("scoring_results") or {}).get("final_score") or 0.0)
    out.write_text(json.dumps(grades, ensure_ascii=False, indent=1))
    (trial / "verifier" / "apex_grading_error.json").unlink(missing_ok=True)
    write_reward(trial, reward)
    return f"  {trial.name:<44} reward={reward}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    load_dotenv()
    judge = {"model": os.environ.get("APEX_JUDGE_MODEL", "openrouter/google/gemini-3.5-flash"),
             "api_key": os.environ.get("APEX_JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
             "base_url": os.environ.get("APEX_JUDGE_BASE_URL")}
    if not judge["api_key"]:
        raise SystemExit("set APEX_JUDGE_API_KEY or OPENROUTER_API_KEY")
    if not GRADING.is_dir():
        raise SystemExit(f"clone https://github.com/Mercor-Intelligence/archipelago into {GRADING.parent}")
    subprocess.run(["uv", "sync"], cwd=GRADING, check=True, capture_output=True)
    trials = family_trials(args.job, "apex")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(lambda t: judge_trial(t, args.force, judge), trials):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
