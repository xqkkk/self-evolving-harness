"""First stage: multi-task pretraining (Iterations 1-5).

    python -m evolution.stage1 --config configs/stage1.json --model gpt-5.6-sol-high --name sol-run1

Each iteration draws 12 previously unused training tasks from each of the five benchmarks (60 tasks),
runs them with the current harness, and lets the same agent update the harness from the complete run
records, its notes, and the record of its previous update session. Everything for one line lands in
``runs/<name>/``::

    lineage/gen00/harness_workspace/        the seed
    lineage/gen01/harness_workspace/        after Iteration 1 (+ DIFF.md against gen00)
    lineage/lineage.json                    per-iteration record
    iteration01-eval-<benchmark>/           Harbor job: the batch's trials of one benchmark
    iteration01-update/                     Harbor job: the update session
    update_tasks/iteration01-update/        the assembled update task

Re-running with the same ``--name`` resumes: recorded iterations and finished jobs are skipped.
"""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import ROOT, copy_workspace, job_finished, load_dotenv, read_json, reward_of, trials, write_json
from .harbor import Leg, host_judge, run_leg
from .schedule import cosine_budget, load_split, random_batches
from .sessions import adopt, record, update_with_repairs


def run_batch(legs: list[Leg], *, model: str, workspace: Path, line_dir: Path, prefix: str, env: str | None,
              parallel: bool) -> list[Path]:
    """Run every leg of a batch, then judge on the host where a benchmark needs it."""

    def one(leg: Leg) -> Path:
        name = f"{prefix}-{leg.name}"
        job = line_dir / name
        if not job_finished(job):
            run_leg(leg, model=model, workspace=workspace, jobs_dir=line_dir, job_name=name, env=env)
        host_judge(leg, job)
        return job

    if parallel:
        with ThreadPoolExecutor(max_workers=len(legs)) as pool:
            return list(pool.map(one, legs))
    return [one(leg) for leg in legs]


def legs_from_config(benchmarks: list[dict], tasks: dict[str, list[str]]) -> list[Leg]:
    return [Leg(name=b["name"], dataset=b["dataset"], tasks=tasks[b["name"]], env_kwargs=b.get("env_kwargs") or {},
                host_judge=b.get("host_judge")) for b in benchmarks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "stage1.json")
    ap.add_argument("--model", required=True, help="model alias in the providers file")
    ap.add_argument("--name", required=True, help="line name; reuse it to resume")
    ap.add_argument("--seed-workspace", type=Path, default=ROOT / "harnesses" / "seed")
    ap.add_argument("--env", default="daytona", help="Harbor environment ('' = local Docker)")
    ap.add_argument("--sequential", action="store_true", help="run the benchmark legs of a batch one after another")
    args = ap.parse_args()

    load_dotenv()
    cfg = read_json(args.config)
    n, size, seed = cfg["iterations"], cfg["tasks_per_benchmark"], cfg["batch_seed"]
    line_dir = ROOT / "runs" / args.name
    gen0 = line_dir / "lineage" / "gen00" / "harness_workspace"
    if not gen0.exists():
        gen0.parent.mkdir(parents=True, exist_ok=True)
        copy_workspace(args.seed_workspace, gen0)
    write_json(line_dir / "config.json", {**cfg, "model": args.model})

    # Eq. 4: per benchmark, one seeded shuffle of the training split cut into n disjoint batches.
    batches = {b["name"]: random_batches(load_split(ROOT / b["train_split"]), size, n, seed) for b in cfg["benchmarks"]}
    done = {e["iteration"] for e in (read_json(line_dir / "lineage" / "lineage.json") or {}).get("iterations", [])}

    for it in range(1, n + 1):
        if it in done:
            print(f"[{it}/{n}] already recorded, skipped")
            continue
        prev_ws = line_dir / "lineage" / f"gen{it - 1:02d}" / "harness_workspace"
        legs = legs_from_config(cfg["benchmarks"], {name: b[it - 1] for name, b in batches.items()})
        print(f"[{it}/{n}] gen{it - 1:02d}: rollout on {sum(len(l.tasks) for l in legs)} training tasks", flush=True)
        jobs = run_batch(legs, model=args.model, workspace=prev_ws, line_dir=line_dir,
                         prefix=f"iteration{it:02d}-eval", env=args.env or None, parallel=not args.sequential)
        passed = sum(1 for j in jobs for t in trials(j) if reward_of(t) == 1.0)
        total = sum(len(trials(j)) for j in jobs)
        print(f"    passed {passed}/{total}", flush=True)

        budget = cosine_budget(it, n, cfg["budget"]["a_max"], cfg["budget"]["a_min"])
        carry = []
        if cfg.get("carry_previous_session") and it > 1:
            prev = (read_json(line_dir / "lineage" / "lineage.json") or {}).get("iterations", [])
            prev = next((e for e in prev if e["iteration"] == it - 1), None)
            if prev and (line_dir / prev["adopted_from"]).is_dir():
                carry = [line_dir / prev["adopted_from"]]
        job, attempts, stop = update_with_repairs(
            line_dir=line_dir, iteration=it, prev_ws=prev_ws, evidence_jobs=jobs, budget=budget, model=args.model,
            env=args.env or None, max_input_tokens=cfg["max_input_tokens"], repair_sessions=cfg["repair_sessions"],
            carry=carry)
        entry = {"iteration": it, "harness": f"gen{it - 1:02d}", "budget": budget, "passed": passed, "total": total,
                 "batch": {name: b[it - 1] for name, b in batches.items()}, "update_sessions": attempts}
        if job is None:
            record(line_dir, {**entry, "stopped": stop})
            print(f"    {stop}: stopping", flush=True)
            return 1
        gen_dir = line_dir / "lineage" / f"gen{it:02d}"
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        record(line_dir, {**entry, **adopt(job, gen_dir, prev_ws)})
        print(f"    -> gen{it:02d} adopted ({job.name})", flush=True)
    print(f"lineage: {line_dir / 'lineage'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
