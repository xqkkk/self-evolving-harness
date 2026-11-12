"""Second stage: continual training on a new domain with success anchors (Iterations 6-7).

    python -m evolution.stage2 --config configs/stage2.json --model gpt-5.6-sol-high \
        --stage1 runs/sol-run1 --name sol-run1-stage2 [--anchor-dir configs/splits/stage2-anchors/gpt-5.6-sol]

Starting from Iteration 5 of a first-stage line, each iteration runs 40 new-domain training tasks
(different for the two iterations) together with 20 success anchors: first-stage training tasks the
model already solved, four per first-stage benchmark. The anchors are rerun with the current harness
and their records go to the proposer alongside the new-domain records, so that an edit for the new
domain has to keep serving the old tasks (replay against catastrophic forgetting). The budget of new
lines is raised back to 200 and then lowered to 100; an update session may take two hours, and an
iteration allows up to three update attempts. The second iteration's update task also carries the
first iteration's update session(s).

Anchors are sampled from the first-stage line's own passed training trials
(``schedule.sample_success_anchors``); ``--anchor-dir`` instead reads them from
``<dir>/iteration{06,07}-<benchmark>.txt`` (the samples used in the paper are provided there).
"""

import argparse
import shutil
import sys
from pathlib import Path

from .common import ROOT, copy_workspace, load_dotenv, read_json, write_json
from .schedule import load_split, sample_success_anchors, success_pool
from .sessions import adopt, record, run_session
from .stage1 import legs_from_config, run_batch
from .update_task import build_update_task


def anchors_for(cfg: dict, args, iterations: list[int]) -> dict[int, dict[str, list[str]]]:
    benches = [b["name"] for b in cfg["anchors"]["benchmarks"]]
    if args.anchor_dir:
        return {it: {b: load_split(args.anchor_dir / f"iteration{it:02d}-{b}.txt") for b in benches} for it in iterations}
    stage1_iterations = range(1, cfg["anchors"]["stage1_iterations"] + 1)
    jobs = {b: [args.stage1 / f"iteration{i:02d}-eval-{b}" for i in stage1_iterations] for b in benches}
    drawn = sample_success_anchors(success_pool(jobs), cfg["anchors"]["per_benchmark_per_round"], len(iterations),
                                   cfg["anchors"]["seed"])
    return {it: drawn[k + 1] for k, it in enumerate(iterations)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "stage2.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage1", type=Path, required=True, help="first-stage line directory (runs/<name>)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--anchor-dir", type=Path, default=None)
    ap.add_argument("--env", default="daytona")
    args = ap.parse_args()

    load_dotenv()
    cfg = read_json(args.config)
    iterations = cfg["iterations"]                      # [6, 7]
    line_dir = ROOT / "runs" / args.name
    start = line_dir / "lineage" / f"gen{iterations[0] - 1:02d}" / "harness_workspace"
    if not start.exists():
        start.parent.mkdir(parents=True, exist_ok=True)
        copy_workspace(args.stage1 / "lineage" / f"gen{iterations[0] - 1:02d}" / "harness_workspace", start)
    anchors = read_json(line_dir / "anchors.json")
    if anchors is None:
        anchors = {str(k): v for k, v in anchors_for(cfg, args, iterations).items()}
        write_json(line_dir / "anchors.json", anchors)
    write_json(line_dir / "config.json", {**cfg, "model": args.model, "stage1": str(args.stage1)})
    done = {e["iteration"] for e in (read_json(line_dir / "lineage" / "lineage.json") or {}).get("iterations", [])}
    new = cfg["new_domain"]
    previous_sessions: list[Path] = []

    for k, it in enumerate(iterations):
        sessions = sorted(line_dir.glob(f"iteration{it:02d}-update*"))
        if it in done:
            previous_sessions = [s for s in sessions if s.is_dir()]
            print(f"[iteration {it}] already recorded, skipped")
            continue
        ws = line_dir / "lineage" / f"gen{it - 1:02d}" / "harness_workspace"
        tasks = {new["name"]: load_split(ROOT / new["round_tasks"][k]), **anchors[str(it)]}
        legs = legs_from_config([new] + cfg["anchors"]["benchmarks"], tasks)
        print(f"[iteration {it}] {len(tasks[new['name']])} new-domain tasks + "
              f"{sum(len(v) for b, v in tasks.items() if b != new['name'])} success anchors", flush=True)
        jobs = run_batch(legs, model=args.model, workspace=ws, line_dir=line_dir, prefix=f"iteration{it:02d}-eval",
                         env=args.env or None, parallel=True)

        task = line_dir / "update_tasks" / f"iteration{it:02d}-update"
        if not task.exists():
            build_update_task(task, workspace=ws, evidence_jobs=jobs + previous_sessions, budget=cfg["budgets"][k],
                              max_input_tokens=cfg["max_input_tokens"], agent_timeout_sec=cfg["update_timeout_sec"])
        winner, attempts = None, []
        for a in range(1, cfg["update_attempts"] + 1):
            name = f"iteration{it:02d}-update" + ("" if a == 1 else f"-try{a}")
            job, kind, reason = run_session(task_dir=task, model=args.model, runner_ws=ws, base_ws=ws,
                                            jobs_dir=line_dir, job_name=name, env=args.env or None)
            attempts.append({"session": name, "outcome": kind, "reason": reason})
            print(f"    {name}: {kind} {reason}", flush=True)
            if kind == "ok":
                winner = job
                break
        entry = {"iteration": it, "harness": f"gen{it - 1:02d}", "budget": cfg["budgets"][k], "update_sessions": attempts}
        if winner is None:
            record(line_dir, {**entry, "stopped": "every update attempt failed"})
            return 1
        gen_dir = line_dir / "lineage" / f"gen{it:02d}"
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        record(line_dir, {**entry, **adopt(winner, gen_dir, ws)})
        previous_sessions = [line_dir / a["session"] for a in attempts if (line_dir / a["session"]).is_dir()]
        print(f"    -> gen{it:02d} adopted ({winner.name})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
