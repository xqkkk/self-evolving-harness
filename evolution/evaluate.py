"""Held-out evaluation of a frozen harness.

    python -m evolution.evaluate --config configs/evaluation.json --model gpt-5.6-sol-high \
        --workspace harnesses/gpt-5.6-sol/run1-iteration5 --name sol-run1-it5 [--only tb21,gaia2]

Runs every benchmark of the configuration on its held-out split (benchmarks with few held-out tasks
are run several times: Terminal-Bench 2.1 three times, OfficeQA Pro and tau3-Bench banking twice),
judges on the host where needed, and prints the score of each
benchmark: the success rate in percent, or the mean official score x 100 for benchmarks with partial
credit (APEX-Agents, Claw-Eval). Results land in ``runs/eval/<name>/``.
"""

import argparse
import sys
from pathlib import Path

from .common import ROOT, load_dotenv, read_json, reward_of, trials, write_json
from .harbor import Leg
from .schedule import load_split
from .stage1 import run_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "evaluation.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--only", default=None, help="comma-separated benchmark names")
    ap.add_argument("--env", default="daytona")
    args = ap.parse_args()

    load_dotenv()
    cfg = read_json(args.config)
    only = set(args.only.split(",")) if args.only else None
    benches = [b for group in ("in_distribution", "out_of_distribution") for b in cfg.get(group, [])
               if only is None or b["name"] in only]
    legs = [Leg(name=b["name"], dataset=b["dataset"], tasks=load_split(ROOT / b["split"]), attempts=b.get("attempts", 1),
                env_kwargs=b.get("env_kwargs") or {}, host_judge=b.get("host_judge")) for b in benches]
    out_dir = ROOT / "runs" / "eval" / args.name
    jobs = run_batch(legs, model=args.model, workspace=args.workspace.resolve(), line_dir=out_dir, prefix="heldout",
                     env=args.env or None, parallel=True)
    scores = {}
    for leg, job in zip(legs, jobs):
        rewards = [float(reward_of(t) or 0.0) for t in trials(job)]
        scores[leg.name] = {"trials": len(rewards), "score": round(100 * sum(rewards) / len(rewards), 2) if rewards else None}
        print(f"{leg.name:10s} {scores[leg.name]['score']}  ({len(rewards)} trials)")
    write_json(out_dir / "scores.json", {"model": args.model, "workspace": str(args.workspace), "scores": scores})
    return 0


if __name__ == "__main__":
    sys.exit(main())
