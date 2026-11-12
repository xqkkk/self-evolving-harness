"""Batches, the budget of new lines, and success anchors.

* First-stage batches: every benchmark's training split is shuffled with ``random.Random(seed)`` and
  cut into consecutive batches, so each training task is used exactly once (Eq. 4).
* Budget of new lines (the learning rate): a cosine schedule from ``a_max`` to ``a_min`` over the
  first stage's iterations (Eq. 6); with a_max = 500, a_min = 100 and five iterations this gives
  500, 441, 300, 159, 100.
* Success anchors for the second stage: training tasks that the model already solved in the first
  stage, spread evenly across the first-stage benchmarks and across task lengths.
"""

import math
import random
from datetime import datetime
from pathlib import Path

from .common import read_json, reward_of, task_of, trials


def cosine_budget(iteration: int, iterations: int, a_max: int = 500, a_min: int = 100) -> int:
    """Budget of iteration ``iteration`` (1-based) out of ``iterations``."""
    if iterations <= 1:
        return a_max
    p = min(1.0, (iteration - 1) / (iterations - 1))
    return round(a_min + (a_max - a_min) * (1 + math.cos(math.pi * p)) / 2)


def load_split(path: Path) -> list[str]:
    """A split file holds one task name per line; order matters, since batches are cut after a seeded shuffle."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def random_batches(names: list[str], size: int, count: int, seed: int) -> list[list[str]]:
    """Shuffle with ``random.Random(seed)`` and cut into ``count`` consecutive batches of ``size``."""
    if size * count > len(names):
        raise ValueError(f"{len(names)} tasks cannot fill {count} batches of {size}")
    order = list(names)
    random.Random(seed).shuffle(order)
    return [order[i * size:(i + 1) * size] for i in range(count)]


def _duration(result: dict) -> float:
    a = datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))
    b = datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00"))
    return (b - a).total_seconds()


def success_pool(eval_jobs_by_benchmark: dict[str, list[Path]]) -> dict[str, list[dict]]:
    """Passed first-stage training trials per benchmark (reward 1, no exception), one per task.

    Jobs are read in round order, so a task solved more than once keeps its first occurrence.
    """
    pool: dict[str, list[dict]] = {}
    for bench, jobs in eval_jobs_by_benchmark.items():
        seen: dict[str, dict] = {}
        for job in jobs:
            for trial in trials(job):
                result = read_json(trial / "result.json") or {}
                if result.get("exception_info") or reward_of(trial) is None or reward_of(trial) < 1.0:
                    continue
                name = task_of(trial)
                if name and name not in seen:
                    seen[name] = {"task": name, "trial": str(trial), "duration_s": _duration(result)}
        pool[bench] = list(seen.values())
    return pool


def sample_success_anchors(pool: dict[str, list[dict]], per_benchmark_per_round: int = 4, rounds: int = 2,
                           seed: int = 1) -> dict[int, dict[str, list[str]]]:
    """Draw disjoint anchor sets for the second-stage rounds.

    Per benchmark (in the order of ``pool``): sort the passed tasks by trial duration and cut them into
    ``per_benchmark_per_round`` quantile bins; draw ``rounds`` tasks from each bin with one
    ``random.Random(seed)``, and deal the draws to the rounds in turn. Each round thus gets one task per
    bin and benchmark (4 x 5 = 20 anchors per round in the paper), and the rounds share no task.
    """
    rng = random.Random(seed)
    out: dict[int, dict[str, list[str]]] = {r: {} for r in range(1, rounds + 1)}
    q = per_benchmark_per_round
    for bench, items in pool.items():
        items = sorted(items, key=lambda x: x["duration_s"])
        n = len(items)
        if n < q * rounds:
            raise ValueError(f"{bench}: only {n} passed tasks, need at least {q * rounds}")
        picks: list[dict] = []
        for b in range(q):
            lo = round(b * n / q)
            hi = max(lo + rounds, round((b + 1) * n / q))
            picks += rng.sample(items[lo:hi], rounds)
        for i, p in enumerate(picks):
            out[i % rounds + 1].setdefault(bench, []).append(p["task"])
    return out
