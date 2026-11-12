"""The schedule reproduces the paper's settings.

    python -m pytest tests/test_schedule.py
"""

from pathlib import Path

from evolution.schedule import cosine_budget, load_split, random_batches

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "configs" / "splits"


def test_cosine_budget_matches_the_paper():
    assert [cosine_budget(i, 5) for i in range(1, 6)] == [500, 441, 300, 159, 100]


def test_batches_use_every_training_task_once():
    for bench in ("tb21", "gaia2", "officeqa", "tau3", "sbp"):
        names = load_split(SPLITS / f"{bench}-train.txt")
        batches = random_batches(names, 12, 5, seed=0)
        flat = [t for b in batches for t in b]
        assert len(flat) == 60 and sorted(flat) == sorted(names)


def test_first_batch_is_stable():
    # The first Terminal-Bench 2.1 batch of every first-stage line in the paper.
    first = sorted(random_batches(load_split(SPLITS / "tb21-train.txt"), 12, 5, seed=0)[0])
    assert first == sorted([
        "terminal-bench/chess-best-move", "terminal-bench/code-from-image", "terminal-bench/dna-assembly",
        "terminal-bench/log-summary-date-ranges", "terminal-bench/mteb-retrieve", "terminal-bench/overfull-hbox",
        "terminal-bench/pytorch-model-recovery", "terminal-bench/raman-fitting", "terminal-bench/reshard-c4-data",
        "terminal-bench/sparql-university", "terminal-bench/train-fasttext", "terminal-bench/vulnerable-secret",
    ])
