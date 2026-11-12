"""SWE-Bench Pro: close the git-history answer-key leak of the Harbor registry tasks.

The registry tasks (``scale-ai/swe-bench-pro``) prepare the repository with ``git reset --hard <base_commit>``. That
rewinds the working tree but keeps ``.git``, so every later commit, including the one that fixes the task, is still
reachable: ``git log --all -S<symbol>`` or ``git show <fix>`` prints the official patch and its tests.

The verifier needs one object from that same history: its ``test.sh`` runs the last line of ``before_repo_set_cmd``,
``git checkout <fix_commit> -- <test files>``, to restore the gold test files. Deleting ``.git`` alone therefore makes
every trial score 0. This builder rewrites each task in two steps:

* Dockerfile: after the upstream reset, delete ``.git`` and create a one-commit repository at the base tree, so ``git``
  still works for the agent's own diffs but nothing before or after the base exists.
* Gold tests: at build time, fetch every file named by the checkout line (plus test fixtures that some records carry
  in ``patch`` instead of ``test_patch``) from raw.githubusercontent.com at the fix commit, store them under the
  task's ``tests/gold/``, and replace the checkout line with ``cp -rf /tests/gold/. ./``. Harbor uploads ``tests/``
  only after the agent has finished, so the gold files are never visible to the agent.

Networking stays as upstream configured it (the tasks allow internet access, as SWE-Bench Pro is usually evaluated).
Fetching the upstream fix over the network counts as cheating and is caught by ``anti_cheat``.

    uv run harbor datasets download scale-ai/swe-bench-pro -o datasets       # the registry tasks
    uv run python -m adapters.swe_bench_pro                                  # -> datasets/swe-bench-pro-nogit
    uv run python -m adapters.swe_bench_pro --check                          # audit an existing output tree
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adapters.common import DATASETS, source_dir

_SCRUB = """
# --- leak fix (adapters/swe_bench_pro.py): drop all git history so no future
# commit, including the fix, is reachable via git show/log/diff ---
RUN rm -rf /app/.git \\
 && cd /app \\
 && git init -q \\
 && git config user.email b@b \\
 && git config user.name b \\
 && git config commit.gpgsign false \\
 && git add -A \\
 && git commit -q -m base --no-verify
"""
_GOLD_RESTORE = "cp -rf /tests/gold/. ./"
_BASE_RESET = re.compile(r"git reset --hard [0-9a-f]{7,40}")
_CHECKOUT = re.compile(r"^git checkout ([0-9a-f]{40}) -- (.+)$")


def scrub_dockerfile(text: str) -> str:
    return text if "leak fix" in text else text.rstrip() + "\n" + _SCRUB


def gold_spec(cfg: dict) -> tuple[str, str, list[str]]:
    """(repo, fix_commit, files) named by the record's gold-test checkout."""
    last = cfg["before_repo_set_cmd"].strip().split("\n")[-1].strip()
    m = _CHECKOUT.match(last)
    if not m:
        raise ValueError(f"unrecognized gold checkout line: {last!r}")
    files = m.group(2).split()
    # Some records put newly required test fixtures in `patch` instead of `test_patch`. They belong with the
    # verifier's gold tests, never in the agent's environment; production source is excluded strictly.
    for patch in (cfg.get("patch") or "", cfg.get("test_patch") or ""):
        names = re.findall(r"^\+\+\+ b/(.+)$", patch, re.M) + re.findall(r"^rename to (.+)$", patch, re.M)
        for name in names:
            path = Path(name)
            if not ({"testdata", "fixtures"} & set(path.parts)):
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe fixture path: {name}")
            files.append(name)
    return cfg["repo"], m.group(1), list(dict.fromkeys(files))


def fetch_gold_file(cache: Path, repo: str, commit: str, path: str) -> Path:
    cached = cache / repo / commit / path
    if cached.is_file():
        return cached
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
            return cached
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"fetch failed: {url}: {last}")


def rewrite_config(cfg: dict) -> dict:
    lines = cfg["before_repo_set_cmd"].strip().split("\n")
    lines[-1] = _GOLD_RESTORE
    return {**cfg, "before_repo_set_cmd": "\n".join(lines)}


def convert_task(task: Path, dst: Path, cache: Path) -> str | None:
    cfg = json.loads((task / "tests" / "config.json").read_text())
    try:
        repo, commit, files = gold_spec(cfg)
        cached = [fetch_gold_file(cache, repo, commit, f) for f in files]
    except Exception as e:  # noqa: BLE001 - reported per task
        return f"{task.name}: {e}"
    dest = dst / task.name
    shutil.copytree(task, dest)
    dockerfile = dest / "environment" / "Dockerfile"
    dockerfile.write_text(scrub_dockerfile(dockerfile.read_text()))
    (dest / "tests" / "config.json").write_text(json.dumps(rewrite_config(cfg), indent=2))
    for rel, src in zip(files, cached):
        gold = dest / "tests" / "gold" / rel
        gold.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, gold)
    return None


def check(src: Path, dst: Path) -> int:
    bad = 0
    tasks = [t for t in sorted(dst.iterdir()) if (t / "task.toml").is_file()]
    for task in tasks:
        problems = []
        cfg = json.loads((task / "tests" / "config.json").read_text())
        if cfg["before_repo_set_cmd"].strip().split("\n")[-1].strip() != _GOLD_RESTORE:
            problems.append("gold checkout not replaced")
        _, _, files = gold_spec(json.loads((src / task.name / "tests" / "config.json").read_text()))
        missing = [f for f in files if not (task / "tests" / "gold" / f).is_file()]
        if missing:
            problems.append(f"missing gold files: {missing}")
        if "leak fix" not in (task / "environment" / "Dockerfile").read_text():
            problems.append("Dockerfile not scrubbed")
        if problems:
            bad += 1
            print(f"  {task.name}: {'; '.join(problems)}")
    print(f"{len(tasks)} tasks, {bad} with problems")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DATASETS / "swe-bench-pro", help="the registry tasks")
    ap.add_argument("--out", type=Path, default=DATASETS / "swe-bench-pro-nogit")
    ap.add_argument("--cache", type=Path, default=source_dir("sbp-gold-cache"), help="cache of fetched gold files")
    ap.add_argument("--check", action="store_true", help="audit the output tree, write nothing")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"{args.src} missing: uv run harbor datasets download scale-ai/swe-bench-pro -o datasets")
    if args.check:
        return check(args.src, args.out)
    tasks = [t for t in sorted(args.src.iterdir()) if (t / "environment" / "Dockerfile").is_file()]
    no_reset = [t.name for t in tasks if not _BASE_RESET.search((t / "environment" / "Dockerfile").read_text())]
    if no_reset:
        raise SystemExit(f"no base reset in: {no_reset[:5]}")
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        errors = [e for e in pool.map(lambda t: convert_task(t, args.out, args.cache), tasks) if e]
    if errors:
        print("\n".join(f"  {e}" for e in errors))
        raise SystemExit(f"{len(errors)} tasks failed; the output is incomplete")
    print(f"wrote {len(tasks)} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
