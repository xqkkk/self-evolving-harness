"""Scan evaluation runs for answer-key lookups and leaked answer content.

    python -m anti_cheat.scan JOB_OR_TRIAL [...] [--tasks-root DIR] [--out report.json]

Works on the trial directories of any Harbor job, for our kernel, Codex and Claude Code alike (see
``trajectories.py``). Two automatic steps flag candidates; a third, the review, decides:

1. Command patterns (``patterns.py``). Per benchmark, commands that fetch, clone or read answer material: a task's
   official solution or tests (Terminal-Bench), the upstream fix (git history, PR / commit / compare URLs, newer
   package releases of the repository under repair: SWE-Bench Pro, SWE-bench Verified, DeepSWE), the dataset or its
   mirrors, the GAIA2 oracle files or the sidecar, the tau2 task definitions and evaluation criteria, Claw-Eval's
   grader.py / task.yaml, APEX gold files, BrowseComp-Plus ciphertext; plus search-engine queries and any
   provider-side web search.
2. Answer content. Unique long fragments (at least 45 characters) of the task's own answer material (the gold patch,
   the GAIA2 oracle script, the expected answer, the tau3 task definition, the official solution) are searched for in
   what the commands printed. Three or more fragments in the output of one command mark a leak. Commands that write a
   file are skipped (an agent echoing what it wrote itself is normal). For the code benchmarks, whose gold patches
   overlap with the repository itself, only network fetches from code-hosting sites count; for the others, any read
   from the environment does.
3. Review. Every flagged trial is checked by hand (or by a reviewing agent) against its trajectory. A run confirmed to
   have obtained the answer this way scores 0; a candidate that turns out to be legitimate (for example, reading the
   task's own input document) keeps its score.

The report lists, per flagged trial, the family, task, reward, the matching commands with the head of their output,
and the content-leak verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .patterns import FAMILY_PATTERNS, FETCH_HINT, GENERIC
from .trajectories import steps

# Commands that read content back from the environment or the network (an agent's own `cat` of what it wrote is
# excluded through WRITE_FILE below).
READBACK = re.compile(r"\b(curl|wget|urllib|urlopen|requests\.get|httpx|git fetch|git clone|git ls-remote|git remote|"
                      r"git pull|git show|git log|cat|less|head|tail|sed|grep|rg|python[0-9]?|jq|find|ls|Read|Grep|"
                      r"WebFetch)\b", re.I)
NETWORK = re.compile(r"\b(curl|wget|urllib|urlopen|requests\.get|httpx|git fetch|git clone|git ls-remote|git remote|"
                     r"git pull|WebFetch)\b", re.I)
# Fetching an answer means reaching a code-hosting site; the repository under repair is full of http:// constants.
CODE_HOST = re.compile(r"https?://[^\s\"']*(github\.com|githubusercontent\.com|codeload\.github|gitlab\.com|"
                       r"bitbucket\.org|sourceforge\.net)", re.I)
# Only explicit file writes are excluded: fetching and saving to disk before reading is a common way to cheat.
WRITE_FILE = re.compile(r"\bcat\b[^|<]*?>\s*\S+|\btee\b\s+(-a\s+)?[^|<\s]+")
MIN_MARK = 45
MIN_HITS = 3
NETWORK_GATED = {"sbp", "swebv", "deepswe", "tb2.1"}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def family_of(cfg: dict) -> str:
    task = cfg.get("task") or {}
    s = f"{task.get('source', '')} {task.get('name', '')} {task.get('path', '')}".lower()
    for key, fam in (("apex-agents", "apex"), ("apex_agents", "apex"), ("terminal-bench", "tb2.1"),
                     ("deep-swe", "deepswe"), ("deepswe", "deepswe"), ("gaia2", "gaia2"), ("tau3", "tau3"),
                     ("swebench-verified", "swebv"), ("swebench_verified", "swebv"), ("swe-bench-pro", "sbp"),
                     ("instance_", "sbp"), ("browsecomp", "bcplus"), ("bcplus", "bcplus"), ("claw-eval", "claw"),
                     ("claw_eval", "claw"), ("officeqa", "officeqa")):
        if key in s:
            return fam
    return "other"


def reward_of(trial: Path):
    result = read_json(trial / "result.json") or {}
    return ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")


def task_dir(trial: Path, tasks_root: Path) -> Path | None:
    path = ((read_json(trial / "config.json") or {}).get("task") or {}).get("path") or ""
    if not path:
        return None
    for candidate in (Path(path), tasks_root / path, tasks_root / Path(path).name):
        if candidate.is_dir():
            return candidate
    return None


def _marks(text: str, prefix: str = "+") -> list[str]:
    out = []
    for line in text.splitlines():
        if prefix and not line.startswith(prefix):
            continue
        s = line[len(prefix):].strip() if prefix else line.strip()
        if len(s) >= MIN_MARK and not s.startswith(("#", "//")):
            out.append(s)
    return out[:60]


def gold_marks(trial: Path, family: str, tasks_root: Path, tb_tasks: list[Path]) -> list[str]:
    """Unique fragments of the task's answer material; empty when there is nothing to compare with."""
    d = task_dir(trial, tasks_root)
    if family in ("sbp", "swebv"):
        cfg = read_json(d / "tests" / "config.json") if d else None
        if not cfg:
            return []
        # SWE-Bench Pro: the gold patch; SWE-bench Verified: the gold patch and the test patch.
        return _marks((cfg.get("patch") or "") + ("\n" + (cfg.get("test_patch") or "") if family == "swebv" else ""))
    if family == "gaia2":
        scenario = read_json(d / "environment" / "scenario.json") if d else None
        return _marks(json.dumps(scenario.get("events"), ensure_ascii=False), prefix="") if scenario else []
    if family in ("officeqa", "tau3"):
        for name in ("tests/expected.json", "tests/config.json", "tests/task.json"):
            if d and (d / name).is_file():
                return _marks((d / name).read_text(errors="ignore"), prefix="")
        return []
    if family in ("tb2.1", "deepswe"):
        roots = [d] if d else []
        name = trial.name.rsplit("__", 1)[0]
        roots += [r / name for r in tb_tasks]
        for root in roots:
            text = "".join(f.read_text(errors="ignore") + "\n"
                           for f in sorted((root / "solution").rglob("*")) if f.is_file()) if root.is_dir() else ""
            if text.strip():
                return _marks(text, prefix="")
    return []


def pattern_hits(family: str, trial_steps) -> list[dict]:
    pats = [re.compile(p, re.I) for p in FAMILY_PATTERNS.get(family, []) + GENERIC]
    hits = []
    for step in trial_steps:
        if step.web_search:
            hits.append({"pattern": "web-search", "command": step.command[:600], "output_head": step.output[:600]})
            continue
        for pat in pats:
            if pat.search(step.command) and (FETCH_HINT.search(step.command) or family in ("gaia2", "tau3")):
                hits.append({"pattern": pat.pattern, "command": step.command[:600], "output_head": step.output[:600]})
                break
    return hits


def content_leak(family: str, marks: list[str], trial_steps) -> dict | None:
    gate = NETWORK if family in NETWORK_GATED else READBACK
    for step in trial_steps:
        if gate is NETWORK and not CODE_HOST.search(step.command):
            continue
        if WRITE_FILE.search(step.command) or not gate.search(step.command):
            continue
        n = sum(1 for m in marks if m in step.output)
        if n >= MIN_HITS:
            return {"command": step.command[:300], "fragments_found": n, "network": bool(NETWORK.search(step.command))}
    return None


def is_trial(d: Path) -> bool:
    return (d / "config.json").is_file() and ((d / "agent").is_dir() or (d / "trial.log").is_file())


def find_trials(root: Path) -> list[Path]:
    if is_trial(root):
        return [root]
    if root.name in ("lineage", "evolve_task") or not root.is_dir():
        return []  # evolution workspaces and update tasks hold copies of evidence, not evaluation runs
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            out += find_trials(child)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="job or trial directories")
    ap.add_argument("--tasks-root", type=Path, default=Path.cwd(), help="where task paths in config.json resolve")
    ap.add_argument("--tb-tasks", type=Path, action="append",
                    default=[Path.home() / ".cache" / "harbor" / "tasks" / "packages" / "terminal-bench"],
                    help="registry task cache holding Terminal-Bench solutions")
    ap.add_argument("--out", type=Path, default=Path("anti_cheat_report.json"))
    args = ap.parse_args()

    flagged, counts = [], {}
    for trial in (t for p in args.paths for t in find_trials(p)):
        family = family_of(read_json(trial / "config.json") or {})
        if family == "other":
            continue
        trial_steps = steps(trial)
        counts.setdefault(family, {"trials": 0, "flagged": 0, "no_answer_material": 0})
        counts[family]["trials"] += 1
        hits = pattern_hits(family, trial_steps)
        marks = gold_marks(trial, family, args.tasks_root, args.tb_tasks)
        if not marks:
            counts[family]["no_answer_material"] += 1
        leak = content_leak(family, marks, trial_steps) if marks else None
        if hits or leak:
            counts[family]["flagged"] += 1
            config = read_json(trial / "config.json") or {}
            flagged.append({"family": family, "trial": str(trial), "reward": reward_of(trial),
                            "task": (config.get("task") or {}).get("name") or Path((config.get("task") or {}).get("path", "")).name,
                            "pattern_hits": hits[:12], "n_pattern_hits": len(hits), "content_leak": leak})
    args.out.write_text(json.dumps({"counts": counts, "flagged": flagged}, indent=1, ensure_ascii=False))
    for fam, c in sorted(counts.items()):
        print(f"{fam:9s} {c['trials']:5d} trials  {c['flagged']:4d} flagged  "
              f"({c['no_answer_material']} without answer material for step 2)", file=sys.stderr)
    print(f"{len(flagged)} trials to review -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
