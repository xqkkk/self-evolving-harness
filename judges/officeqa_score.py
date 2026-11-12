"""OfficeQA Pro verifier: the official ``reward.py`` at the official 0.0% tolerance.

Runs inside the task container as the task's verifier:

    python3 officeqa_score.py <answer.txt> <expected.json> <out_dir>

``reward.py`` is the benchmark's official scorer (https://github.com/databricks/officeqa); place it next
to this file (it is not redistributed here). Writes ``<out_dir>/reward.txt`` (1 or 0) and
``<out_dir>/officeqa_result.json`` with the extracted answer, the scorer's rationale, and the verdicts at
looser tolerances (0.1% / 1% / 5%) for analysis only; the reward is the 0.0% verdict.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reward import (  # noqa: E402  (official, verbatim)
    _is_direct_answer_only,
    extract_final_answer_from_xml,
    fuzzy_match_answer,
    score_answer,
)

OFFICIAL_TOLERANCE = 0.0
EXTRA_TOLERANCES = (0.001, 0.01, 0.05)


def main() -> int:
    answer_path, expected_path, out_dir = (Path(p) for p in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = json.loads(expected_path.read_text())["answer"]
    raw = answer_path.read_text(errors="replace") if answer_path.is_file() else ""
    reward = score_answer(expected, raw, tolerance=OFFICIAL_TOLERANCE)

    extracted, _ = extract_final_answer_from_xml(raw)
    try:
        direct_ok, direct_why = _is_direct_answer_only(expected, extracted)
    except Exception as e:  # empty answers raise inside the scorer
        direct_ok, direct_why = False, f"{type(e).__name__}: {e}"
    try:
        match_ok, match_why = fuzzy_match_answer(expected, extracted, OFFICIAL_TOLERANCE)
    except Exception as e:
        match_ok, match_why = False, f"{type(e).__name__}: {e}"

    (out_dir / "officeqa_result.json").write_text(json.dumps({
        "reward": reward, "tolerance": OFFICIAL_TOLERANCE, "expected": expected,
        "answer_file_present": answer_path.is_file(), "final_answer_tag_present": extracted != raw,
        "predicted": extracted[:500], "direct_answer_gate": {"ok": direct_ok, "rationale": direct_why},
        "match": {"ok": match_ok, "rationale": match_why},
        "reward_at_tolerance": {str(t): score_answer(expected, raw, tolerance=t) for t in EXTRA_TOLERANCES},
    }, indent=2, ensure_ascii=False))
    (out_dir / "reward.txt").write_text(f"{int(reward)}\n")
    print(f"expected={expected!r} predicted={extracted[:200]!r} reward={reward}")
    print(f"gate: {direct_why} | match: {match_why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
