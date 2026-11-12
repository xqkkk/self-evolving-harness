"""Score a finished BrowseComp-Plus job on the host with the official grader prompt.

    python -m judges.browsecomp_plus <job-dir> [--force] [--workers 8]

The in-container verifier only collects ``/app/answer.txt`` (the sandbox has no network and no LLM).
This pass grades each answer with the official BrowseComp grader prompt (texttron/BrowseComp-Plus,
``search_agent/prompts.py``, GRADER_TEMPLATE) and writes the reward back. The question and the gold
answer are stored encrypted in the task's ``tests/expected.json`` (decrypted here only, as in the
official evaluation script). The full grader output, gold answer and reasoning included, is stored in
``verifier/bcplus_official_result.json``.

Judge (OpenAI-compatible chat endpoint): ``BCPLUS_JUDGE_MODEL``, ``BCPLUS_JUDGE_BASE_URL``,
``BCPLUS_JUDGE_API_KEY``, ``BCPLUS_JUDGE_REASONING_EFFORT`` (default max). The paper used DeepSeek V4
Flash at max effort.
"""

import argparse
import base64
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import family_trials, judge_settings, load_dotenv, read_json, task_dir, write_reward

# Official grader prompt, verbatim (texttron/BrowseComp-Plus, search_agent/prompts.py).
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def decrypt(ciphertext_b64: str, password: str) -> str:
    """The benchmark's own XOR decryption with a SHA-256-derived key."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = hashlib.sha256(password.encode("utf-8")).digest()
    key = key * (len(encrypted) // len(key)) + key[: len(encrypted) % len(key)]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def grade(client, settings: dict, question: str, answer: str, response: str):
    prompt = GRADER_TEMPLATE.format(question=question, response=response, correct_answer=answer)
    last = None
    for _ in range(3):
        try:
            extra = {"reasoning_effort": settings["effort"]} if settings["effort"] else {}
            reply = client.chat.completions.create(model=settings["model"], messages=[{"role": "user", "content": prompt}],
                                                   max_tokens=32768, extra_body=extra)
            text = reply.choices[0].message.content or ""
            m = re.search(r"correct:\s*\**\s*(yes|no)", text, re.IGNORECASE)
            ex = re.search(r"extracted_final_answer:\s*\**\s*(.*?)(?=\n|$)", text, re.IGNORECASE)
            if m:
                ok = m.group(1).lower() == "yes"
                return (1.0 if ok else 0.0), {"correct": ok, "extracted_final_answer": ex.group(1).strip() if ex else None,
                                              "rationale": text, "parse_error": False}
            last = f"grader output had no correct: verdict ({text[-200:]!r})"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    return None, {"parse_error": True, "error": last}


def judge_trial(client, settings: dict, trial: Path, force: bool) -> str:
    out = trial / "verifier" / "bcplus_official_result.json"
    if out.is_file() and not force:
        return f"  {trial.name:<44} already judged"
    tdir = task_dir(trial)
    exp = read_json(tdir / "tests" / "expected.json") if tdir else None
    if not exp:
        return f"  {trial.name:<44} no task directory, skipped"
    question, answer = decrypt(exp["query_enc"], exp["canary"]), decrypt(exp["answer_enc"], exp["canary"])
    answer_file = trial / "verifier" / "answer.txt"
    if not answer_file.is_file():
        reward, verdict = 0.0, {"correct": False, "extracted_final_answer": None, "parse_error": False, "note": "no answer file"}
    else:
        reward, verdict = grade(client, settings, question, answer, answer_file.read_text()[:20000])
        if reward is None:
            return f"  {trial.name:<44} grading failed: {verdict.get('error')}"
    out.write_text(json.dumps({**verdict, "reward": reward, "correct_answer": answer, "judge_model": settings["model"]},
                              ensure_ascii=False, indent=1))
    write_reward(trial, reward)
    return f"  {trial.name:<44} reward={reward}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv()
    settings = judge_settings("BCPLUS_JUDGE", effort="max")
    from openai import OpenAI

    client = OpenAI(base_url=settings["base_url"], api_key=settings["api_key"], timeout=600)
    trials = family_trials(args.job, "browsecomp")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(lambda t: judge_trial(client, settings, t, args.force), trials):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
