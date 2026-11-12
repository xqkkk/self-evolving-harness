"""Score a finished Claw-Eval job on the host with the official graders.

    python -m judges.claw_eval <job-dir> [--force] [--workers 3]

The in-container verifier only collects evidence: the environment snapshot (including the recordings of
rendered deliverables), the per-service audit ledgers and the dispatch log of the mock services. The
score comes from the task's own ``tasks/<id>/grader.py`` in a pristine clone of the benchmark
(``datasets/.claw-src``, https://github.com/claw-eval/claw-eval, commit 5680b8b), run here with the trace
rebuilt from the kernel trajectory:

* messages: the task prompt plus every assistant text turn (the graders read the last assistant text as
  the answer); for multi-turn tasks, the conversation with the simulated user from its own log;
* dispatches: the mock services' request log, mapped back to tool names through ``tool_endpoints``;
* audit data and the environment snapshot, plus a ``local_file:<path>`` entry for each of the task's
  ``local_grader_files`` (reference images), as the benchmark's own ``cli.py`` adds them.

reward = the official ``task_score`` = safety x (0.80 completion + 0.20 robustness), in [0, 1].

Judge models, as in the benchmark's own configuration (no effort set, so each runs at its default):
general and multimodal tasks ``CLAW_JUDGE_*`` (default ``google/gemini-3.5-flash`` on OpenRouter, effort
high); multi-turn tasks ``CLAW_MULTITURN_JUDGE_*`` (default ``anthropic/claude-opus-4.6`` on OpenRouter).
"""

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import ROOT, family_trials, judge_settings, load_dotenv, read_json, task_dir, write_reward

SRC = ROOT / "datasets" / ".claw-src"


class _Client:
    """Presents ``.chat.completions.create`` to the benchmark's LLMJudge, adding the reasoning effort and
    retrying transient errors. An error that names a field is a real parameter error and is re-raised."""

    def __init__(self, client, effort: str | None, tries: int = 6):
        self._client, self._effort, self._tries = client, effort, tries

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kw):
        if self._effort:
            kw["extra_body"] = {**(kw.get("extra_body") or {}), "reasoning": {"effort": self._effort}}
        last = None
        for attempt in range(self._tries):
            try:
                return self._client.chat.completions.create(**kw)
            except Exception as exc:  # noqa: BLE001
                if "not permitted" in str(exc) or "Unsupported" in str(exc):
                    raise
                last = exc
                time.sleep(min(2 ** attempt, 20))
        raise last


def make_judge(settings: dict):
    """The benchmark's LLMJudge with an OpenAI-compatible client; images go as ``image_url`` data URIs."""
    import httpx
    from openai import OpenAI
    from claw_eval.graders.llm_judge import LLMJudge

    judge = LLMJudge.__new__(LLMJudge)
    client = OpenAI(base_url=settings["base_url"], api_key=settings["api_key"], http_client=httpx.Client(timeout=300.0))
    judge.client = _Client(client, settings["effort"])
    judge.model_id = settings["model"]
    judge._use_gemini = False  # both text and visual paths through chat.completions
    judge._call_log = []
    return judge


def build_messages(trial: Path, prompt: str):
    """user(prompt) + the conversation with the simulated user (multi-turn) + one assistant message per
    model turn that produced text."""
    from claw_eval.models.content import TextBlock
    from claw_eval.models.message import Message
    from claw_eval.models.trace import TokenUsage, TraceMessage

    def wrap(role, text, usage=None):
        return TraceMessage(trace_id=trial.name, message=Message(role=role, content=[TextBlock(text=text)]),
                            usage=usage or TokenUsage())

    out = [wrap("user", prompt)]
    ua_log = trial / "verifier" / "user_agent.jsonl"
    if ua_log.is_file():
        for line in ua_log.open():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("assistant"):
                out.append(wrap("assistant", ev["assistant"]))
            if ev.get("user"):
                out.append(wrap("user", ev["user"]))
    traj = trial / "agent" / "kernel_trajectory.jsonl"
    if not traj.is_file():
        return out
    for line in traj.open():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "model_call":
            continue
        resp = ev.get("response") or {}
        text = resp.get("text") if isinstance(resp.get("text"), str) else ""
        u = resp.get("usage") or {}
        if text.strip():
            out.append(wrap("assistant", text, TokenUsage(
                input_tokens=u.get("input_tokens") or u.get("prompt_tokens") or 0,
                output_tokens=u.get("output_tokens") or u.get("completion_tokens") or 0)))
    return out


def build_dispatches(trial: Path, spec: dict):
    """Mock-service request log -> ToolDispatch, with tool names from the task's ``tool_endpoints``."""
    from claw_eval.models.trace import ToolDispatch

    by_key = {}
    for ep in spec.get("tool_endpoints") or []:
        m = re.match(r"https?://[^/]*?:(\d+)(/.*)$", ep["url"])
        if m:
            by_key[(int(m.group(1)), m.group(2))] = (ep["tool_name"], ep["url"])
    out = []
    log = trial / "verifier" / "dispatches.jsonl"
    if not log.is_file():
        return out
    for i, line in enumerate(log.open()):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (rec.get("port", 0), rec.get("path", ""))
        name, url = by_key.get(key, (f"{rec.get('service', '?')}{rec.get('path', '')}",
                                     f"http://localhost:{rec.get('port', 0)}{rec.get('path', '')}"))
        out.append(ToolDispatch(
            trace_id=trial.name, tool_use_id=f"call_{i}", tool_name=name, endpoint_url=url,
            request_body=rec.get("request_body") if isinstance(rec.get("request_body"), dict) else {},
            response_status=int(rec.get("response_status", 200)), response_body=rec.get("response_body"),
            latency_ms=float(rec.get("latency_ms", 0.0))))
    return out


def with_reference_files(snapshot: dict, spec: dict, src_task: Path) -> dict:
    """Add the task's reference files the way the benchmark's cli.py does (``local_file:<path>``)."""
    snap = dict(snapshot)
    for rel in spec.get("local_grader_files") or []:
        p = src_task / rel
        snap[f"local_file:{rel}"] = ({"encoding": "base64", "content": base64.b64encode(p.read_bytes()).decode()}
                                     if p.exists() else {"error": f"not found: {p}"})
    return snap


def judge_trial(trial: Path, force: bool, judges: dict) -> str:
    out_path = trial / "verifier" / "claw_official_result.json"
    if out_path.is_file() and not force:
        return f"  {trial.name:<44} already judged"
    # A run that never started leaves no trajectory, and the graders would give it robustness 1.0 for a
    # task it never attempted: not a score. An agent timeout leaves a full trajectory and is scored.
    if not (trial / "agent" / "kernel_trajectory.jsonl").is_file():
        return f"  {trial.name:<44} no run record, not scored"
    tdir = task_dir(trial)
    exp = read_json(tdir / "tests" / "expected.json") if tdir else None
    if not exp:
        return f"  {trial.name:<44} no task identity, skipped"
    task_id = exp["task_id"]

    import yaml
    from claw_eval.graders.registry import get_grader
    from claw_eval.models.scoring import compute_task_score, is_pass
    from claw_eval.models.task import TaskDefinition

    src_task = SRC / "tasks" / task_id
    task_yaml = src_task / "task.yaml"
    spec = yaml.safe_load(task_yaml.read_text())
    # Oracle-backed graders resolve fixtures relative to task_file: it must be the pristine source path.
    task = TaskDefinition(**spec, task_file=str(task_yaml.resolve()))
    grader = get_grader(task_id, tasks_dir=SRC / "tasks", task_dir=src_task)
    split = exp.get("split", "general")
    judge = make_judge(judges["multi_turn" if split == "multi_turn" else "default"])
    try:
        scores = grader.grade(
            messages=build_messages(trial, spec["prompt"]["text"]), dispatches=build_dispatches(trial, spec), task=task,
            audit_data=read_json(trial / "verifier" / "audit.json") or {}, judge=judge, media_events=[],
            env_snapshot=with_reference_files(read_json(trial / "verifier" / "env_snapshot.json") or {}, spec, src_task))
    except Exception as exc:  # noqa: BLE001
        # A grader that could not run is not a zero: leave the result absent so a re-run picks it up.
        (trial / "verifier" / "claw_grading_error.json").write_text(
            json.dumps({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=1))
        return f"  {trial.name:<44} grading did not finish (not scored): {type(exc).__name__}"
    score = compute_task_score(scores)
    out_path.write_text(json.dumps({
        "task_id": task_id, "split": split, "judge_model": judge.model_id, "scores": scores.model_dump(),
        "task_score": score, "passed": is_pass(score), "judge_calls": len(judge.get_call_log()),
    }, ensure_ascii=False, indent=1))
    (trial / "verifier" / "claw_grading_error.json").unlink(missing_ok=True)
    write_reward(trial, score)
    return f"  {trial.name:<44} score={score:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    load_dotenv()
    if not (SRC / "tasks").is_dir():
        raise SystemExit(f"missing the Claw-Eval source clone at {SRC}")
    sys.path.insert(0, str(SRC / "src"))
    judges = {"default": judge_settings("CLAW_JUDGE", model="google/gemini-3.5-flash", effort="high"),
              "multi_turn": judge_settings("CLAW_MULTITURN_JUDGE", model="anthropic/claude-opus-4.6")}
    trials = family_trials(args.job, "claw")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(lambda t: judge_trial(t, args.force, judges), trials):
            print(line, flush=True)
    done = [d for d in (read_json(t / "verifier" / "claw_official_result.json") for t in trials) if d]
    if done:
        print(f"judged {len(done)}/{len(trials)}, mean task_score {sum(d['task_score'] for d in done) / len(done):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
