"""Judge one GAIA2 trial offline with ARE's official GraphPerEventJudge.

Runs inside the dedicated judge environment (``judges/gaia2.py`` creates it and invokes this script per
trial), which carries the ``meta-agents-research-environments`` dependency tree.

    python gaia2_worker.py <scenario.json> <are_trace.json> <out.json>

The recipe is ARE's own offline-validation path, the one behind ``are-benchmark judge``:

1. rebuild the scenario from the task's original scenario.json (never from the trace: the sidecar
   exports a turn-rewired event graph, and preprocessing must start from the pristine one);
2. ``preprocess_scenario(offline_validation=True)``: replay the oracle script in oracle mode
   (deterministic, no LLM), attach the official judge, segment turns by ``send_message_to_user``;
3. read the agent's completed events from the sidecar's official trace export;
4. ``scenario.validate(env)`` over an event-log-only Environment, exactly as ``ScenarioRunner`` judges.

Environment: ``GAIA2_JUDGE_MODEL``, ``GAIA2_JUDGE_BASE_URL``, ``GAIA2_JUDGE_API_KEY`` (an
OpenAI-compatible chat endpoint, called through LiteLLM), ``GAIA2_JUDGE_REASONING_EFFORT``,
``GAIA2_JUDGE_MAX_TOKENS``, ``GAIA2_JUDGE_NUM_VOTES``, and ``DEMO_FS_PATH`` (the official filesystem
snapshot, needed by oracle replays). Output JSON: ``{"success": bool|null, "rationale": str, ...}``.
"""

import json
import os
import sys
from pathlib import Path


def _install_case_insensitive_judge_parsing() -> None:
    """Match ARE's soft-checker verdict tokens case-insensitively.

    Ported from RHO (https://github.com/wbopan/retro-harness, MIT License). ``LLMChecker.__call__`` uses a
    case-sensitive substring test for ``[[True]]``/``[[False]]``, while LiteLLM normalizes True/False to
    lowercase and non-Llama judge models emit lowercase verdicts anyway; an unrecognized verdict reads
    as a refusal and fails every soft-checked tool call. Only the parsing changes, not the semantics.
    """
    from are.simulation.validation.utils import llm_utils as _llm

    if getattr(_llm.LLMChecker, "_ci_parsing", False):
        return

    def _case_insensitive_call(self, user_prompt_args):
        votes = []
        success, failure = self.success_str.lower(), self.failure_str.lower()
        for _ in range(self.num_votes):
            response = self.judge(user_prompt_args)
            if response is None:
                continue
            low = response.lower()
            if success in low:
                votes.append(True)
            elif failure in low:
                votes.append(False)
        if not votes:
            return None
        return sum(votes) >= len(votes) / 2

    _llm.LLMChecker.__call__ = _case_insensitive_call
    _llm.LLMChecker._ci_parsing = True


def _install_judge_call_params() -> None:
    """Inject ``reasoning_effort`` and ``max_tokens`` into the ARE engine's calls.

    ARE's LiteLLMEngine passes no sampling, thinking or length parameters, so the temperature stays at the
    model default, as intended. Two settings must be explicit, though: a reasoning tier that is only
    selectable per request, and an output cap large enough for that tier's thinking (a truncated
    response loses the ``[[True]]`` verdict and reads as a rejection).
    """
    effort = os.environ.get("GAIA2_JUDGE_REASONING_EFFORT", "").strip()
    max_tokens = int(os.environ.get("GAIA2_JUDGE_MAX_TOKENS", "32768"))
    from are.simulation.agents.llm.litellm import litellm_engine as _engine

    if getattr(_engine, "_params_injected", False):
        return
    _orig = _engine.completion

    def _with_params(*args, **kwargs):
        if effort:
            kwargs.setdefault("reasoning_effort", effort)
            kwargs.setdefault("allowed_openai_params", ["reasoning_effort"])  # LiteLLM may not know the model
        if max_tokens > 0:
            kwargs.setdefault("max_tokens", max_tokens)
        return _orig(*args, **kwargs)

    _engine.completion = _with_params
    _engine._params_injected = True


def _install_judge_num_votes() -> None:
    """Turn on the official checkers' majority voting (default: 1 vote).

    The judge samples at the model's default temperature, so a single-sample soft check is stochastic:
    the same trace re-judged can flip on a borderline comparison. ``LLMChecker`` ships a voting mechanism
    for this; the official call site never turns it on. Majority voting is symmetric: it stabilizes
    verdicts without loosening or tightening the rubric.
    """
    votes = int(os.environ.get("GAIA2_JUDGE_NUM_VOTES", "3"))
    if votes <= 1:
        return
    from are.simulation.validation import tool_judge as _tj

    if getattr(_tj, "_votes_injected", False):
        return
    _orig = _tj.build_llm_checkers

    def _with_votes(*args, **kwargs):
        kwargs.setdefault("num_votes", votes)
        return _orig(*args, **kwargs)

    _tj.build_llm_checkers = _with_votes
    _tj._votes_injected = True


def _build_judge_engine():
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine, LiteLLMModelConfig

    return LiteLLMEngine(model_config=LiteLLMModelConfig(
        model_name=os.environ["GAIA2_JUDGE_MODEL"], provider="openai",
        endpoint=os.environ["GAIA2_JUDGE_BASE_URL"], api_key=os.environ["GAIA2_JUDGE_API_KEY"]))


def main() -> int:
    scenario_path, trace_path, out_path = (Path(p) for p in sys.argv[1:4])
    _install_case_insensitive_judge_parsing()
    _install_judge_call_params()
    _install_judge_num_votes()

    from are.simulation.data_handler.importer import JsonScenarioImporter
    from are.simulation.environment import Environment
    from are.simulation.scenarios.config import MAX_SCENARIO_DURATION, MAX_TIME_SCENARIO_DURATION
    from are.simulation.scenarios.scenario_imported_from_json.utils import preprocess_scenario
    from are.simulation.types import CapabilityTag, EventLog
    from are.simulation.validation.configs import GraphPerEventJudgeConfig

    importer = JsonScenarioImporter()
    scenario, _, _ = importer.import_from_json_to_benchmark(scenario_path.read_text(), apps_to_skip=None,
                                                            load_completed_events=False)
    if scenario is None:
        raise SystemExit(f"failed to import scenario from {scenario_path}")

    raw_trace = trace_path.read_text()
    try:
        json.loads(raw_trace)
    except json.JSONDecodeError:
        # A trace copied while the sidecar was rewriting it: keep the first complete JSON document.
        obj, _end = json.JSONDecoder().raw_decode(raw_trace)
        raw_trace = json.dumps(obj)
    try:
        _, completed_events, _ = importer.import_from_json_to_benchmark(raw_trace, apps_to_skip=None,
                                                                        load_completed_events=True)
    except ValueError as e:
        # The agent called a tool with an empty required argument: the runtime accepted it, the official
        # importer refuses it. Such a trace cannot be judged by the official pipeline; it scores 0.
        out_path.write_text(json.dumps({"success": False, "rationale": f"trace unimportable by official importer: {e}",
                                        "exception": None}, ensure_ascii=False))
        return 0
    completed_events = completed_events or []

    max_duration = MAX_TIME_SCENARIO_DURATION if CapabilityTag.Time in (scenario.tags or ()) else MAX_SCENARIO_DURATION
    preprocess_scenario(scenario, judge_config=GraphPerEventJudgeConfig(engine=_build_judge_engine()),
                        max_scenario_duration=max_duration, offline_validation=True)
    env = Environment()
    env.event_log = EventLog.from_list_view(completed_events)
    result = scenario.validate(env)
    out_path.write_text(json.dumps({
        "success": result.success,
        "rationale": result.rationale,
        "exception": str(result.exception) if result.exception else None,
        "nb_turns": getattr(scenario, "nb_turns", None),
        "n_completed_events": len(completed_events),
    }, ensure_ascii=False, indent=2))

    # Debrief for the next update session: the sidecar withholds the scenario script, the world in plain
    # text and the apps' end state from the agent. Judging happens after the run and before the update,
    # so these can be given to the proposer as part of the run record; the agent container never sees
    # this file. The script is copied from the pristine scenario.json; the oracle-replay world is exported
    # on a best-effort basis.
    try:
        debrief: dict = {
            "__source__": ("host-side judge debrief for the next update session: the pristine scenario script "
                           "(events = oracle actions and the scheduled user/environment turns) plus, when "
                           "exportable, the oracle-replay world. The agent container never sees this file."),
            "scenario_events": json.loads(scenario_path.read_text()).get("events"),
        }
        try:
            from are.simulation.data_handler.exporter import JsonScenarioExporter

            debrief["oracle_export"] = json.loads(JsonScenarioExporter().export_to_json(env, scenario, scenario.scenario_id))
        except Exception as e:  # the export is best-effort
            debrief["oracle_export"] = None
            debrief["oracle_export_error"] = str(e)[:200]
        (out_path.parent / "oracle_debrief.json").write_text(json.dumps(debrief, ensure_ascii=False))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
