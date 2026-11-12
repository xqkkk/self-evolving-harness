"""Host-side judges.

Some benchmarks cannot be graded inside the task container: their official grading calls an LLM judge
or needs files the agent must never see. For these, the in-container verifier only collects evidence,
and a host-side pass scores each trial with the benchmark's official grading and writes the reward back
into the trial (``verifier/reward.txt`` and ``result.json`` at ``verifier_result.rewards.reward``):

    python -m judges.gaia2            <job-dir>   # GAIA2: ARE's official GraphPerEventJudge
    python -m judges.claw_eval        <job-dir>   # Claw-Eval: the task's own grader.py
    python -m judges.apex_agents      <job-dir>   # APEX-Agents: Archipelago's grading runner
    python -m judges.browsecomp_plus  <job-dir>   # BrowseComp-Plus: the official grader prompt

Each pass is idempotent (``--force`` re-judges), skips trials in which the agent never ran, and never
turns a judge failure into a zero: an unjudged trial keeps no reward and is picked up by the next pass. The full grader output, rationale included, is stored in the
trial; it reaches the proposer as part of the run records of training tasks. Ground truth never enters a
task container while the agent runs.

Judge models are configured through environment variables (``<PREFIX>_JUDGE_MODEL``, ``_BASE_URL``,
``_API_KEY``, ...); see each module and ``.env.example``.

``officeqa_score.py`` is different: it runs inside the OfficeQA Pro container as the task's verifier,
around the benchmark's official ``reward.py``.
"""
