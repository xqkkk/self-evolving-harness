# Benchmark builders

No benchmark content is distributed with this repository. Each builder reads the benchmark's official release and
writes Harbor task directories under `datasets/`; `configs/splits/` names the tasks the paper used. You need to accept
each benchmark's own license and terms (several datasets are gated on Hugging Face and need `HF_TOKEN`).

| Benchmark | Official source (license) | Build | Paper role |
|---|---|---|---|
| Terminal-Bench 2.1 | Harbor registry `terminal-bench/terminal-bench-2-1` | nothing to build; the loop runs it from the registry | stage 1 |
| GAIA2 | `meta-agents-research-environments/gaia2` on HF, through Harbor's GAIA2 adapter; ARE 1.2.0 | `python -m adapters.gaia2.build_tasks` | stage 1 |
| OfficeQA Pro | `databricks/officeqa` on HF (gated; data CC-BY-SA 4.0) and github.com/databricks/officeqa (`reward.py`, Apache-2.0) | `python -m adapters.officeqa tasks`, then `snapshot` | stage 1 |
| tau3-Bench | github.com/sierra-research/tau2-bench (MIT), through Harbor's tau3 adapter | `python -m adapters.tau3.build_tasks` | stage 1 |
| SWE-Bench Pro | Harbor registry `scale-ai/swe-bench-pro` | `python -m adapters.swe_bench_pro` | stage 1 |
| SWE-bench Verified | Harbor registry `swebench-verified` (Harbor's `adapters/swebench`) | `uv run harbor datasets download swebench-verified -o datasets` | out-of-distribution |
| DeepSWE 1.1 | github.com/datacurve-ai/deep-swe (Apache-2.0; already in Harbor format) | clone it to `datasets/deep-swe` | out-of-distribution |
| BrowseComp-Plus | `Tevatron/browsecomp-plus` and `Tevatron/browsecomp-plus-corpus` on HF (MIT) | `python -m adapters.browsecomp_plus tasks`, then `snapshot` | out-of-distribution |
| APEX-Agents | `mercor/apex-agents` on HF (gated; CC-BY 4.0) and github.com/Mercor-Intelligence/archipelago (Apache-2.0) | `python -m adapters.apex_agents tasks`, then `snapshot` | out-of-distribution |
| Claw-Eval | github.com/claw-eval/claw-eval (MIT, commit `5680b8b`) and `claw-eval/Claw-Eval` on HF (the videos and large documents) | `python -m adapters.claw_eval.build_tasks --fixtures <HF download>` | out-of-distribution, stage 2 |

Each module's docstring gives the exact download commands. GAIA2 and tau3 build on Harbor's own adapters, which live in
a Harbor checkout (`git clone https://github.com/harbor-framework/harbor datasets/.harbor-src` at commit `8b1e898`).

## What the builders change, and why

The task content (instructions, fixtures, tests, answers) is the benchmark's own. The changes are about delivery,
grading location and keeping answers and credentials out of the agent's reach:

* **No answers in the agent's container.** SWE-Bench Pro's git history (which contains the fix) is replaced by a
  one-commit repository, and the gold tests are restored from `tests/` only for grading. tau3's main image no longer
  holds the tau2 task definitions. Claw-Eval's `grader.py` and `task.yaml` never enter a container (the build aborts if
  they do). BrowseComp-Plus answers stay encrypted until the host judge decrypts them.
* **No keys in any container.** Graders that need a model run on the host after the job (`judges/`). Services inside a
  task that need a model (tau3's user simulator and NL-assertion judge, Claw-Eval's simulated user) or the live web
  (Claw-Eval `web_real`) talk to an in-container relay (`relay/queue_server.py`); the host answers it with
  `python -m adapters.relay.host --preset tau3|claw --label run=<name>`, run alongside the job. Label the
  benchmark's sandboxes for the relay with `"env_kwargs": {"labels": {"run": "<name>"}}` in the config.
* **No network unless the benchmark needs it.** OfficeQA Pro and BrowseComp-Plus are corpus-only (the official
  agents had web search; mirrors of their answer keys are on GitHub and Hugging Face). SWE-Bench Pro, SWE-bench Verified
  and DeepSWE keep the network, as they are usually evaluated; fetching the upstream fix is caught by `anti_cheat`.
  Claw-Eval tasks reach only the hosts their own text cites (8 of 300 link documents such as arXiv PDFs).
* **GAIA2 follow-up turns fire.** The sidecar (`gaia2/are_mcp_server.py`, derived from Harbor's template) applies ARE's
  official scenario preprocessing and resolves oracle references, so multi-turn scenarios deliver their later turns;
  see its docstring for the full list of changes.
* **Pinned versions.** tau2-bench at `a2c0247` (upstream changed the banking grading between versions), Archipelago's
  environment at `8064965`, ARE at 1.2.0.
