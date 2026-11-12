# Released harnesses

Each directory holds the code of one harness evaluated in the paper: `harness.py` and any modules it imports.
`lineage/` has the code after every iteration of the same runs.

| Directory | Model | Stage | Python lines |
|---|---|---|---:|
| `seed/` | (all) | seed | 49 |
| `glm-5.3/iteration5/` | GLM-5.3 (max) | 1 | 1086 |
| `gpt-5.6-sol/run1-iteration5/` | GPT-5.6 Sol (high) | 1, run 1 | 430 |
| `gpt-5.6-sol/run2-iteration5/` | GPT-5.6 Sol (high) | 1, run 2 | 562 |
| `gpt-5.6-sol/run3-iteration5/` | GPT-5.6 Sol (high) | 1, run 3 | 621 |
| `gpt-5.6-sol/run1-iteration7/` | GPT-5.6 Sol (high) | 2, from run 1 | 524 |
| `glm-5.3-flash/iteration5/` | GLM-5.3-Flash (max) | 1 | 889 |
| `glm-5.3-flash/iteration7/` | GLM-5.3-Flash (max) | 2 | 1123 |
| `gpt-5.6-luna/iteration5/` | GPT-5.6 Luna (high) | 1 | 221 |

Held-out results reported in the paper (average over the five in-distribution benchmarks, Table 4):

| Model | Seed | Production harness | Iteration 5 |
|---|---:|---:|---:|
| GLM-5.3 | 53.75 | 58.88 (Claude Code) | 59.01 |
| GPT-5.6 Sol | 55.69 | 59.56 (Codex) | 60.64 ± 0.38 (runs: 60.71, 60.99, 60.23) |
| GLM-5.3-Flash | 53.18 | 54.33 (Claude Code) | 55.12 |
| GPT-5.6 Luna | 46.47 | 54.77 (Codex) | 37.24 |

Out-of-distribution average of GPT-5.6 Sol (Table 5): Seed 58.46, Codex 71.23, Iteration 5 (run 1) 70.93.
Claw-Eval after the second stage (Figure 3): GPT-5.6 Sol 68.56 → 71.10 (Codex 69.50), GLM-5.3-Flash 63.52 → 66.33
(Claude Code 64.46).

A harness is written for the kernel in `harness_kernel/` and is model-agnostic in form, but each was evolved by,
and evaluated with, its own model.

```bash
python -m pytest tests/test_harnesses.py        # offline: a scripted run of every harness that executes no command
python -m evolution.evaluate --model gpt-5.6-sol-high \
    --workspace harnesses/gpt-5.6-sol/run1-iteration5 --name sol-run1-it5     # held-out evaluation
```

`python -m harness_kernel.smoke <workspace>` is the acceptance check that runs inside the update task's container.
It executes the harness's own commands (for example, the GLM-5.3 harness writes a task note to `/tmp` and a report to
`/logs/agent`), so run it in a container rather than on your workstation.
