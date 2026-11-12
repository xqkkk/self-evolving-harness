# Harness code after every iteration

The harness code of every iteration of the evolution runs reported in the paper. Each `iterationNN/` holds the
code of the harness after that iteration (`harness.py` and the modules it imports), and `iterationNN.diff` is the
unified diff of the code against the previous iteration; Iteration 1 is compared with `harnesses/seed`.

Added and removed lines count the lines of the unified diff, as in the paper; the size is the total number of
lines of the Python files, including blank lines and comments.

## GPT-5.6 Sol, run 1 (Iterations 6-7: second stage on Claw-Eval): `gpt-5.6-sol/run1/`

| Iteration | Python lines | Added | Removed |
|---|---:|---:|---:|
| Seed | 49 | | |
| 1 | 279 | 262 | 32 |
| 2 | 343 | 91 | 27 |
| 3 | 394 | 54 | 3 |
| 4 | 414 | 32 | 12 |
| 5 | 430 | 20 | 4 |
| 6 | 486 | 58 | 2 |
| 7 | 524 | 42 | 4 |

## GPT-5.6 Sol, run 2: `gpt-5.6-sol/run2/`

| Iteration | Python lines | Added | Removed |
|---|---:|---:|---:|
| Seed | 49 | | |
| 1 | 292 | 274 | 31 |
| 2 | 332 | 67 | 27 |
| 3 | 468 | 174 | 38 |
| 4 | 523 | 70 | 15 |
| 5 | 562 | 40 | 1 |

## GPT-5.6 Sol, run 3: `gpt-5.6-sol/run3/`

| Iteration | Python lines | Added | Removed |
|---|---:|---:|---:|
| Seed | 49 | | |
| 1 | 429 | 410 | 30 |
| 2 | 547 | 133 | 15 |
| 3 | 602 | 66 | 11 |
| 4 | 624 | 44 | 22 |
| 5 | 621 | 31 | 34 |

## GPT-5.6 Luna: `gpt-5.6-luna/`

| Iteration | Python lines | Added | Removed |
|---|---:|---:|---:|
| Seed | 49 | | |
| 1 | 173 | 130 | 6 |
| 2 | 193 | 28 | 8 |
| 3 | 204 | 17 | 6 |
| 4 | 221 | 22 | 5 |
| 5 | 221 | 3 | 3 |

## GLM-5.3: `glm-5.3/`

| Iteration | Python lines | Added | Removed |
|---|---:|---:|---:|
| Seed | 49 | | |
| 1 | 504 | 498 | 43 |
| 2 | 800 | 323 | 27 |
| 3 | 882 | 148 | 66 |
| 4 | 997 | 128 | 13 |
| 5 | 1086 | 91 | 2 |

## GLM-5.3-Flash (Iterations 6-7: second stage on Claw-Eval): `glm-5.3-flash/`

| Iteration | Python lines | Added | Removed |
|---|---:|---:|---:|
| Seed | 49 | | |
| 1 | 506 | 496 | 39 |
| 2 | 640 | 159 | 25 |
| 3 | 695 | 94 | 39 |
| 4 | 760 | 100 | 35 |
| 5 | 889 | 164 | 35 |
| 6 | 1074 | 193 | 8 |
| 7 | 1123 | 76 | 27 |
