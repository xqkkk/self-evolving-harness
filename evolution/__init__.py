"""The outer loop of harness self-evolution (Algorithm 1 in the paper).

Each iteration has three steps:

1. rollout: the solver runs the iteration's task batch with the current harness H_t, one Harbor job per
   benchmark (``harbor.py``), and host-side judges score the benchmarks that are not graded inside the
   container (``judges/``);
2. update: the same agent, on the same harness, receives an update task that holds the harness source,
   its notes, and the complete run records, and edits the harness (``update_task.py``);
3. check: the edited harness is adopted if the update session ended normally and passed the
   engineering checks; otherwise repair sessions follow (``sessions.py``).

``stage1.py`` runs the first stage (multi-task pretraining, Iterations 1-5), ``stage2.py`` the second
(continual training on a new domain with success anchors, Iterations 6-7), and ``evaluate.py`` the
held-out evaluation of a frozen harness.
"""
