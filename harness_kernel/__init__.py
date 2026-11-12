"""The fixed execution interface (K in the paper).

The kernel is the one part of the system that the evolving harness never edits. It

1. loads ``harness.py`` from a harness workspace and calls its ``async run(instruction, model, world)``;
2. hands the harness a model client (``await model.chat(messages, tools)``) and a world
   (``await world.exec(command, timeout_s=...)``);
3. records every model call and every command, unabridged, to ``kernel_trajectory.jsonl``.

The same kernel runs both roles of the agent: the solver on benchmark tasks and the proposer on the
update task. During an update session it is copied into the task container as ``/app/harness_kernel``
so the proposer can read it, and ``python -m harness_kernel.smoke`` is the offline check the
acceptance gate runs.

Importing this package must not pull in Harbor or any model SDK: inside the update-task container
only ``loader``, ``model``, ``recorder``, ``world`` and ``smoke`` are used.
"""
