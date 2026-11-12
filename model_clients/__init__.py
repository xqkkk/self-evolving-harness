"""Host-only model clients.

These modules are resolved by ``harness_kernel.providers`` on the host and are never copied into a
task container: the update task stages only ``harness_kernel/`` and the harness workspace. They hold
one property of the experiments outside the evolving harness's view: the input-token ceiling. Every
model in the paper ran with the same ceiling of 258,400 input tokens per request, enforced client-side.
An over-ceiling request is answered locally with the same ``context_length_exceeded`` error the
provider returns for a genuine overflow, so the harness sees an ordinary provider error and must manage
its own context.
"""
