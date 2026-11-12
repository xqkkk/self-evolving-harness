"""Every released harness loads and completes a scripted run on this kernel.

The world here records commands instead of executing them, so the test never runs a harness's
commands on the host; the real smoke check (``python -m harness_kernel.smoke``) runs inside the update
task's container.

    python -m pytest tests/test_harnesses.py
"""

from pathlib import Path

import pytest

from harness_kernel.smoke import run_smoke
from harness_kernel.world import ExecOutcome

ROOT = Path(__file__).resolve().parent.parent
HARNESSES = [
    "seed",
    "gpt-5.6-sol/run1-iteration5",
    "gpt-5.6-sol/run2-iteration5",
    "gpt-5.6-sol/run3-iteration5",
    "gpt-5.6-sol/run1-iteration7",
    "gpt-5.6-luna/iteration5",
    "glm-5.3/iteration5",
    "glm-5.3-flash/iteration5",
    "glm-5.3-flash/iteration7",
]


class DryWorld:
    """Records commands; every command 'succeeds' with empty output."""

    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.commands.append(command)
        return ExecOutcome("", "", 0)


@pytest.mark.parametrize("name", sorted(HARNESSES))
def test_scripted_run(name):
    result = run_smoke(ROOT / "harnesses" / name, world=DryWorld())
    assert result["model_calls"] >= 1
