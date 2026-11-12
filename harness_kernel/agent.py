"""Harbor adapter: loads a harness workspace and runs it against one task.

    harbor run -p <task-dir> -a harness_kernel.agent:KernelAgent -m <model-alias> \
        --ak workspace=<harness-workspace>

``<model-alias>`` is looked up in the providers file (see ``providers.py``).
"""

import asyncio
import os
import traceback
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .loader import load_workspace, workspace_hash
from .providers import resolve_model
from .recorder import FILENAME, Recorder
from .world import HarborWorld


class KernelAgent(BaseAgent):
    # Appended to every task instruction, for every harness (Seed, evolved harnesses alike).
    NO_CHEAT = (
        "\nNever cheat: do not obtain the answer from any source that carries it "
        "rather than solving the task — the upstream fix or its pull request, a "
        "published solution or test file, a dataset copy of the answer, or an "
        "oracle/evaluation artifact left in the environment. Reading ordinary "
        "documentation and library source is fine. Solve the task yourself.\n"
    )

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *args,
        workspace: str | None = None,
        max_tokens: str | int | None = None,
        **kwargs,
    ):
        super().__init__(logs_dir, model_name, *args, **kwargs)
        ws = workspace or os.environ.get("HARNESS_WORKSPACE") or "harnesses/seed"
        self.workspace = Path(ws).expanduser().resolve()
        self.max_tokens = int(max_tokens) if max_tokens else None

    @staticmethod
    def name() -> str:
        return "harness_kernel"

    def version(self) -> str | None:
        try:
            return workspace_hash(self.workspace)
        except OSError:
            return "unknown"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    def _mcp_note(self) -> str:
        """Append the MCP servers declared by the task to the instruction."""
        if not self.mcp_servers:
            return ""
        lines = ["", "MCP servers reachable from inside the task container:"]
        for s in self.mcp_servers:
            if s.url:
                lines.append(f"- {s.name}: {s.transport} at {s.url}")
            else:
                cmd = " ".join([s.command or "", *s.args]).strip()
                lines.append(f"- {s.name}: stdio via `{cmd}`")
        return "\n".join(lines) + "\n"

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name:
            raise ValueError("harness_kernel requires a model: pass -m/--model to harbor run")

        recorder = Recorder(Path(self.logs_dir) / FILENAME)
        model = resolve_model(self.model_name, recorder, max_tokens=self.max_tokens)
        world = HarborWorld(environment, recorder)
        instruction = instruction + self.NO_CHEAT + self._mcp_note()
        recorder.record(
            "run_start",
            instruction=instruction,
            model=self.model_name,
            workspace_hash=self.version(),
            trajectory_format=(
                "model_call.request: `messages_appended` = previous call's messages "
                "again, unchanged, plus these; `messages` = the whole list (first call, "
                "or the history was rewritten); `tools` present only when changed."
            ),
            mcp_servers=[
                {"name": s.name, "transport": s.transport, "url": s.url, "command": s.command,
                 "args": list(s.args or [])}
                for s in (self.mcp_servers or [])
            ],
        )
        error: str | None = None
        cancelled = False
        try:
            module = load_workspace(self.workspace)
            await module.run(instruction, model, world)
        except asyncio.CancelledError:
            # Harbor cancels the agent when the task's time limit is reached.
            cancelled = True
            error = "cancelled (agent timeout)"
            raise
        except Exception as e:
            # A harness crash ends the run; the trial is still graded as it stands.
            error = f"{type(e).__name__}: {e}"
            recorder.record("harness_crash", error=error, traceback=traceback.format_exc())
            self.logger.error("harness crashed: %s", error)
        finally:
            context.n_input_tokens = model.input_tokens
            context.n_cache_tokens = model.cache_tokens
            context.n_output_tokens = model.output_tokens
            context.cost_usd = model.cost_usd or None
            context.metadata = {
                "workspace_hash": self.version(),
                "model_calls": model.calls,
                "harness_error": error,
            }
            recorder.record(
                "run_end",
                error=error,
                cancelled=cancelled,
                model_calls=model.calls,
                input_tokens=model.input_tokens,
                cache_tokens=model.cache_tokens,
                output_tokens=model.output_tokens,
                duration_s=recorder.elapsed(),
            )
            recorder.close()
