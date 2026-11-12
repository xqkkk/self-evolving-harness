"""Where tool effects land: ``HarborWorld`` (the trial container) or ``LocalWorld`` (a local scratch dir).

Every command is recorded in full: the request, exit code, complete stdout and stderr and timing,
or the error that ended it, cancellation included.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXEC_TIMEOUT_S = 3600


def _exec_request(command, cwd, env, timeout_s, user) -> dict:
    """The command as the harness asked for it, recorded verbatim."""
    return {
        "command": command,
        "cwd": cwd,
        "env": env,
        "timeout_s": timeout_s or DEFAULT_EXEC_TIMEOUT_S,
        "user": user,
    }


def _describe(e: BaseException) -> str:
    if isinstance(e, asyncio.CancelledError):
        return "cancelled (agent timeout)"
    return f"{type(e).__name__}: {e}"


@dataclass
class ExecOutcome:
    stdout: str
    stderr: str
    return_code: int


class HarborWorld:
    """Executes commands in the trial's container through Harbor's ``BaseEnvironment``."""

    def __init__(self, environment, recorder):
        self.environment = environment
        self.recorder = recorder

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | int | None = None,
    ) -> ExecOutcome:
        request = _exec_request(command, cwd, env, timeout_s, user)
        started = time.monotonic()
        try:
            r = await self.environment.exec(
                command, cwd=cwd, env=env, timeout_sec=int(timeout_s or DEFAULT_EXEC_TIMEOUT_S), user=user
            )
        except BaseException as e:
            self.recorder.record("exec", **request, error=_describe(e),
                                 duration_s=round(time.monotonic() - started, 3))
            raise
        outcome = ExecOutcome(r.stdout or "", r.stderr or "", r.return_code)
        self.recorder.record("exec", **request, return_code=outcome.return_code, stdout=outcome.stdout,
                             stderr=outcome.stderr, duration_s=round(time.monotonic() - started, 3))
        return outcome


class LocalWorld:
    """Executes commands in a local scratch directory through a subprocess (used by the smoke check)."""

    def __init__(self, root: Path, recorder):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.recorder = recorder

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | int | None = None,
    ) -> ExecOutcome:
        request = _exec_request(command, cwd, env, timeout_s, user)
        started = time.monotonic()
        kwargs = {}
        if user is not None:
            kwargs["user"] = user
        proc = await asyncio.create_subprocess_shell(
            command, cwd=cwd or self.root, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s or DEFAULT_EXEC_TIMEOUT_S)
        except BaseException as e:
            proc.kill()
            self.recorder.record("exec", **request, error=_describe(e),
                                 duration_s=round(time.monotonic() - started, 3))
            raise
        outcome = ExecOutcome(out.decode(errors="replace"), err.decode(errors="replace"), proc.returncode or 0)
        self.recorder.record("exec", **request, return_code=outcome.return_code, stdout=outcome.stdout,
                             stderr=outcome.stderr, duration_s=round(time.monotonic() - started, 3))
        return outcome
