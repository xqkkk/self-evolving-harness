"""Offline checks of the clients' input ceiling and of the update-task assembly. No request leaves the machine.

    python -m pytest tests/test_offline.py
"""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx
import openai

from harness_kernel.recorder import Recorder

ROOT = Path(__file__).resolve().parent.parent
BIG = [{"role": "user", "content": "word " * 2000}]


def test_capped_responses_answers_over_ceiling_locally():
    from model_clients.capped_responses import CappedResponsesModel

    m = CappedResponsesModel(Recorder(), max_input_tokens=100, base_url="http://127.0.0.1:9/v1", api_key="test",
                             model="m", max_output_tokens=16)
    try:
        asyncio.run(m.chat(BIG))
    except openai.BadRequestError as e:
        assert "context_length_exceeded" in str(e)
    else:
        raise AssertionError("expected a local context_length_exceeded error")


def test_openrouter_chat_answers_over_ceiling_locally():
    from model_clients.openrouter_chat import OpenRouterChatModel

    m = OpenRouterChatModel(Recorder(), model="m", api_key="test", base_url="http://127.0.0.1:9/v1", max_input_tokens=100)
    try:
        asyncio.run(m.chat(BIG))
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 400 and "context_length_exceeded" in e.response.text
    else:
        raise AssertionError("expected a local context_length_exceeded error")


def test_update_task_assembly_and_line_count(tmp_path):
    from evolution.update_task import build_update_task
    from harness_kernel.smoke import count_added_lines

    job = tmp_path / "job"
    trial = job / "task-a__abc"
    (trial / "agent").mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task": {"name": "t"}}))
    (trial / "agent" / "kernel_trajectory.jsonl").write_text(json.dumps({"type": "run_start", "path": str(ROOT)}) + "\n")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "are_trace.json").write_text("{}")
    task = build_update_task(tmp_path / "task", workspace=ROOT / "harnesses" / "seed", evidence_jobs=[job], budget=441)
    text = (task / "instruction.md").read_text()
    assert "at most 441 new lines" in text and "258,400 input tokens" in text and "{{" not in text
    app = task / "environment" / "app"
    assert (app / "added_lines_budget.txt").read_text().strip() == "441"
    copied = app / "eval_trials" / "task-a__abc"
    assert not (copied / "verifier" / "are_trace.json").exists()          # bulky export left out
    assert str(ROOT) not in (copied / "agent" / "kernel_trajectory.jsonl").read_text()  # host path scrubbed
    assert (app / "harness_workspace" / "evolve_notes.md").read_text() == "# Evolve notes\n"  # notes start empty

    # the gate's line count: commit the baseline as the Dockerfile does, then edit the workspace
    subprocess.run(["git", "-C", str(app), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(app), "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(app), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"], check=True)
    ws = app / "harness_workspace"
    (ws / "harness.py").write_text((ws / "harness.py").read_text() + "\n# one\n# two\n")
    (ws / "helper.py").write_text("a = 1\nb = 2\nc = 3\n")
    (ws / "evolve_notes.md").write_text("notes do not count\n" * 50)
    assert count_added_lines(app, ws) == 3 + 3


def test_repair_task_uses_the_repair_prompt(tmp_path):
    from evolution.update_task import build_update_task

    job = tmp_path / "job"
    (job / "s__x").mkdir(parents=True)
    (job / "s__x" / "config.json").write_text("{}")
    task = build_update_task(tmp_path / "task", workspace=ROOT / "harnesses" / "seed", evidence_jobs=[job], budget=100,
                             repair=True, agent_timeout_sec=7200)
    assert "repair the harness" in (task / "instruction.md").read_text()
    assert not (task / "instruction_repair.md").exists()
    assert "timeout_sec = 7200.0" in (task / "task.toml").read_text()
    assert "timeout_sec = 300.0" in (task / "task.toml").read_text()  # the verifier's limit is untouched
    shutil.rmtree(task)
