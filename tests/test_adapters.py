"""Offline checks of the in-sandbox relay, the host relay's routing and the anti-cheat scan. No network access.

    python -m pytest tests/test_adapters.py
"""

import base64
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from adapters.relay.host import ModelUpstream, SandboxQueue
from anti_cheat import scan
from anti_cheat.trajectories import steps

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


def _serve(*args):
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "adapters/relay/queue_server.py"), "--host", "127.0.0.1",
                             "--port", str(port), *args])
    for _ in range(50):
        try:
            _get(f"http://127.0.0.1:{port}/health")
            return proc, port
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("queue server did not start")


def _answer_one(port, reply):
    for _ in range(100):
        _, raw = _get(f"http://127.0.0.1:{port}/relay/pending")
        pending = json.loads(raw)
        if pending:
            _post(f"http://127.0.0.1:{port}/relay/reply/{pending[0]['id']}", reply)
            return pending[0]
        time.sleep(0.05)
    raise RuntimeError("nothing was parked")


def test_queue_server_relays_a_chat_completion_and_a_stream():
    proc, port = _serve()
    try:
        answer = {"id": "x", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                          "finish_reason": "stop"}]}
        for stream in (False, True):
            got = {}
            caller = threading.Thread(target=lambda: got.update(r=_post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                {"model": "gpt-5.2", "messages": [], "stream": stream})))
            caller.start()
            parked = _answer_one(port, {"status": 200, "body": answer})
            caller.join(10)
            assert parked["kind"] == "llm" and "stream" not in parked["body"]
            status, body = got["r"]
            assert status == 200
            if stream:
                assert body.startswith(b"data: ") and body.rstrip().endswith(b"[DONE]") and b'"hi"' in body
            else:
                assert json.loads(body) == answer
    finally:
        proc.kill()


def test_web_stand_in_drops_the_key_and_returns_the_body():
    proc, port = _serve("--http-get")
    try:
        got = {}
        caller = threading.Thread(target=lambda: got.update(r=_get(
            f"http://127.0.0.1:{port}/?api_key=SECRET&url=https%3A%2F%2Fexample.org")))
        caller.start()
        parked = _answer_one(port, {"status": 200, "content_type": "text/html",
                                    "body_b64": base64.b64encode(b"<html>ok</html>").decode()})
        caller.join(10)
        assert parked["kind"] == "http_get" and parked["params"] == [["url", "https://example.org"]]
        assert got["r"] == (200, b"<html>ok</html>")
        try:  # other paths (e.g. the verifier's /audit probe) are not parked
            _get(f"http://127.0.0.1:{port}/audit")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        proc.kill()


def test_host_relay_routing():
    up = ModelUpstream("https://openrouter.ai/api/v1", "k", ["gpt-5.2*=openai/gpt-5.2",
                                                             "deepseek-v4-flash*=deepseek/deepseek-v4-flash-0731"])
    assert up.target("gpt-5.2-2025-12-11") == "openai/gpt-5.2"
    assert up.target("deepseek-v4-flash") == "deepseek/deepseek-v4-flash-0731"
    assert up.target("claude") is None
    assert up({"path": "/v1/chat/completions", "body": {"model": "unknown"}}, "t")["status"] == 400
    q = SandboxQueue("8088@llm-relay")
    assert q.port == 8088 and "docker exec -i" in q._python("print(1)")
    assert SandboxQueue("9200")._python("print(1)").startswith("python3 -c ")


def _trial(tmp: Path, name: str, task: dict, agent_files: dict) -> Path:
    t = tmp / name
    (t / "agent").mkdir(parents=True)
    (t / "config.json").write_text(json.dumps({"task": task}))
    (t / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}))
    for rel, text in agent_files.items():
        (t / "agent" / rel).write_text(text)
    return t


def test_anti_cheat_flags_an_upstream_fetch_in_all_three_formats(tmp_path):
    patch = "\n".join("+    value = compute_the_fixed_result(argument_number_%d, other)" % i for i in range(5))
    task = tmp_path / "datasets" / "swe-bench-pro-nogit" / "instance_x__y-1"
    (task / "tests").mkdir(parents=True)
    (task / "tests" / "config.json").write_text(json.dumps({"patch": patch}))
    fetch = "curl -s https://github.com/o/r/commit/abc.patch"
    leaked = patch.replace("+", "")
    kernel = json.dumps({"type": "exec", "command": fetch, "stdout": leaked, "stderr": ""})
    codex = json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": fetch,
                                                           "aggregated_output": leaked}})
    cc = "\n".join(json.dumps(e) for e in (
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "1", "name": "Bash",
                                                        "input": {"command": fetch}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "1", "content": leaked}]}}))
    spec = {"path": str(task)}
    for name, files in (("kernel", {"kernel_trajectory.jsonl": kernel}), ("codex", {"codex.txt": codex}),
                        ("cc", {"claude-code.txt": cc})):
        t = _trial(tmp_path, name, spec, files)
        family = scan.family_of(json.loads((t / "config.json").read_text()))
        s = steps(t)
        assert family == "sbp" and len(s) == 1
        assert scan.pattern_hits(family, s), name
        assert scan.content_leak(family, scan.gold_marks(t, family, tmp_path, []), s)["fragments_found"] == 5, name
