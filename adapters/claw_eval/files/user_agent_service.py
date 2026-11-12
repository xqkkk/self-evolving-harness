"""In-container user simulator for Claw-Eval's multi-turn tasks.

Claw-Eval's own runner drives multi-turn tasks by watching for an assistant
message with no tool call and then generating the user's next line from a
persona (official model: Claude Opus 4.6). Our agent lives in a shell, so there
is no such hook — it has to address the user explicitly. This service is that
address: the agent POSTs its message, the simulated user answers.

The model call goes to the local relay (adapters/relay/queue_server.py on :9200);
the host daemon (python -m adapters.relay.host --preset claw) answers it with
the host's API key, so no key enters the container.

Every exchange is appended to /logs/verifier/user_agent.jsonl so the judge can
rebuild the full conversation for the grader — the graders score clarification
quality, so the user's turns have to be in `messages`, not just the agent's.

    POST /user/message {"text": "..."}  -> {"reply": ..., "round": n,
                                            "rounds_left": m, "done": bool}
    GET  /user/health

stdlib only.
"""

from __future__ import annotations

import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CONFIG = Path(os.environ.get("CLAW_USER_AGENT_CONFIG", "/claw/user_agent.json"))
LOG = Path(os.environ.get("CLAW_USER_AGENT_LOG", "/logs/verifier/user_agent.jsonl"))
RELAY = os.environ.get("CLAW_RELAY_URL", "http://localhost:9200/v1/chat/completions")
PORT = int(os.environ.get("CLAW_USER_AGENT_PORT", "9199"))
# The benchmark's own simulator model. The host relay routes by this name; no reasoning
# effort is passed, as in the benchmark's own config.
MODEL = os.environ.get("CLAW_USER_AGENT_MODEL", "anthropic/claude-opus-4.6")

_cfg = json.loads(CONFIG.read_text()) if CONFIG.is_file() else {}
PERSONA = _cfg.get("persona") or ""
MAX_ROUNDS = int(_cfg.get("max_rounds") or 8)
LANGUAGE = _cfg.get("language") or "en"

_history: list[dict] = []   # the conversation as the simulated user sees it
_rounds = 0

_SYSTEM = (
    "You are role-playing a real person talking to an AI assistant. Stay in "
    "character and answer only from what this person would plausibly know.\n\n"
    "Your character:\n{persona}\n\n"
    "Rules:\n"
    "- Reply the way this person would write in a chat: short, natural, no "
    "narration, no stage directions, no meta-commentary.\n"
    "- If the assistant asks for something your character does not know, say "
    "so plainly rather than inventing precise figures.\n"
    "- Do not solve the task for the assistant and do not volunteer everything "
    "at once; answer what was asked.\n"
    "- Reply in {language}.\n"
)


def _ask_model(assistant_text: str) -> str:
    _history.append({"role": "user", "content": assistant_text})
    messages = [{"role": "system", "content": _SYSTEM.format(
        persona=PERSONA, language="Chinese" if LANGUAGE == "zh" else "English")}]
    # From the simulated user's side the roles are mirrored: what the agent
    # said is the "user" turn, what the persona said is the "assistant" turn.
    messages += _history
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": 2048, "stream": False,
    }).encode()
    req = urllib.request.Request(
        RELAY, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode())
    reply = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    _history.append({"role": "assistant", "content": reply})
    return reply


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/user/health", "/user"):
            self._send(200, {"ok": True, "rounds_left": MAX_ROUNDS - _rounds})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        global _rounds
        if self.path.rstrip("/") != "/user/message":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "body must be JSON"})
            return
        text = (payload.get("text") or payload.get("message") or "").strip()
        if not text:
            self._send(400, {"error": "field 'text' is required"})
            return
        if _rounds >= MAX_ROUNDS:
            self._send(200, {"reply": "", "round": _rounds, "rounds_left": 0,
                             "done": True,
                             "note": "conversation budget exhausted; finish your answer"})
            return
        try:
            reply = _ask_model(text)
        except Exception as exc:  # noqa: BLE001
            self._send(502, {"error": f"user simulator unavailable: {exc}"})
            return
        _rounds += 1
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"round": _rounds, "assistant": text, "user": reply},
                                ensure_ascii=False) + "\n")
        self._send(200, {"reply": reply, "round": _rounds,
                         "rounds_left": MAX_ROUNDS - _rounds,
                         "done": _rounds >= MAX_ROUNDS})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
