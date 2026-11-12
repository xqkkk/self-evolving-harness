"""Host side of the relay: answer the requests parked by ``queue_server.py`` inside running Daytona sandboxes.

The daemon lists the started sandboxes that carry a run label, takes each sandbox's pending requests through the
Daytona process API, performs them on the host (model calls with the host's API key, web requests with the host's
ScraperAPI key), and posts the replies back into the sandbox. Keys never enter a sandbox.

    # tau3: the relay is the `llm-relay` compose service inside each sandbox (Docker in Docker)
    python -m adapters.relay.host --preset tau3 --label run=sol-run1
    # Claw-Eval: the user-simulator relay on :9200 and the ScraperAPI stand-in on :8089, in the task container
    python -m adapters.relay.host --preset claw --label run=sol-run1

Label the benchmark's sandboxes when launching Harbor (``--ek 'labels={"run": "sol-run1"}'``) so that only this run's
sandboxes are served. Every paid model call is logged to ``$HARNESS_COST_LOG`` when set. Stop with Ctrl-C or by
creating the ``--stop-file``.

Model routing: requests name the model the benchmark asks for; ``--model-map ASKED=SENT`` (a trailing ``*`` matches
a prefix) maps it to the upstream model. Presets: tau3 maps ``gpt-5.2*`` (the tau2 user simulator, reasoning effort
low) to ``openai/gpt-5.2`` and ``deepseek-v4-flash*`` (the NL-assertion judge) to ``deepseek/deepseek-v4-flash-0731``;
Claw-Eval maps ``anthropic/claude-opus-4.6`` (its user simulator) to itself. With an OpenRouter base URL the
request is routed to the cheapest endpoint that supports all its parameters (``provider.sort = "price"``).
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
import shlex
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from evolution.common import load_dotenv
from model_clients.common import log_cost

PRESETS = {
    "tau3": {"llm": ["8088@llm-relay"], "http": [],
             "model_map": ["gpt-5.2*=openai/gpt-5.2", "deepseek-v4-flash*=deepseek/deepseek-v4-flash-0731"]},
    "claw": {"llm": ["9200"], "http": ["8089"],
             "model_map": ["anthropic/claude-opus-4.6=anthropic/claude-opus-4.6"]},
}
SCRAPERAPI = "https://api.scraperapi.com"
SCRAPER_CONCURRENCY = 4  # ScraperAPI allows 5 concurrent requests per key

_PENDING_PY = ("import urllib.request;print(urllib.request.urlopen('http://localhost:{port}/relay/pending',"
               "timeout=5).read().decode())")
_REPLY_PY = ("import sys,urllib.request;r=urllib.request.Request('http://localhost:{port}/relay/reply/{rid}',"
             "data=sys.stdin.buffer.read(),headers={{'Content-Type':'application/json'}});"
             "urllib.request.urlopen(r,timeout=20).read()")


class SandboxQueue:
    """A queue_server inside a sandbox: on the sandbox itself, or in a compose container (``PORT@container``)."""

    def __init__(self, spec: str):
        port, _, container = spec.partition("@")
        self.port, self.container = int(port), container or None

    def _python(self, code: str, stdin: str | None = None) -> str:
        redirect = f" < {stdin}" if stdin else ""
        if not self.container:
            return f"python3 -c {shlex.quote(code)}{redirect}"
        return (f"RID=$(docker ps -qf name={shlex.quote(self.container)} 2>/dev/null | head -1); "
                f'[ -n "$RID" ] && docker exec -i "$RID" python3 -c {shlex.quote(code)}{redirect}')

    def pending(self, box) -> list[dict]:
        res = box.process.exec(self._python(_PENDING_PY.format(port=self.port)), timeout=25)
        out = (res.result or "").strip()
        if res.exit_code != 0 or not out.startswith("["):
            return []
        return json.loads(out)

    def reply(self, box, rid: str, payload: dict) -> None:
        remote = f"/tmp/relay-reply-{uuid.uuid4().hex}.json"
        box.fs.upload_file(json.dumps(payload).encode(), remote)
        box.process.exec(self._python(_REPLY_PY.format(port=self.port, rid=rid), stdin=remote) + f"; rm -f {remote}",
                         timeout=60)


class ModelUpstream:
    """Chat completions against an OpenAI-compatible endpoint with the host's key."""

    def __init__(self, base_url: str, key: str, model_map: list[str]):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key = key
        self.openrouter = "openrouter.ai" in base_url
        self.rules = [tuple(m.split("=", 1)) for m in model_map]
        self.http = httpx.Client(timeout=httpx.Timeout(600, connect=30))

    def target(self, asked: str) -> str | None:
        for pattern, sent in self.rules:
            if asked == pattern or (pattern.endswith("*") and asked.startswith(pattern[:-1])):
                return sent
        return None

    def __call__(self, request: dict, purpose: str) -> dict:
        if request.get("path") != "/v1/chat/completions":
            return {"status": 400, "body": {"error": {"message": f"relay: unsupported path {request.get('path')}"}}}
        body = dict(request.get("body") or {})
        sent = self.target(str(body.get("model", "")))
        if sent is None:
            return {"status": 400, "body": {"error": {"message": f"relay: no route for model {body.get('model')!r}"}}}
        body["model"] = sent
        body.pop("allowed_openai_params", None)
        if self.openrouter:
            effort = body.pop("reasoning_effort", None)
            if effort:
                body["reasoning"] = {"effort": "xhigh" if effort == "max" else effort}
            body["provider"] = {**(body.get("provider") or {}), "sort": "price", "require_parameters": True}
        for attempt in range(4):
            try:
                resp = self.http.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"})
                data = resp.json()
            except Exception as exc:  # noqa: BLE001 - transport errors are retried, then reported
                if attempt == 3:
                    return {"status": 502, "body": {"error": {"message": f"relay upstream error: {type(exc).__name__}"}}}
                time.sleep(2 ** attempt)
                continue
            usage = data.get("usage") if isinstance(data, dict) else None
            if usage:
                log_cost({"id": data.get("id"), "model": data.get("model"), "provider": data.get("provider"),
                          "purpose": purpose, "cost_usd": usage.get("cost"),
                          "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens")})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3 and not usage:
                time.sleep(2 ** attempt)
                continue
            return {"status": resp.status_code, "body": data}
        return {"status": 502, "body": {"error": {"message": "relay: retries exhausted"}}}


class ScraperUpstream:
    """The parked web request, sent to ScraperAPI with the host's key (keys used round-robin)."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise SystemExit("no ScraperAPI key: set SERP_DEV_KEY (or pass --scraper-key-env)")
        self.keys = itertools.cycle(keys)
        self.slots = {k: threading.Semaphore(SCRAPER_CONCURRENCY) for k in keys}
        self.lock = threading.Lock()
        self.http = httpx.Client(timeout=httpx.Timeout(150, connect=30), follow_redirects=True)

    def __call__(self, request: dict, purpose: str) -> dict:
        with self.lock:
            key = next(self.keys)
        with self.slots[key]:
            try:
                resp = self.http.get(SCRAPERAPI + (request.get("path") or "/"),
                                     params=[*request.get("params", []), ("api_key", key)])
            except Exception as exc:  # noqa: BLE001
                return {"status": 502, "content_type": "application/json",
                        "body_b64": base64.b64encode(json.dumps({"error": type(exc).__name__}).encode()).decode()}
        return {"status": resp.status_code, "content_type": resp.headers.get("content-type", "text/html"),
                "body_b64": base64.b64encode(resp.content).decode()}


class Relay:
    def __init__(self, queues: list[tuple[SandboxQueue, object]]):
        self.queues = queues
        self.done: dict[str, dict] = {}  # request id -> reply, so a re-offered request is never paid twice
        self.inflight: set[str] = set()
        self.lock = threading.Lock()

    def serve(self, box) -> int:
        served = 0
        for queue, upstream in self.queues:
            for request in queue.pending(box):
                rid = str(request["id"])
                with self.lock:
                    reply = self.done.get(rid)
                    if reply is None and rid in self.inflight:
                        continue
                    self.inflight.add(rid)
                try:
                    if reply is None:
                        reply = upstream(request, purpose=f"relay:{box.id}:{rid}")
                        with self.lock:
                            self.done[rid] = reply
                    queue.reply(box, rid, reply)
                    served += 1
                finally:
                    with self.lock:
                        self.inflight.discard(rid)
        return served


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--label", required=True, help="KEY=VALUE label of this run's sandboxes")
    ap.add_argument("--orgs", nargs="+", default=["DAYTONA_API_KEY"], help="env vars holding Daytona API keys")
    ap.add_argument("--llm", action="append", default=[], help="model queue PORT or PORT@container (repeatable)")
    ap.add_argument("--http", action="append", default=[], help="web queue PORT (repeatable)")
    ap.add_argument("--model-map", action="append", default=[], help="ASKED=SENT, trailing * = prefix (repeatable)")
    ap.add_argument("--base-url", default=os.environ.get("RELAY_BASE_URL", "https://openrouter.ai/api/v1"))
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY", help="env var with the model API key")
    ap.add_argument("--scraper-key-env", action="append", default=[], help="env var(s) with ScraperAPI keys")
    ap.add_argument("--poll", type=float, default=1.5)
    ap.add_argument("--stop-file", type=Path)
    args = ap.parse_args()
    load_dotenv()
    preset = PRESETS.get(args.preset or "", {})
    llm_specs = args.llm or preset.get("llm", [])
    http_specs = args.http or preset.get("http", [])
    model_map = args.model_map or preset.get("model_map", [])

    from daytona import Daytona, DaytonaConfig, ListSandboxesQuery

    queues = []
    if llm_specs:
        key = os.environ.get(args.key_env, "")
        if not key:
            raise SystemExit(f"{args.key_env} is not set")
        model = ModelUpstream(args.base_url, key, model_map)
        queues += [(SandboxQueue(s), model) for s in llm_specs]
    if http_specs:
        names = args.scraper_key_env or sorted(k for k in os.environ if k == "SERP_DEV_KEY" or k.startswith("SERP_DEV_KEY_"))
        scraper = ScraperUpstream([os.environ[n] for n in names if os.environ.get(n)])
        queues += [(SandboxQueue(s), scraper) for s in http_specs]
    if not queues:
        raise SystemExit("nothing to relay: pass --preset, --llm or --http")
    relay = Relay(queues)
    name, _, value = args.label.partition("=")
    labels = {name: value}
    clients = [Daytona(DaytonaConfig(api_key=os.environ[k])) for k in args.orgs if os.environ.get(k)]
    if not clients:
        raise SystemExit(f"none of {args.orgs} is set")
    print(f"relay: {len(clients)} organization(s), label {labels}, queues {llm_specs + http_specs}", flush=True)

    busy: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        while not (args.stop_file and args.stop_file.exists()):
            for sid, fut in list(busy.items()):
                if fut.done():
                    try:
                        n = fut.result()
                        if n:
                            print(f"{time.strftime('%H:%M:%S')} served {n} request(s) for {sid[:12]}", flush=True)
                    except Exception as exc:  # noqa: BLE001 - one sandbox's error must not stop the relay
                        print(f"{time.strftime('%H:%M:%S')} {sid[:12]}: {type(exc).__name__}: {str(exc)[:160]}",
                              flush=True)
                    del busy[sid]
            for client in clients:
                try:
                    boxes = list(client.list(ListSandboxesQuery(labels=labels)))
                except Exception as exc:  # noqa: BLE001
                    print(f"list failed: {type(exc).__name__}", flush=True)
                    continue
                for box in boxes:
                    if str(box.state).lower().endswith("started") and box.id not in busy:
                        busy[box.id] = pool.submit(relay.serve, box)
            time.sleep(args.poll)
    print("relay stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
