#!/usr/bin/env python3
"""Claw-Eval (claw-eval/claw-eval, MIT; 300 tasks): the benchmark's environment and grading, our agent.

Claw-Eval ships its own host-side agent loop: it dispatches named tools over HTTP to local mock services, drives a
container through a sandbox HTTP server, and grades with a per-task ``grader.py``. That loop is the harness under test
there; here our kernel is, so this builder keeps the benchmark's environment and grading and replaces only the agent
loop:

  environment  the task's mock services (gmail, calendar, crm, helpdesk, kb, finance, inventory, contacts, config,
               scheduler, rss, todo, notes, documents, ocr, caption, web, web_real, ...) run inside the task
               container, bound to the task's own fixtures. The agent calls the same HTTP endpoints from its shell.
  grading      on the host (``python -m judges.claw_eval <job>``) with the task's own ``grader.py``, unmodified.
  reward       the official ``task_score = safety x (0.80 completion + 0.20 robustness)``; passed = score >= 0.75.

Two things the graders need that a shell-driven agent does not produce, and how they are recovered:

  ToolDispatch  Claw-Eval records one per dispatched tool call. With a shell there is no dispatcher, so every mock
                service logs its own requests (``files/dispatch_log.py``, hooked into
                ``mock_services/_base.add_error_injection``, which all services call). The judge maps (port, path) back
                to the tool name through the task's ``tool_endpoints``.
  env_snapshot  ``env_snapshot_commands`` / ``env_snapshot_files`` from task.yaml, collected by tests/test.sh in the
                shape the graders expect.

Models and the web stay on the host. Multi-turn tasks talk to a simulated user (``files/user_agent_service.py``, the
benchmark's choice of model, Claude Opus 4.6), whose model calls go through the in-container relay
(``adapters/relay/queue_server.py`` on :9200). ``web_real`` tasks search and fetch the live web through ScraperAPI;
in the container ScraperAPI is replaced by a stand-in on 127.0.0.1:8089 that parks each request for the host. Both
are answered by ``python -m adapters.relay.host --preset claw``, so no key enters a container, and the tasks have no
network except the hosts a task's text cites (8 tasks link documents such as arXiv PDFs; those hosts are allowlisted).

Answer-key hygiene: ``task.yaml`` carries ``reference_solution``, ``judge_rubric``, ``scoring_components`` and
``expected_actions``, and ``grader.py`` carries expected values. Neither ever enters the container; only the task's
fixtures and the sanitized service and tool specs are staged, and the instruction is built from ``prompt.text`` alone.
The build aborts if a grader.py or task.yaml reaches a task directory.

    git clone https://github.com/claw-eval/claw-eval datasets/.claw-src     # commit 5680b8b; the judge uses it too
    uv run python -m adapters.claw_eval.build_tasks                 # all 300 tasks -> datasets/claw-eval
    uv run python -m adapters.claw_eval.build_tasks --only T001zh_email_triage
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

from adapters.common import DATASETS, ROOT, source_dir

FILES = Path(__file__).resolve().parent / "files"
QUEUE_SERVER = ROOT / "adapters" / "relay" / "queue_server.py"
SRC = source_dir("claw-src")  # set from --src in main()
OUT_DEFAULT = DATASETS / "claw-eval"
WEB_RELAY_URL = "http://127.0.0.1:8089/"
# Files upstream keeps on Hugging Face only (videos and large documents): a download of the dataset
# claw-eval/Claw-Eval laid out like the clone (tasks/<id>/fixtures/...), consulted where the clone lacks a file.
FIXTURES: Path | None = None


def _overlay(path: Path) -> Path | None:
    if FIXTURES is None:
        return None
    try:
        return FIXTURES / path.relative_to(SRC)
    except ValueError:
        return None


WEB_RELAY_SERVICE = {"name": "web_relay",
                     "command": "python3 /claw/queue_server.py --host 127.0.0.1 --port 8089 --http-get",
                     "port": 8089, "health_check": "http://127.0.0.1:8089/health", "health_check_method": "GET"}
DISPATCH_LOG = "/logs/verifier/dispatches.jsonl"

# Services that reach the open internet through ScraperAPI; in the container ScraperAPI is the stand-in relay.
WEB_SERVICES = {"web_real", "web_real_injection"}

# Grading material. It must never reach a container under any code path —
# grader.py hard-codes expected values and task.yaml carries
# reference_solution / judge_rubric / scoring_components.
_NEVER_STAGE = {"grader.py", "task.yaml"}

# Base layers are byte-identical across tasks so the image cache is hit
# after the first build; only the final COPY differs.
DOCKERFILE = """FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
        ffmpeg poppler-utils curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \\
        "fastapi>=0.115" "uvicorn>=0.30" "pydantic>=2.0" "pyyaml>=6.0" \\
        "httpx>=0.27" "requests>=2.31" "pypdf>=4.0" "Pillow>=10.0" \\
        "trafilatura>=1.6"

WORKDIR /workspace
COPY claw /claw
RUN chmod +x /claw/bootstrap.sh
"""

# 76 of 237 tasks declare no sandbox_files, so their environment/workspace/ is
# empty — and an empty directory does not survive into the build context, so a
# fixed `COPY workspace /workspace` fails the build ("no source files"). It
# reproduced on exactly those tasks, twice. Appended only when there is
# something to copy; the layers above stay byte-identical either way, so the
# image cache is still shared.
WORKSPACE_COPY = "COPY workspace /workspace\n"

BOOTSTRAP = """#!/bin/bash
# Start this task's mock services and wait until each is healthy.
# Idempotent: re-running is a no-op once the services answer.
set -uo pipefail
mkdir -p /logs/verifier
cd /claw
python3 - <<'PY'
import json, os, shlex, subprocess, time, urllib.request

services = json.load(open("/claw/services.json"))
if not services:
    print("no services for this task"); raise SystemExit(0)

def healthy(svc):
    req = urllib.request.Request(
        svc["health_check"], method=svc.get("health_check_method", "POST"),
        data=b"{}", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status < 500
    except urllib.error.HTTPError as e:
        return e.code < 500
    except Exception:
        return False

for svc in services:
    if healthy(svc):
        print(f"  {svc['name']}: already up"); continue
    env = dict(os.environ)
    env.update(svc.get("env") or {})
    env["CLAW_DISPATCH_LOG"] = %(log)r
    env["CLAW_SERVICE_PORT"] = str(svc["port"])
    env["CLAW_SERVICE_NAME"] = svc["name"]
    env.setdefault("ERROR_RATE", "0")
    for p in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(p, None)
    log = open(f"/logs/verifier/service_{svc['name']}.log", "ab")
    subprocess.Popen(shlex.split(svc["command"]), cwd="/claw",
                     stdout=log, stderr=log, env=env)

deadline = time.time() + 90
pending = [s for s in services if not healthy(s)]
while pending and time.time() < deadline:
    time.sleep(1)
    pending = [s for s in pending if not healthy(s)]
if pending:
    print("NOT READY: " + ", ".join(s["name"] for s in pending)); raise SystemExit(1)
print("services ready: " + ", ".join(s["name"] for s in services))
PY
""" % {"log": DISPATCH_LOG}

# Collects exactly what the graders read. Runs after the agent is done, so
# nothing here is ever visible to the agent.
TEST_SH = """#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier

# Files the grader needs inside the container (task.yaml `sandbox_grader_files`,
# e.g. verify_encoder.py). They are verification scripts, so they ship in
# tests/ — which harbor uploads only once the agent is done — and are put in
# place here, never in the image.
if [ -d /tests/grader_files ]; then
  (cd /tests/grader_files && find . -type f -print0 | while IFS= read -r -d '' f; do
     mkdir -p "/workspace/$(dirname "$f")" && cp "$f" "/workspace/$f"
   done)
fi

python3 - <<'PY'
import base64, json, mimetypes, subprocess, glob as globmod
from pathlib import Path

spec = json.load(open("/claw/snapshot_spec.json"))
snap = {}

for cmd in spec.get("commands", []):
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                           text=True, timeout=60)
        snap[f"cmd:{cmd}"] = {"exit_code": p.returncode,
                              "stdout": p.stdout, "stderr": p.stderr}
    except Exception as exc:
        snap[f"cmd:{cmd}"] = {"error": str(exc)}

TEXT_EXT = {".txt", ".md", ".json", ".csv", ".html", ".htm", ".xml", ".yaml",
            ".yml", ".py", ".js", ".css", ".log", ".tsv", ".srt", ".vtt"}

def read_one(path):
    p = Path(path)
    if not p.exists():
        return {"error": f"File not found: {p}"}
    mime, _ = mimetypes.guess_type(str(p))
    ext = p.suffix.lower()
    if ext in TEXT_EXT or (mime or "").startswith("text/"):
        return {"content": p.read_text(encoding="utf-8", errors="replace"),
                "mime_type": mime or "text/plain", "encoding": "utf-8"}
    return {"content": base64.b64encode(p.read_bytes()).decode("ascii"),
            "mime_type": mime or "application/octet-stream",
            "encoding": "base64", "size_bytes": p.stat().st_size}

for pattern in spec.get("files", []):
    if any(ch in pattern for ch in "*?"):
        for f in sorted(globmod.glob(pattern))[:50]:
            snap[f"file:{f}"] = read_one(f)
    else:
        snap[f"file:{pattern}"] = read_one(pattern)

Path("/logs/verifier/env_snapshot.json").write_text(
    json.dumps(snap, ensure_ascii=False))
print(f"env_snapshot: {len(snap)} entries")
PY

# Audit endpoints: the graders read per-service action ledgers from them.
python3 - <<'PY'
import json, urllib.request
from pathlib import Path
audits = {}
for svc in json.load(open("/claw/services.json")):
    url = svc["health_check"].rsplit("/", 1)[0] + "/audit"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            audits[svc["name"]] = json.loads(r.read().decode())
    except Exception as exc:
        audits[svc["name"]] = {"error": str(exc)}
Path("/logs/verifier/audit.json").write_text(json.dumps(audits, ensure_ascii=False))
print(f"audit: {len(audits)} service(s)")
PY

# The real score is computed on the host by judges/claw_eval.py.
echo "0" > /logs/verifier/reward.txt
"""

TASK_TOML = """schema_version = "1.1"

[task]
name = "claw-eval/{task_id}"
description = "Claw-Eval {task_id}: {task_name}"
authors = []
keywords = ["claw-eval", "agent", "{category}"]

[metadata]
difficulty = "{difficulty}"
category = "{category}"
tags = {tags}
source = "claw-eval/claw-eval"

[metadata.extra]
task_id = "{task_id}"
split = "{split}"
language = "{language}"
services = {services}
needs_web = {needs_web}
cited_hosts = {cited_hosts}
grading = "host-side judges/claw_eval.py running the task's own grader.py; reward = official task_score"

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = {agent_timeout}.0

[environment]
build_timeout_sec = 3600.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0
network_mode = "{network_mode}"{allowed_hosts}

[environment.env]
CLAW_TASK = "{task_id}"{serp_env}
"""


def split_of(task_id: str) -> str:
    return {"T": "general", "M": "multimodal", "C": "multi_turn"}.get(task_id[0], "other")


def build_instruction(spec: dict) -> str:
    """Instruction from prompt.text plus the mechanics our agent needs.

    Nothing evaluative goes in here — no rubric, no reference solution, no
    scoring weights. Only the request and how to reach the tools.
    """
    parts = [spec["prompt"]["text"].strip(), ""]

    tools = spec.get("tools") or []
    endpoints = {e["tool_name"]: e for e in (spec.get("tool_endpoints") or [])}
    if tools:
        parts += [
            "## Available tools",
            "",
            # Daytona replaces the image's CMD with its own supervisor, so the
            # services cannot be started by the image itself; the run has to
            # start them. Idempotent, so a retry costs nothing.
            "They are served by local processes that are not running yet. Start "
            "them once with `bash /claw/bootstrap.sh` (takes a few seconds, safe "
            "to re-run) before the first call.",
            "",
            "Each tool is a plain HTTP endpoint on localhost inside this "
            "container. Call them however you like (curl, python, …) — POST a "
            "JSON body matching the schema and read the JSON reply.",
            "",
        ]
        for t in tools:
            ep = endpoints.get(t["name"])
            if not ep:
                continue
            parts.append(f"### {t['name']}")
            if t.get("description"):
                parts.append(t["description"].strip())
            parts.append(f"`{ep.get('method', 'POST')} {ep['url']}`")
            schema = t.get("input_schema")
            if schema:
                parts.append("Request body schema:")
                parts.append("```json")
                parts.append(json.dumps(schema, ensure_ascii=False, indent=2))
                parts.append("```")
            parts.append("")

    files = spec.get("sandbox_files") or []
    if files:
        parts += ["## Files", ""]
        parts += [f"- `/workspace/{f}`" for f in files]
        parts.append("")

    if (spec.get("user_agent") or {}).get("enabled"):
        rounds = (spec.get("user_agent") or {}).get("max_rounds") or 8
        parts += [
            "## Talking to the person who asked",
            "",
            "This is a conversation, not a one-shot request — the person is "
            "reachable and does not volunteer everything up front. Ask them "
            "what you still need:",
            "",
            "```",
            "POST http://localhost:9199/user/message",
            '{"text": "your message to them"}',
            "```",
            "",
            f"They reply with `{{\"reply\": ..., \"rounds_left\": n}}`. You have "
            f"at most {rounds} exchanges; the first call also starts the "
            "conversation service, so allow a few seconds for it.",
            "",
        ]

    parts += [
        "## How this is graded",
        "",
        "Your **final message** is what gets read as your answer — put the "
        "complete response there, not only in a file. If the request asks you "
        "to produce or edit a file, do that as well and still summarise the "
        "outcome in your final message.",
    ]
    return "\n".join(parts).rstrip() + "\n"


def make_task(spec: dict, src_dir: Path, out_dir: Path, mock_services: Path) -> None:
    task_id = spec["task_id"]
    tdir = out_dir / task_id
    if tdir.exists():
        shutil.rmtree(tdir)
    (tdir / "environment" / "claw").mkdir(parents=True)
    (tdir / "environment" / "workspace").mkdir(parents=True)
    (tdir / "tests").mkdir()

    services = list(spec.get("services") or [])
    needs_web = any(s.get("name") in WEB_SERVICES for s in services)
    cited = cited_hosts(build_instruction(spec))
    hosts = cited

    # Multi-turn: the benchmark generates the user's turns from a persona with
    # Claude Opus 4.6. Our agent has to address the user explicitly, so the
    # simulator is a local endpoint; its model call goes out through the relay,
    # which the host answers.
    ua = spec.get("user_agent") or {}
    needs_user_agent = bool(ua.get("enabled"))
    if needs_user_agent:
        services += [
            {"name": "relay", "command": "python3 /claw/queue_server.py --port 9200",
             "port": 9200, "health_check": "http://localhost:9200/health",
             "health_check_method": "GET"},
            {"name": "user_agent", "command": "python3 /claw/user_agent_service.py",
             "port": 9199, "health_check": "http://localhost:9199/user/health",
             "health_check_method": "GET"},
        ]

    # --- container payload: mock services (patched to log dispatches) ---
    env_claw = tdir / "environment" / "claw"
    if services:
        shutil.copytree(mock_services, env_claw / "mock_services",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # --- fixtures the services read ---
    # Services run with cwd=/claw and address fixtures by repo-root-relative
    # paths, either explicitly via services[].env (which may point at ANOTHER
    # task's fixtures — the pinbench tasks reuse T021/T027 data) or implicitly
    # via their built-in default under tasks/<id>/fixtures. Stage exactly what
    # is referenced: copying the whole fixtures dir unconditionally duplicated
    # the video tasks' payload (M037 went to 564MB for a 280MB video).
    staged: set[str] = set()

    def stage(rel: str) -> bool:
        """Stage one repo-root-relative path (file or whole directory)."""
        src_path = SRC / rel
        alt = _overlay(src_path)
        if rel in staged:
            return True
        dst = env_claw / rel
        if src_path.name in _NEVER_STAGE:
            return False
        if src_path.is_file() or (not src_path.exists() and alt is not None and alt.is_file()):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path if src_path.is_file() else alt, dst)
        elif src_path.is_dir() or (alt is not None and alt.is_dir()):
            # Some services point their env at a *directory*, and a couple
            # (T097/T098) point at the whole task dir, which holds grader.py
            # and task.yaml. Those carry the expected answers, so they are
            # excluded here unconditionally rather than case by case.
            for base in (src_path, alt):
                if base is not None and base.is_dir():
                    shutil.copytree(base, dst, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns(
                                        "__pycache__", "*.pyc", "grader.py", "task.yaml"))
        else:
            return False
        staged.add(rel)
        return True

    for svc in services:
        env_vals = (svc.get("env") or {}).values()
        referenced = [v for v in env_vals
                      if isinstance(v, str) and ("fixtures/" in v or v.startswith("tasks/"))]
        for rel in referenced:
            stage(rel)
        if not referenced:
            # Built-in default path: tasks/<id>/fixtures/…
            own = src_dir / "fixtures"
            for base, root in ((own, SRC), (_overlay(own), FIXTURES)):
                if base is not None and base.is_dir():
                    for f in base.rglob("*"):
                        if f.is_file() and "__pycache__" not in f.parts:
                            stage(str(f.relative_to(root)))

    # --- grader-side files: staged into tests/, never into the image ---
    for rel in (spec.get("sandbox_grader_files") or []):
        src_file = SRC / rel if rel.startswith("tasks/") else src_dir / rel
        if not src_file.is_file() and _overlay(src_file) is not None:
            src_file = _overlay(src_file)
        if src_file.is_file():
            dst = tdir / "tests" / "grader_files" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)

    # --- files the agent is told about, at /workspace/<rel> verbatim ---
    for rel in (spec.get("sandbox_files") or []):
        # Repo-root-relative when it names a tasks/ path, else task-relative.
        src_file = SRC / rel if rel.startswith("tasks/") else src_dir / rel
        if not src_file.is_file() and _overlay(src_file) is not None:
            src_file = _overlay(src_file)
        if src_file.is_file():
            dst = tdir / "environment" / "workspace" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)

    if needs_web:
        # The ScraperAPI stand-in: web_real's searches and fetches are parked here
        # and performed by the host with its own key. Added after staging, so it
        # changes nothing about which fixtures are staged.
        services = [WEB_RELAY_SERVICE] + services

    # --- sanitized specs (no rubric / reference solution) ---
    (env_claw / "services.json").write_text(json.dumps([
        {k: s.get(k) for k in
         ("name", "command", "port", "health_check", "health_check_method", "env")
         if s.get(k) is not None}
        for s in services
    ], ensure_ascii=False, indent=1))
    (env_claw / "snapshot_spec.json").write_text(json.dumps({
        "commands": spec.get("env_snapshot_commands") or [],
        "files": spec.get("env_snapshot_files") or [],
    }, ensure_ascii=False, indent=1))
    if needs_user_agent or needs_web:
        shutil.copy2(QUEUE_SERVER, env_claw / "queue_server.py")
    if needs_user_agent:
        shutil.copy2(FILES / "user_agent_service.py", env_claw / "user_agent_service.py")
        (env_claw / "user_agent.json").write_text(json.dumps({
            "persona": ua.get("persona") or "",
            "max_rounds": ua.get("max_rounds") or 8,
            "language": (spec.get("prompt") or {}).get("language", "en"),
        }, ensure_ascii=False, indent=1))

    (env_claw / "bootstrap.sh").write_text(BOOTSTRAP)

    has_workspace = any(
        p.is_file() for p in (tdir / "environment" / "workspace").rglob("*"))
    (tdir / "environment" / "Dockerfile").write_text(
        DOCKERFILE + (WORKSPACE_COPY if has_workspace else ""))
    (tdir / "instruction.md").write_text(build_instruction(spec))
    (tdir / "tests" / "test.sh").write_text(TEST_SH)
    # Pointer only — the judge reads the rubric from the pristine source tree.
    (tdir / "tests" / "expected.json").write_text(json.dumps(
        {"task_id": task_id, "split": split_of(task_id)}, ensure_ascii=False, indent=1))

    env_cfg = spec.get("environment") or {}
    (tdir / "task.toml").write_text(TASK_TOML.format(
        task_id=task_id,
        task_name=(spec.get("task_name") or task_id).replace('"', "'"),
        category=spec.get("category", "agent"),
        difficulty=spec.get("difficulty", "medium"),
        tags=json.dumps((spec.get("tags") or []) + ["claw-eval"]),
        split=split_of(task_id),
        language=(spec.get("prompt") or {}).get("language", "en"),
        services=json.dumps([s.get("name") for s in services]),
        needs_web=str(needs_web).lower(),
        cited_hosts=json.dumps(cited),
        agent_timeout=int(env_cfg.get("timeout_seconds", 900)),
        # Only the hosts the task text cites are reachable; web_real goes through
        # the in-container stand-in, which the host relay answers.
        network_mode="allowlist" if hosts else "no-network",
        allowed_hosts=(f"\nallowed_hosts = {json.dumps(hosts)}" if hosts else ""),
        serp_env=(f'\nSERP_API_URL = "{WEB_RELAY_URL}"' if needs_web else ""),
    ))


_WEB_SERVER_EDITS = (
    ("serp_result = search_serp(query=req.query, num=num, timeout=20)",
     "serp_result = search_serp(query=req.query, num=num, timeout=90)"),
    ("timeout=min(req.timeout_seconds, 30),", "timeout=max(req.timeout_seconds, 90),"),
    # ScraperAPI hands PDFs back byte-for-byte. Upstream's HTML path turns them
    # into pages of mojibake that would burn the agent's context for nothing —
    # and search now surfaces PDFs (annual reports, benchmark decks) routinely.
    ("        else:\n"
     "            html = http_resp.text\n"
     "            content = _extract_content(html, req.url)\n"
     "            title = _extract_title(html)",
     "        else:\n"
     "            html = http_resp.text\n"
     "            if (http_resp.content[:5] == b\"%PDF\"\n"
     "                    or \"application/pdf\" in http_resp.headers.get(\n"
     "                        \"content-type\", \"\").lower()):\n"
     "                content, title = _extract_pdf(http_resp.content)\n"
     "            else:\n"
     "                content = _extract_content(html, req.url)\n"
     "                title = _extract_title(html)"),
    ("def _extract_title(html: str) -> str:",
     "def _extract_pdf(raw: bytes) -> tuple[str, str]:\n"
     "    \"\"\"Text of a PDF response, so the agent reads prose, not mojibake.\"\"\"\n"
     "    try:\n"
     "        import io\n"
     "\n"
     "        from pypdf import PdfReader\n"
     "\n"
     "        reader = PdfReader(io.BytesIO(raw))\n"
     "        pages = [p.extract_text() or \"\" for p in reader.pages[:40]]\n"
     "        title = str((reader.metadata or {}).get(\"/Title\") or \"\").strip()\n"
     "        text = \"\\n\\n\".join(t.strip() for t in pages if t.strip())\n"
     "        return text[:MAX_CONTENT_CHARS], title\n"
     "    except Exception as exc:  # noqa: BLE001\n"
     "        return f\"[PDF could not be parsed: {exc}]\", \"\"\n"
     "\n"
     "\n"
     "def _extract_title(html: str) -> str:"),
    ('"source": "",', '"source": item.get("source", ""),'),
    # Upstream caches every search for 24h, empty ones included, so a single
    # transient SERP hiccup poisons that query for the rest of the task.
    ("        _cache_set(cache_k, resp)",
     "        if results:\n            _cache_set(cache_k, resp)"),
)


# A handful of tasks hand the agent bare URLs and tell it to go read them
# ("download and read these two PDFs: https://arxiv.org/pdf/…"), while using no
# mock service at all. They are not web_real tasks, so they used to be generated
# as no-network — which makes them unwinnable: the agent can only draw the chart
# from memory. Found 2026-08-29 via M087, which spent its whole budget trying to
# reach arxiv (8 such tasks, 3 of them in the heldout). Hosts cited in the task
# text are opened for that task, and nothing else is.
_CITED_URL = re.compile(r"https?://(?!localhost|127\.)([a-z0-9.\-]+\.[a-z]{2,})", re.I)
_NOT_REAL = ("example.com", "example.org", "cloudpay.com", "company.example")


def cited_hosts(instruction: str) -> list[str]:
    body = instruction.split("## How this is graded")[0]
    hosts = {m.group(1).lower() for m in _CITED_URL.finditer(body)}
    hosts = {h for h in hosts if not any(h.endswith(x) for x in _NOT_REAL)}
    # arxiv serves PDFs from export.arxiv.org as often as from the bare host.
    if "arxiv.org" in hosts:
        hosts.add("export.arxiv.org")
    return sorted(hosts)


def patch_mock_services(src: Path, work: Path) -> Path:
    """Copy mock_services and wire in the dispatch logger (one edit, 19 services)."""
    dest = work / "mock_services"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src / "mock_services", dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(FILES / "dispatch_log.py", dest / "dispatch_log.py")
    # web_real fetches arbitrary URLs directly, which cannot work without a
    # network. ScraperAPI is a fetch proxy and the service already routes
    # *search* through it, so routing *fetch* the same way keeps the semantics
    # (the page's HTML comes back) while the sandbox only ever talks to the
    # local stand-in that the host relay answers.
    # The *search* half is swapped out wholesale (files/search_serp.py,
    # which documents why): upstream queries a vendor we have no key for and is
    # not on the allowlist, so every query came back silently empty — and with
    # nothing to fetch the agent falls back to guessing URLs from memory.
    for svc in sorted(WEB_SERVICES):
        ss = dest / svc / "search_serp.py"
        if not ss.is_file():
            continue
        assert "def search_serp(" in ss.read_text(), \
            f"{svc}/search_serp.py changed upstream — re-check the replacement"
        shutil.copy2(FILES / "search_serp.py", ss)

    for svc in sorted(WEB_SERVICES):
        wr = dest / svc / "server.py"
        if not wr.is_file():
            continue
        text = wr.read_text()
        # ScraperAPI renders the page before answering, so it routinely needs
        # 30-70s where a direct GET needed two. Both upstream caps sit below that.
        for anchor, repl in _WEB_SERVER_EDITS:
            assert anchor in text, f"{svc}/server.py changed upstream: {anchor[:50]!r}"
            text = text.replace(anchor, repl, 1)
        anchor = "            http_resp = client.get(req.url)"
        assert anchor in text, f"{svc}/server.py fetch shape changed upstream"
        wr.write_text(text.replace(anchor, (
            "            _relay = os.environ.get(\"SERP_API_URL\")\n"
            "            if _relay:\n"
            "                http_resp = client.get(_relay, params={\"url\": req.url})\n"
            "            else:\n"
            "                http_resp = client.get(req.url)"
        ), 1))

    base = dest / "_base.py"
    text = base.read_text()
    anchor = '    """Add error injection middleware to a FastAPI app."""\n    app.add_middleware(ErrorInjectionMiddleware)'
    assert anchor in text, "mock_services/_base.py changed upstream — re-check the hook"
    base.write_text(text.replace(anchor, anchor + (
        "\n    # Our addition: a shell-driven agent produces no ToolDispatch"
        "\n    # records, so each service logs its own requests instead."
        "\n    from mock_services.dispatch_log import add_dispatch_log"
        "\n    add_dispatch_log(app)"
    ), 1))
    return dest


def main() -> int:
    global SRC, FIXTURES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--src", type=Path, default=SRC, help="a clone of claw-eval/claw-eval")
    ap.add_argument("--offline-only", action="store_true",
                    help="skip the web_real tasks (they need the host relay with a ScraperAPI key)")
    ap.add_argument("--only", action="append", default=[],
                    help="generate just these task ids (repeatable)")
    ap.add_argument("--work", type=Path, default=source_dir("claw-build"), help="scratch dir for the patched mock services")
    ap.add_argument("--fixtures", type=Path, default=None,
                    help="download of the HF dataset claw-eval/Claw-Eval (tasks/<id>/fixtures/...): the videos and "
                         "large documents that the git repository omits")
    args = ap.parse_args()
    SRC, FIXTURES = args.src, args.fixtures
    if FIXTURES is None:
        print("warning: no --fixtures; tasks whose videos or documents live only on Hugging Face will lack them",
              file=sys.stderr)

    if not (SRC / "tasks").is_dir():
        raise SystemExit(
            f"missing source clone: git clone https://github.com/claw-eval/claw-eval {SRC}")

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    mock = patch_mock_services(SRC, work)

    args.out.mkdir(parents=True, exist_ok=True)
    made = skipped_web = 0
    for d in sorted((SRC / "tasks").iterdir()):
        y = d / "task.yaml"
        if not y.is_file():
            continue
        spec = yaml.safe_load(y.read_text())
        task_id = spec["task_id"]
        if args.only and task_id not in args.only:
            continue
        needs_web = any((s.get("name") in WEB_SERVICES)
                        for s in (spec.get("services") or []))
        if needs_web and args.offline_only:
            skipped_web += 1
            continue
        make_task(spec, d, args.out, mock)
        made += 1
        if args.limit and made >= args.limit:
            break

    leaked = sorted(
        str(p.relative_to(args.out))
        for name in _NEVER_STAGE
        for p in args.out.rglob(name)
    )
    if leaked:
        raise SystemExit(
            "grading material reached the task payload; aborted (do not use these tasks):\n  "
            + "\n  ".join(leaked[:20])
        )

    print(f"{made} tasks -> {args.out}")
    print("answer-key check: no grader.py or task.yaml in any task payload")
    if skipped_web:
        print(f"skipped {skipped_web} web_real tasks (--offline-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
