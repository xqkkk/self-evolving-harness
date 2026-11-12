# are-sidecar image, reshaped from the adapter template for one reason: the
# official build does an anonymous huggingface snapshot_download of the 248MB
# gaia2_filesystem, and at job scale (dozens of concurrent DinD builds behind
# ONE daytona egress IP) HF rate-limits the IP and every build dies. Two
# mitigations: an optional HF_TOKEN build arg (authenticated requests get real
# limits; put HF_TOKEN in .env), and retry with backoff so anonymous builds
# ride out the window. Everything else matches the Harbor template (Apache-2.0).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1
WORKDIR /opt/are-sidecar

RUN pip install --no-cache-dir meta-agents-research-environments==1.2.0

RUN pip install --no-cache-dir fastmcp

ARG HF_TOKEN=
RUN python - <<'PY'
import os
import time

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None
last = None
for attempt in range(6):
    try:
        snapshot_download(
            repo_id="meta-agents-research-environments/gaia2_filesystem",
            repo_type="dataset",
            local_dir="/opt/gaia2-filesystem",
            allow_patterns=["demo_filesystem/*", "demo_filesystem/**"],
            token=token,
        )
        break
    except Exception as e:  # HF rate limiting surfaces as several exception types; back off on all
        last = e
        wait = 45 * (attempt + 1)
        print(f"snapshot_download attempt {attempt + 1} failed: {e}; retry in {wait}s")
        time.sleep(wait)
else:
    raise SystemExit(f"gaia2_filesystem download failed after retries: {last}")
PY

ENV DEMO_FS_PATH=/opt/gaia2-filesystem/demo_filesystem
ENV SCENARIO_PATH=/opt/gaia2-task/scenario.json
ENV PORT=8000

COPY are_mcp_server.py .

EXPOSE 8000

CMD ["python", "are_mcp_server.py"]
