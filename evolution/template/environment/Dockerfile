FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pytest==8.4.1 pytest-json-ctrf==0.3.5

WORKDIR /app
COPY app/ /app/

# Baseline commit: the verifier diffs against this to check that changes are
# non-empty and confined to harness_workspace/.
RUN printf '__pycache__/\n*.pyc\n' > /app/.gitignore \
 && git -C /app init -q -b main \
 && git -C /app -c user.email=evolve@harness_kernel -c user.name=harness_kernel add -A \
 && git -C /app -c user.email=evolve@harness_kernel -c user.name=harness_kernel commit -qm baseline
