"""What the harness knows about the run, beyond relaying commands.

Two things the model cannot see for itself and keeps losing track of: how the
world has actually changed under it, and what it decided to do. `changed_files`
counts files created or modified since the run began, so a "you have been at
this for fifteen minutes and touched nothing" situation becomes visible
instead of felt; `render_plan` turns the model's own step list into the short
line that rides along on every tool result, which is what makes a plan survive
the context trims that drop the turns it was made in. Both are observations,
not judgements: the harness reports, the model decides.
"""

MARKER = "/tmp/.harness_t0"

# Where a task usually leaves its work. Deliberately broad and deliberately
# shallow (`-xdev`, one walk, no contents read): the point is a count, not an
# inventory, and a task that writes somewhere else is simply not counted.
_PROBE_ROOTS = ". /app /root /home /workspace /work /srv"

_PROBE = (
    f"test -e {MARKER} || echo nomarker; "
    "( for r in " + _PROBE_ROOTS + "; do "
    '[ -d "$r" ] && timeout 10 find "$r" -xdev -type f -newer ' + MARKER + " -print 2>/dev/null; done ) "
    "| sort -u | grep -v '^/tmp/' | wc -l"
)


async def mark_start(world):
    """Drop the timestamp everything else is measured against."""
    try:
        await world.exec(f"touch {MARKER}", timeout_s=15)
    except Exception:
        pass            # the probe notices and reports nothing rather than 0


async def changed_files(world):
    """Files created or modified since `mark_start`, or None if unknowable.

    A missing marker must not read as "nothing changed", so it is told apart
    from a genuine zero; a probe that errors or times out is simply omitted
    from the note rather than turning into a number the model would trust.
    """
    try:
        r = await world.exec(_PROBE, timeout_s=30)
    except Exception:
        return None
    out = r.stdout or ""
    if "nomarker" in out:      # the probe's own zero is not the world's zero
        return None
    if r.return_code not in (0, None) and not any(
            l.strip().isdigit() for l in out.splitlines()):
        return None
    for line in reversed(out.splitlines()):
        if line.strip().isdigit():
            n = int(line)
            return f"{n}+" if n >= 999 else str(n)
    return None


# One look at the container, taken before the first model turn. The first
# thing nearly every run does by hand is `ls`, `pwd`, and `which` — a whole
# model turn spent discovering facts the harness could have gathered in a
# second. Presence on PATH is a lookup, not an execution, so the wide list is
# cheap; versions are asked of the few that matter. Kept small enough to ride
# in the system prompt, and worded as one look rather than the whole picture.
_SURVEY_TOOLS = ("python3 pip3 pip uv node npm npx tsc go cargo rustc gcc g++ cc make cmake "
                 "git curl wget jq sqlite3 psql mysql R Rscript java javac mvn gradle docker "
                 "podman ffmpeg convert magick tesseract pandoc pdftotext unzip tar aws gs")
_SURVEY_VERSIONS = ("python3 gcc git node go rustc R")
SURVEY = (
    'echo "cwd: $(pwd 2>/dev/null)"; echo "user: $(id -un 2>/dev/null)"; '
    "head -2 /etc/os-release 2>/dev/null; "
    'echo "--- on PATH"; for c in ' + _SURVEY_TOOLS + '; do '
    'command -v "$c" >/dev/null 2>&1 && printf "%s " "$c"; done; echo; '
    'echo "--- versions"; for c in ' + _SURVEY_VERSIONS + '; do '
    'command -v "$c" >/dev/null 2>&1 && echo "$c: $("$c" --version 2>&1 | head -1)"; done; '
    'echo "--- machine"; echo "cpus: $(nproc 2>/dev/null)"; '
    'free -m 2>/dev/null | sed -n 2p; df -h / 2>/dev/null | tail -1; '
    'echo "--- here"; ls -la 2>/dev/null | head -25; '
    'echo "--- /app"; ls -la /app 2>/dev/null | head -25; '
    'echo "--- roots"; for d in /app /root /home /workspace /work /srv /data /mnt /opt; do '
    '[ -d "$d" ] && echo "$d: $(ls -m "$d" 2>/dev/null | head -c 300)"; done'
)


async def survey(world, limit=4500):
    """Gather the container facts a first turn would otherwise spend itself on.

    Returns None rather than something partial or wrong: a survey that could
    not run is worth less than no survey, and the model is going to look
    anyway.
    """
    try:
        r = await world.exec(SURVEY, timeout_s=40)
    except Exception:
        return None
    text = ((r.stdout or "").strip() or (r.stderr or "").strip())
    if not text or r.return_code not in (0, None):
        return None
    if len(text) > limit:
        text = text[:limit] + "\n... [survey truncated]"
    return text


def _title(step):
    t = str(step.get("title") or "").strip()
    return (t[:70] + "…") if len(t) > 70 else t


def render_plan(steps):
    """The plan as one line: how far through, what is under hand, what is next.

    Kept short on purpose — it is appended to every tool result, so it has to
    earn its tokens. An empty or mangled plan renders as nothing at all.
    """
    if not isinstance(steps, list) or not steps:
        return ""
    clean = [s for s in steps if isinstance(s, dict) and _title(s)]
    if not clean:
        return ""
    done = sum(1 for s in clean if str(s.get("status") or "") == "done")
    now = next((_title(s) for s in clean if str(s.get("status") or "") == "in_progress"), "")
    if not now:
        now = _title(clean[done]) if done < len(clean) else ""
    return f"plan {done}/{len(clean)}" + (f": {now}" if now else "")
