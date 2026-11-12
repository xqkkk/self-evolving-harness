"""Small helpers shared by the benchmark builders."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"


def source_dir(name: str) -> Path:
    """Default location of an official clone or download, e.g. ``datasets/.claw-src`` (the judges use the same)."""
    return DATASETS / f".{name}"


def toml_str(s: str) -> str:
    """A TOML basic string (JSON string syntax is valid TOML)."""
    return json.dumps(s, ensure_ascii=False)


def toml_list(items) -> str:
    return "[" + ", ".join(toml_str(str(i)) for i in items) + "]"


def write_exec(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def env(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if required and not value:
        raise SystemExit(f"{name} is not set (put it in .env)")
    return value


def load_dotenv() -> None:
    from evolution.common import load_dotenv as _load

    _load()


def create_snapshot(name: str, image, *, cpu: int, memory: int, disk: int, key_var: str = "DAYTONA_API_KEY") -> None:
    """Create (or replace) a Daytona snapshot that every task sandbox of a benchmark starts from.

    A snapshot belongs to one Daytona organization; build it once per organization you run in.
    """
    from daytona import CreateSnapshotParams, Daytona, DaytonaConfig, Resources

    daytona = Daytona(DaytonaConfig(api_key=env(key_var)))
    try:
        existing = daytona.snapshot.get(name)
    except Exception:  # noqa: BLE001 - "not found" is reported as an exception
        existing = None
    if existing is not None:
        print(f"deleting the existing snapshot {name} ({existing.state})", flush=True)
        daytona.snapshot.delete(existing)
        for _ in range(60):  # deletion is asynchronous
            time.sleep(5)
            try:
                daytona.snapshot.get(name)
            except Exception:  # noqa: BLE001
                break
        else:
            raise SystemExit(f"snapshot {name} was not deleted within 5 minutes")
    print(f"building snapshot {name}", flush=True)
    daytona.snapshot.create(
        CreateSnapshotParams(name=name, image=image, resources=Resources(cpu=cpu, memory=memory, disk=disk)),
        on_logs=lambda line: print("  " + line.rstrip(), flush=True),
    )
    print(f"snapshot ready: {name}  (run with --ek snapshot_template_name={name})", flush=True)
