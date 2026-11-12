"""Load a harness workspace (a directory of plain files) as an importable package."""

import hashlib
import importlib.util
import itertools
import sys
import types
from pathlib import Path

_SKIP_PARTS = {".git", "__pycache__"}
_load_counter = itertools.count()


def workspace_hash(ws: Path) -> str:
    """Content hash of a workspace (file names and bytes). The kernel stamps it into every trial, so
    a run record can always be traced back to the exact harness that produced it."""
    ws = Path(ws)
    h = hashlib.sha256()
    for p in sorted(ws.rglob("*")):
        if p.is_file() and not (_SKIP_PARTS & set(p.parts)):
            h.update(p.relative_to(ws).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:12]


def load_workspace(ws: Path):
    """Import ``<ws>/harness.py`` as a fresh module copy per call.

    Sibling modules in the workspace import each other relatively: ``from . import x``.
    """
    ws = Path(ws).resolve()
    harness_path = ws / "harness.py"
    if not harness_path.is_file():
        raise FileNotFoundError(f"no harness.py in workspace: {ws}")

    pkg_name = f"harness_ws_{workspace_hash(ws)}_{next(_load_counter)}"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ws)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg

    mod_name = f"{pkg_name}.harness"
    spec = importlib.util.spec_from_file_location(mod_name, harness_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        sys.modules.pop(pkg_name, None)
        raise
    return module
