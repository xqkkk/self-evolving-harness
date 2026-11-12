"""Resolve a model alias to a client instance.

Aliases live in a JSON file: ``$HARNESS_PROVIDERS`` if set, else ``providers.json`` in the
repository root (see ``providers.example.json``). An entry names its client class and its settings::

    "gpt-5.6-sol-high": {
        "kind": "model_clients.capped_responses:CappedResponsesModel",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "max_output_tokens": 65536,
        "max_input_tokens": 258400
    }

Credentials never go into the file: ``api_key_env`` names the environment variable that holds the
key, and the value is read here, on the host, when the client is built. ``kind`` is either
``"responses"`` (the plain client in this package) or ``"<module>:<Class>"``.
"""

import importlib
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_providers() -> dict:
    path = Path(os.environ.get("HARNESS_PROVIDERS", _ROOT / "providers.json"))
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def resolve_model(model_name: str, recorder, max_tokens: int | None = None):
    providers = _load_providers()
    entry = providers.get(model_name)
    if entry is None:
        raise ValueError(f"unknown model {model_name!r}: add an entry to the providers file "
                         f"(known: {sorted(providers) or 'none'})")
    params = {k: v for k, v in entry.items() if k not in ("kind", "comment")}
    key_env = params.pop("api_key_env", None)
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise ValueError(f"model {model_name!r} needs the environment variable {key_env}")
        params["api_key"] = key

    kind = entry.get("kind", "responses")
    if kind == "responses":
        from .responses_model import ResponsesModel

        params.pop("max_input_tokens", None)
        if max_tokens:
            params["max_output_tokens"] = max_tokens
        return ResponsesModel(recorder, **params)
    if ":" in kind:
        module_name, _, class_name = kind.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        if max_tokens:
            params["max_tokens"] = max_tokens
        return cls(recorder, **params)
    raise ValueError(f"unknown provider kind {kind!r} for model {model_name!r}")
