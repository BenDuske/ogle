"""Optional LLM narrator adapters for :func:`ogle.narrative.narrate`.

The narrative layer takes any ``Callable[[str], str]`` as its ``llm=`` argument and is
fully graceful — if the callable raises, ``narrate`` falls back to the deterministic
markdown, so an alert always goes out. This module ships the concrete callables so the
LLM-phrased incident summary (feature #2 in the README) is actually reachable from the CLI.

**Default is local Ollama** (``ollama:qwen3:latest``) because Ogle's whole point is to run
unattended on the same box as the model: no API key, no cloud dependency, free per call.
The prompt :func:`ogle.narrative.build_llm_prompt` hands the model is *grounded* — it only
rewords facts Ogle already computed and is forbidden from inventing severity — so a small
local model is enough.

Adapters are built from a compact spec string so the CLI can expose a single flag::

    ollama                                  -> ollama, default model + host
    ollama:qwen3:30b                        -> ollama, that model
    ollama:qwen3:latest@http://box:11434    -> that model on a specific host

Only stdlib is used (``urllib``) — no new dependency, and the adapter degrades to the
deterministic report the moment the model host is unreachable.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:latest"
DEFAULT_TIMEOUT = 60.0

# Thinking models (qwen3, deepseek-r1, ...) can wrap chain-of-thought in <think>…</think>.
# We ask Ollama not to emit it (``think: False``) but strip it defensively too, so a
# reasoning block never leaks into an on-call alert.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def ollama_narrator(
    model: str = DEFAULT_OLLAMA_MODEL,
    host: str = DEFAULT_OLLAMA_HOST,
    timeout: float = DEFAULT_TIMEOUT,
) -> Callable[[str], str]:
    """Return a callable that phrases a prompt via a local Ollama server.

    The callable POSTs to ``/api/generate`` (non-streaming) and returns the model's text.
    It raises on any transport/HTTP error — that is deliberate: :func:`ogle.narrative.narrate`
    catches it and falls back to the deterministic markdown, so a model outage never
    suppresses the alert.
    """
    endpoint = host.rstrip("/") + "/api/generate"

    def _call(prompt: str) -> str:
        body = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "think": False}
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
            payload = json.loads(resp.read().decode("utf-8"))
        return _strip_think(payload.get("response") or "")

    return _call


def build_narrator(spec: str) -> Callable[[str], str]:
    """Parse a spec string into a narrator callable.

    ``ollama`` / ``ollama:<model>`` / ``<...>@<host>``. Raises :class:`ValueError` on an
    unknown provider so the CLI can surface a clean usage error instead of a traceback.
    """
    spec = (spec or "").strip()
    host = DEFAULT_OLLAMA_HOST
    if "@" in spec:
        spec, host = spec.split("@", 1)
        host = host.strip() or DEFAULT_OLLAMA_HOST

    if spec in ("", "ollama"):
        return ollama_narrator(host=host)
    if spec.startswith("ollama:"):
        model = spec[len("ollama:") :].strip() or DEFAULT_OLLAMA_MODEL
        return ollama_narrator(model=model, host=host)

    raise ValueError(
        f"unknown narrator spec {spec!r} — expected 'ollama' or 'ollama:<model>'"
    )
