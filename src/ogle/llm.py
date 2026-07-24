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
    anthropic                               -> Anthropic cloud fallback, default model
    anthropic:claude-opus-4-8               -> Anthropic, that model
    anthropic:claude-haiku-4-5@https://host -> that model against a specific base URL

The **Anthropic cloud fallback** is the escalation lane for when no local Ollama is
running (the README's feature #2 promises "local Ollama or Anthropic fallback"). It reads
the key from ``ANTHROPIC_API_KEY`` at call time; if the key is absent the call raises and
:func:`ogle.narrative.narrate` degrades to the deterministic markdown exactly as it does
for an unreachable Ollama host — a missing key never suppresses the alert.

Only stdlib is used (``urllib``) — no new dependency, and every adapter degrades to the
deterministic report the moment its model backend is unreachable.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Callable

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:latest"
DEFAULT_TIMEOUT = 60.0

# Anthropic cloud fallback. Default to the cheapest current model — the narrator only
# *rewords* facts Ogle already computed (grounded prompt, forbidden from inventing
# severity), so a small model is enough, the same rationale as the local-Ollama default.
DEFAULT_ANTHROPIC_HOST = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 1024

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


def _anthropic_text(payload: dict) -> str:
    """Concatenate the text blocks of a Messages-API response.

    The response ``content`` is a list of blocks; only ``type == "text"`` blocks carry
    prose. Strip ``<think>`` defensively too (Anthropic won't emit it, but the same
    reasoning-leak guard the Ollama path uses costs nothing here).
    """
    blocks = payload.get("content") or []
    parts = [
        b.get("text", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ]
    return _strip_think("".join(parts))


def anthropic_narrator(
    model: str = DEFAULT_ANTHROPIC_MODEL,
    api_key: str | None = None,
    host: str = DEFAULT_ANTHROPIC_HOST,
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: int = ANTHROPIC_MAX_TOKENS,
) -> Callable[[str], str]:
    """Return a callable that phrases a prompt via the Anthropic Messages API.

    The callable POSTs to ``/v1/messages`` and returns the model's text. The key is read
    from ``api_key`` or ``$ANTHROPIC_API_KEY`` **at call time**; a missing key (or any
    transport/HTTP error) raises — deliberately, so :func:`ogle.narrative.narrate` falls
    back to the deterministic markdown and the alert still goes out.
    """
    endpoint = host.rstrip("/") + "/v1/messages"

    def _call(prompt: str) -> str:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        body = json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_VERSION,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed host)
            payload = json.loads(resp.read().decode("utf-8"))
        return _anthropic_text(payload)

    return _call


def build_narrator(spec: str) -> Callable[[str], str]:
    """Parse a spec string into a narrator callable.

    ``ollama`` / ``ollama:<model>`` / ``anthropic`` / ``anthropic:<model>`` — any form
    optionally suffixed with ``@<host>`` to point at a specific base URL. Raises
    :class:`ValueError` on an unknown provider so the CLI can surface a clean usage error
    instead of a traceback.
    """
    spec = (spec or "").strip()
    host_override: str | None = None
    if "@" in spec:
        spec, host_override = spec.split("@", 1)
        host_override = host_override.strip() or None

    if spec in ("", "ollama"):
        return ollama_narrator(host=host_override or DEFAULT_OLLAMA_HOST)
    if spec.startswith("ollama:"):
        model = spec[len("ollama:") :].strip() or DEFAULT_OLLAMA_MODEL
        return ollama_narrator(model=model, host=host_override or DEFAULT_OLLAMA_HOST)
    if spec == "anthropic":
        return anthropic_narrator(host=host_override or DEFAULT_ANTHROPIC_HOST)
    if spec.startswith("anthropic:"):
        model = spec[len("anthropic:") :].strip() or DEFAULT_ANTHROPIC_MODEL
        return anthropic_narrator(
            model=model, host=host_override or DEFAULT_ANTHROPIC_HOST
        )

    raise ValueError(
        f"unknown narrator spec {spec!r} — expected 'ollama', 'ollama:<model>', "
        "'anthropic', or 'anthropic:<model>'"
    )
