"""Unit tests for ogle.llm — the optional LLM narrator adapters.

No network: the Ollama HTTP call is exercised by monkeypatching ``urllib.request.urlopen``
with a fake response. The point is (1) the request is shaped right, (2) thinking blocks are
stripped, and (3) a transport failure propagates so ``narrate`` can fall back.
"""

import io
import json

import pytest

from ogle import llm
from ogle.narrative import narrate, render_markdown, build_incident
from ogle.scorer import DriftFinding, DriftKind, Severity


# ---- spec parsing --------------------------------------------------------------------
def test_build_narrator_defaults_to_local_ollama():
    n = llm.build_narrator("ollama")
    assert callable(n)


def test_build_narrator_bare_and_empty_are_default():
    assert callable(llm.build_narrator(""))
    assert callable(llm.build_narrator("ollama"))


def test_build_narrator_rejects_unknown_provider():
    with pytest.raises(ValueError):
        llm.build_narrator("gpt-9")


# ---- request shaping (monkeypatched transport) ---------------------------------------
class _FakeResp:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture_urlopen(monkeypatch, payload):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(payload)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_ollama_narrator_posts_prompt_and_returns_response(monkeypatch):
    captured = _capture_urlopen(monkeypatch, {"response": "  table X row count fell 40%.  "})
    n = llm.build_narrator("ollama:qwen3:30b@http://box:11434")
    out = n("PROMPT-TEXT")

    assert out == "table X row count fell 40%."           # trimmed
    assert captured["url"] == "http://box:11434/api/generate"
    assert captured["method"] == "POST"
    assert captured["body"]["model"] == "qwen3:30b"
    assert captured["body"]["prompt"] == "PROMPT-TEXT"
    assert captured["body"]["stream"] is False
    assert captured["body"]["think"] is False


def test_ollama_narrator_strips_thinking_block(monkeypatch):
    _capture_urlopen(
        monkeypatch,
        {"response": "<think>let me reason about severity</think>\nSchema drift on customers."},
    )
    out = llm.build_narrator("ollama")("P")
    assert "think" not in out.lower()
    assert out == "Schema drift on customers."


# ---- graceful fallback through narrate() ---------------------------------------------
def _finding():
    return DriftFinding(
        urn="urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)",
        kind=DriftKind.VOLUME,
        severity=Severity.HIGH,
        message="row count dropped 40%",
    )


def test_narrate_falls_back_when_transport_raises(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", boom)
    findings = [_finding()]
    out = narrate(findings, llm=llm.build_narrator("ollama"))
    # narrate swallows the error and returns the deterministic report.
    assert out == render_markdown(build_incident(findings))
