"""Unit tests for ogle.watch — the scheduler-facing single-tick wrapper.

Covers the exit-code contract dispatch (0/1/2 -> ok/page/error), that the notifier is
invoked exactly once and only on a page, that page-on-error is opt-in, that stdout is
captured into the page text (and doesn't leak), and that the `ogle watch` subcommand
PRESERVES the underlying check's exit code so a scheduler can still branch on it.

All tests inject a fake check-runner so nothing here needs the datahub SDK, Docker, or a
real GMS — the watch layer is pure glue over `ogle check`'s contract.
"""

import json

import pytest

from ogle.cli import build_parser, main
from ogle.signature import build_signature
from ogle.watch import (
    EXIT_ERROR,
    EXIT_HEALTHY,
    EXIT_INCIDENT,
    make_command_notifier,
    run_tick,
)

CUSTOMERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"


class _Recorder:
    """A notifier that records every page it receives."""

    def __init__(self):
        self.pages = []

    def __call__(self, text):
        self.pages.append(text)


def _runner(code, *, stdout="", stderr=""):
    """Build a fake check-runner that emits `stdout`/`stderr` and returns `code`."""

    def _run(argv):
        import sys

        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        return code

    return _run


# ---- exit-code contract dispatch --------------------------------------------------
def test_healthy_is_quiet():
    note = _Recorder()
    out = run_tick([], notifier=note, run_check=_runner(EXIT_HEALTHY, stdout="all clear"))
    assert out.action == "ok"
    assert out.paged is False
    assert note.pages == []
    assert out.code == 0
    assert out.report_text == "all clear"


def test_incident_pages_once_with_narrative():
    note = _Recorder()
    out = run_tick(
        ["--signatures", "s.json"],
        notifier=note,
        run_check=_runner(EXIT_INCIDENT, stdout="HIGH drift on customers [serving]"),
    )
    assert out.action == "page"
    assert out.paged is True
    assert note.pages == ["HIGH drift on customers [serving]"]
    assert out.code == 1


def test_incident_with_empty_stdout_still_pages_a_fallback():
    note = _Recorder()
    out = run_tick([], notifier=note, run_check=_runner(EXIT_INCIDENT, stdout=""))
    assert out.paged is True
    assert len(note.pages) == 1
    assert "incident" in note.pages[0].lower()


def test_error_does_not_page_by_default():
    note = _Recorder()
    out = run_tick(
        [], notifier=note, run_check=_runner(EXIT_ERROR, stderr="ogle check: bad input")
    )
    assert out.action == "error"
    assert out.paged is False
    assert note.pages == []
    assert out.code == 2
    assert out.error_text == "ogle check: bad input"


def test_error_pages_when_opted_in():
    note = _Recorder()
    out = run_tick(
        [],
        notifier=note,
        page_on_error=True,
        run_check=_runner(EXIT_ERROR, stderr="live walk failed: connection refused"),
    )
    assert out.action == "error"
    assert out.paged is True
    assert note.pages == ["live walk failed: connection refused"]


def test_stdout_is_captured_not_leaked(capsys):
    """The check's narrative must ride in report_text, not leak to the real stdout."""
    note = _Recorder()
    run_tick([], notifier=note, run_check=_runner(EXIT_HEALTHY, stdout="noisy narrative"))
    captured = capsys.readouterr()
    assert "noisy narrative" not in captured.out


def test_unexpected_code_is_treated_as_quiet():
    """A non-contract exit code (e.g. 3) must not page — fail safe, don't cry wolf."""
    note = _Recorder()
    out = run_tick([], notifier=note, run_check=_runner(3))
    assert out.action == "ok"
    assert out.paged is False


def test_check_args_are_forwarded():
    seen = {}

    def _run(argv):
        seen["argv"] = list(argv)
        return EXIT_HEALTHY

    run_tick(["--signatures", "s.json", "--no-update"], notifier=_Recorder(), run_check=_run)
    assert seen["argv"] == ["--signatures", "s.json", "--no-update"]


# ---- command notifier -------------------------------------------------------------
def test_command_notifier_shells_and_passes_stdin(monkeypatch):
    calls = {}

    def _fake_run(cmd, *, input, text, check):
        calls["cmd"] = cmd
        calls["input"] = input
        calls["check"] = check

    monkeypatch.setattr("ogle.watch.subprocess.run", _fake_run)
    notify = make_command_notifier(["mail", "-s", "drift"])
    notify("the page body")
    assert calls["cmd"] == ["mail", "-s", "drift"]
    assert calls["input"] == "the page body"
    # check=False so a failing pager never crashes the watch tick.
    assert calls["check"] is False


# ---- `ogle watch` subcommand: exit code is preserved end-to-end -------------------
def _write_sigs(path, sigs, serving=None):
    payload = {"signatures": [s.to_dict() for s in sigs]}
    if serving is not None:
        payload["serving_urns"] = list(serving)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_watch_subcommand_healthy_exit_code(tmp_path, capsys):
    """First sighting seeds baselines -> exit 0; watch must pass that through."""
    sig = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig])
    store = tmp_path / "store.json"
    code = main(["watch", "--", "--store", str(store), "--signatures", str(sigs)])
    assert code == 0
    assert "healthy" in capsys.readouterr().out


def test_watch_subcommand_input_error_exit_code(tmp_path, capsys):
    """A missing signatures file -> check exits 2 -> watch preserves 2, prints error status."""
    store = tmp_path / "store.json"
    code = main(
        [
            "watch",
            "--",
            "--store",
            str(store),
            "--signatures",
            str(tmp_path / "nope.json"),
        ]
    )
    assert code == 2
    assert "check error" in capsys.readouterr().out


def test_watch_parser_registered():
    parser = build_parser()
    ns = parser.parse_args(["watch", "--page-on-error", "--", "--signatures", "s.json"])
    assert ns.page_on_error is True
    assert ns.check_args == ["--signatures", "s.json"]
