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
    NotifyError,
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
def _completed(returncode):
    """A stand-in for subprocess.CompletedProcess exposing just `.returncode`."""
    import types

    return types.SimpleNamespace(returncode=returncode)


def test_command_notifier_shells_and_passes_stdin(monkeypatch):
    calls = {}

    def _fake_run(cmd, *, input, text, check):
        calls["cmd"] = cmd
        calls["input"] = input
        calls["check"] = check
        return _completed(0)

    monkeypatch.setattr("ogle.watch.subprocess.run", _fake_run)
    notify = make_command_notifier(["mail", "-s", "drift"])
    notify("the page body")
    assert calls["cmd"] == ["mail", "-s", "drift"]
    assert calls["input"] == "the page body"
    # check=False: we inspect the return code ourselves so a failing pager becomes a
    # NotifyError (visible) rather than a raised CalledProcessError (crashes the tick).
    assert calls["check"] is False


def test_command_notifier_raises_on_nonzero_exit(monkeypatch):
    """A pager that exits non-zero must raise NotifyError, not silently 'succeed'."""
    monkeypatch.setattr("ogle.watch.subprocess.run", lambda *a, **k: _completed(3))
    notify = make_command_notifier(["pager", "--broken"])
    with pytest.raises(NotifyError) as ei:
        notify("body")
    assert "exited 3" in str(ei.value)


def test_command_notifier_raises_when_unspawnable(monkeypatch):
    """A command that can't even start (OSError) is the classic misconfigured-pager trap."""

    def _boom(*a, **k):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr("ogle.watch.subprocess.run", _boom)
    notify = make_command_notifier(["does-not-exist"])
    with pytest.raises(NotifyError) as ei:
        notify("body")
    assert "could not run notifier" in str(ei.value)


# ---- page delivery failure is surfaced, never silently dropped --------------------
def _raiser(exc):
    def _notify(_text):
        raise exc

    return _notify


def test_incident_page_delivery_failure_falls_back_to_stderr(capsys):
    """A notifier that fails must NOT report a clean page: page_delivered=False, the page
    text still lands on stderr, and the reason is captured — not swallowed."""
    out = run_tick(
        [],
        notifier=_raiser(NotifyError("notifier ['pager'] exited 3")),
        run_check=_runner(EXIT_INCIDENT, stdout="HIGH drift on customers"),
    )
    assert out.action == "page"
    assert out.paged is True             # a page was DISPATCHED
    assert out.page_delivered is False   # …but it did NOT get through
    assert "exited 3" in out.delivery_error
    err = capsys.readouterr().err
    assert "PAGE DELIVERY FAILED" in err
    assert "HIGH drift on customers" in err  # the page text was not lost


def test_unexpected_notifier_exception_is_contained(capsys):
    """Even a non-NotifyError from a custom notifier must fall back, not crash the tick."""
    out = run_tick(
        [],
        notifier=_raiser(ValueError("kaboom")),
        run_check=_runner(EXIT_INCIDENT, stdout="drift story"),
    )
    assert out.page_delivered is False
    assert "ValueError" in out.delivery_error
    assert "drift story" in capsys.readouterr().err


def test_successful_delivery_marks_page_delivered():
    """The happy path: a working notifier -> page_delivered True, no delivery_error."""
    note = _Recorder()
    out = run_tick(
        [], notifier=note, run_check=_runner(EXIT_INCIDENT, stdout="drift")
    )
    assert out.paged is True
    assert out.page_delivered is True
    assert out.delivery_error == ""


def test_watch_subcommand_warns_on_delivery_failure(tmp_path, capsys, monkeypatch):
    """End-to-end: `ogle watch --notify-cmd <broken>` on a real incident preserves the
    check's exit code (1) but prints PAGE DELIVERY FAILED to stderr."""
    # First run seeds the baseline (exit 0); a second run with a changed signature fires
    # a NEW incident (exit 1). Simpler: force the incident via a fake check runner is not
    # reachable through the CLI, so drive real drift.
    monkeypatch.setattr("ogle.watch.subprocess.run", lambda *a, **k: _completed(127))
    store = tmp_path / "store.json"
    sig1 = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig1])
    # seed
    main(["watch", "--", "--store", str(store), "--signatures", str(sigs)])
    # drift: drop a column -> HIGH schema drift -> new incident -> page attempted
    sig2 = build_signature(CUSTOMERS_URN, schema_fields=[], row_count=10)
    _write_sigs(tmp_path / "s.json", [sig2])
    capsys.readouterr()  # clear
    code = main(
        [
            "watch",
            "--notify-cmd",
            "does-not-matter",
            "--",
            "--store",
            str(store),
            "--signatures",
            str(sigs),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1                     # check contract preserved
    assert "PAGED" in captured.out
    assert "PAGE DELIVERY FAILED" in captured.err


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
