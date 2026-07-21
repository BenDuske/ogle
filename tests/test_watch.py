"""Unit tests for ogle.watch — the scheduler-facing single-tick wrapper.

Covers the exit-code contract dispatch (0/1/2 -> ok/page/error), that the notifier is
invoked exactly once and only on a page, that page-on-error is opt-in, that stdout is
captured into the page text (and doesn't leak), and that the `ogle watch` subcommand
PRESERVES the underlying check's exit code so a scheduler can still branch on it.

All tests inject a fake check-runner so nothing here needs the datahub SDK, Docker, or a
real GMS — the watch layer is pure glue over `ogle check`'s contract.
"""

import json
import subprocess

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

    def _fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls.update(kwargs)
        return _completed(0)

    monkeypatch.setattr("ogle.watch.subprocess.run", _fake_run)
    notify = make_command_notifier(["mail", "-s", "drift"])
    notify("the page body")
    assert calls["cmd"] == ["mail", "-s", "drift"]
    assert calls["input"] == "the page body"
    # check=False: we inspect the return code ourselves so a failing pager becomes a
    # NotifyError (visible) rather than a raised CalledProcessError (crashes the tick).
    assert calls["check"] is False
    # stdin is encoded UTF-8, not the process locale, so an emoji-bearing narrative doesn't
    # crash the pipe on a cp1252 Windows console.
    assert calls["encoding"] == "utf-8"


def test_command_notifier_pipes_emoji_narrative_without_crashing():
    """Regression: 🔴 severity markers in the narrative must survive the stdin pipe.

    Real subprocess (no mock): a locale-encoded pipe would raise UnicodeEncodeError under
    a cp1252 console — the bug the live smoke caught. `sys.executable -c pass` is a portable
    no-op pager that still forces the stdin encode to happen.
    """
    import sys

    notify = make_command_notifier([sys.executable, "-c", "import sys; sys.stdin.read()"])
    notify("🔴 HIGH drift on customers 🟠")  # must not raise


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


def test_watch_json_flag_parsed():
    parser = build_parser()
    ns = parser.parse_args(["watch", "--json", "--", "--signatures", "s.json"])
    assert ns.json is True
    # default off, so a bare `watch` keeps the human line
    ns2 = parser.parse_args(["watch", "--", "--signatures", "s.json"])
    assert ns2.json is False


# ---- `ogle watch --json`: structured tick outcome for a scheduler -----------------
def test_watch_json_healthy_tick(tmp_path, capsys):
    """A healthy tick emits JSON (not the human line) with exit_rc 0 and nothing paged."""
    sig = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig])
    store = tmp_path / "store.json"
    code = main(["watch", "--json", "--", "--store", str(store), "--signatures", str(sigs)])
    out = capsys.readouterr().out
    assert code == 0
    assert "ogle watch: healthy" not in out  # human line suppressed in JSON mode
    payload = json.loads(out)["watch"]
    assert payload["action"] == "ok"
    assert payload["exit_rc"] == 0
    assert payload["paged"] is False
    assert payload["delivery_failed"] is False
    assert payload["delivery_error"] == ""


def test_watch_json_carries_delivery_failed_on_broken_pager(tmp_path, capsys, monkeypatch):
    """A real incident with an un-spawnable pager: exit_rc 1, paged True, delivery_failed True,
    and a non-empty delivery_error — the silently-dropped-page signal, machine-readable."""
    monkeypatch.setattr("ogle.watch.subprocess.run", lambda *a, **k: _completed(127))
    store = tmp_path / "store.json"
    sig1 = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig1])
    # seed the baseline (exit 0)
    main(["watch", "--json", "--", "--store", str(store), "--signatures", str(sigs)])
    # drop a column -> HIGH schema drift -> new incident -> page attempted
    sig2 = build_signature(CUSTOMERS_URN, schema_fields=[], row_count=10)
    _write_sigs(tmp_path / "s.json", [sig2])
    capsys.readouterr()  # clear
    code = main(
        [
            "watch",
            "--json",
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
    assert code == 1
    payload = json.loads(captured.out)["watch"]
    assert payload["action"] == "page"
    assert payload["exit_rc"] == 1
    assert payload["paged"] is True
    assert payload["page_delivered"] is False
    assert payload["delivery_failed"] is True
    assert payload["delivery_error"]  # non-empty reason
    # the page still fell back to a loud stderr block (never silently lost)
    assert "PAGE" in captured.err


# ---- `ogle watch --dry-run`: decide-but-suppress, validate wiring without paging ---
def test_dry_run_incident_decides_but_never_delivers():
    """A real incident under --dry-run: the paging DECISION stands (would_page True) but no
    page is dispatched (paged False) and the notifier is never touched."""
    note = _Recorder()
    out = run_tick(
        ["--signatures", "s.json"],
        notifier=note,
        dry_run=True,
        run_check=_runner(EXIT_INCIDENT, stdout="HIGH drift on customers [serving]"),
    )
    assert out.action == "page"
    assert out.would_page is True     # a page WAS warranted
    assert out.paged is False         # ...but suppressed
    assert note.pages == []           # notifier never invoked
    assert out.code == 1              # exit contract preserved
    assert out.report_text == "HIGH drift on customers [serving]"


def test_dry_run_healthy_would_not_page():
    """A clean tick: nothing to page whether or not --dry-run is set."""
    note = _Recorder()
    out = run_tick([], notifier=note, dry_run=True, run_check=_runner(EXIT_HEALTHY))
    assert out.action == "ok"
    assert out.would_page is False
    assert out.paged is False
    assert note.pages == []


def test_dry_run_error_with_page_on_error_would_page_but_suppresses():
    """--dry-run also suppresses the opt-in error page while still recording it would fire."""
    note = _Recorder()
    out = run_tick(
        [],
        notifier=note,
        page_on_error=True,
        dry_run=True,
        run_check=_runner(EXIT_ERROR, stderr="live walk failed"),
    )
    assert out.action == "error"
    assert out.would_page is True
    assert out.paged is False
    assert note.pages == []
    assert out.code == 2


def test_normal_run_would_page_mirrors_paged():
    """Invariant: outside --dry-run, would_page tracks paged exactly (decision == dispatch)."""
    note = _Recorder()
    out = run_tick([], notifier=note, run_check=_runner(EXIT_INCIDENT, stdout="drift"))
    assert out.would_page is True and out.paged is True
    clean = run_tick([], notifier=_Recorder(), run_check=_runner(EXIT_HEALTHY))
    assert clean.would_page is False and clean.paged is False


def test_watch_dry_run_flag_parsed():
    parser = build_parser()
    ns = parser.parse_args(["watch", "--dry-run", "--", "--signatures", "s.json"])
    assert ns.dry_run is True
    ns2 = parser.parse_args(["watch", "--", "--signatures", "s.json"])
    assert ns2.dry_run is False


def test_watch_dry_run_json_reports_would_page_without_delivery(tmp_path, capsys):
    """End-to-end: real drift under `--dry-run --json` -> exit_rc 1, would_page True, paged
    False, dry_run True, and NO page block on stderr (nothing was delivered)."""
    store = tmp_path / "store.json"
    sig1 = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig1])
    main(["watch", "--", "--store", str(store), "--signatures", str(sigs)])  # seed
    sig2 = build_signature(CUSTOMERS_URN, schema_fields=[], row_count=10)     # drop a column
    _write_sigs(tmp_path / "s.json", [sig2])
    capsys.readouterr()  # clear
    code = main(
        ["watch", "--dry-run", "--json", "--", "--store", str(store), "--signatures", str(sigs)]
    )
    captured = capsys.readouterr()
    assert code == 1
    payload = json.loads(captured.out)["watch"]
    assert payload["action"] == "page"
    assert payload["exit_rc"] == 1
    assert payload["would_page"] is True
    assert payload["paged"] is False
    assert payload["dry_run"] is True
    assert payload["delivery_failed"] is False
    assert "PAGE" not in captured.err  # nothing delivered, nothing fell back to stderr


# ---- `ogle watch --notify-retries`: ride out a transient pager outage ------------
class _FlakyNotifier:
    """A notifier that raises NotifyError for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, text):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise NotifyError(f"transient outage (attempt {self.calls})")
        # succeeds on the next call


def test_retry_recovers_from_transient_failure(capsys):
    """A pager that fails twice then works: with 2 retries the page is delivered, not dropped."""
    slept = []
    note = _FlakyNotifier(fail_times=2)
    out = run_tick(
        [],
        notifier=note,
        notify_retries=2,
        sleeper=slept.append,          # record backoff waits instead of really sleeping
        run_check=_runner(EXIT_INCIDENT, stdout="HIGH drift on customers"),
    )
    assert out.paged is True
    assert out.page_delivered is True      # recovered on the 3rd attempt
    assert out.delivery_error == ""
    assert note.calls == 3                 # 1 initial + 2 retries
    assert slept == [1, 2]                 # linear backoff between the three attempts
    assert "PAGE DELIVERY FAILED" not in capsys.readouterr().err


def test_retry_exhausted_still_falls_back(capsys):
    """When every retry also fails, delivery_failed stands and the reason names the attempts."""
    slept = []
    note = _FlakyNotifier(fail_times=99)   # never recovers
    out = run_tick(
        [],
        notifier=note,
        notify_retries=2,
        sleeper=slept.append,
        run_check=_runner(EXIT_INCIDENT, stdout="HIGH drift"),
    )
    assert out.page_delivered is False
    assert note.calls == 3                 # exhausted 1 + 2 retries
    assert "after 3 attempts" in out.delivery_error
    assert "PAGE DELIVERY FAILED" in capsys.readouterr().err


def test_no_retries_is_single_attempt_unchanged(capsys):
    """Default (0 retries): one attempt, no backoff, reason NOT annotated with a count."""
    slept = []
    note = _FlakyNotifier(fail_times=99)
    out = run_tick(
        [],
        notifier=note,
        notify_retries=0,
        sleeper=slept.append,
        run_check=_runner(EXIT_INCIDENT, stdout="drift"),
    )
    assert out.page_delivered is False
    assert note.calls == 1
    assert slept == []                     # no backoff on a single attempt
    assert "attempts" not in out.delivery_error  # single-try reason stays clean


def test_retry_does_not_retry_a_programming_error(capsys):
    """A non-NotifyError (a bug in a custom notifier) must NOT be retried — it won't self-heal."""
    calls = {"n": 0}

    def _buggy(_text):
        calls["n"] += 1
        raise ValueError("kaboom")

    out = run_tick(
        [],
        notifier=_buggy,
        notify_retries=5,
        sleeper=lambda _s: None,
        run_check=_runner(EXIT_INCIDENT, stdout="drift"),
    )
    assert out.page_delivered is False
    assert calls["n"] == 1                  # stopped after the first crash, no retry storm
    assert "ValueError" in out.delivery_error
    assert "attempts" not in out.delivery_error  # only one attempt was made


def test_watch_notify_retries_flag_parsed():
    parser = build_parser()
    ns = parser.parse_args(
        ["watch", "--notify-retries", "3", "--", "--signatures", "s.json"]
    )
    assert ns.notify_retries == 3
    ns2 = parser.parse_args(["watch", "--", "--signatures", "s.json"])
    assert ns2.notify_retries == 0  # default off


def test_watch_negative_retries_rejected(tmp_path, capsys):
    """A negative --notify-retries is a usage error (exit 2), not a silent no-op."""
    store = tmp_path / "store.json"
    sig = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig])
    code = main(
        [
            "watch",
            "--notify-retries",
            "-1",
            "--",
            "--store",
            str(store),
            "--signatures",
            str(sigs),
        ]
    )
    assert code == EXIT_ERROR
    assert "--notify-retries must be >= 0" in capsys.readouterr().err


# ---- --notify-timeout: a hung pager must not wedge the tick -----------------------
def test_command_notifier_passes_timeout_to_subprocess(monkeypatch):
    """When a timeout is set it is handed to subprocess.run so the child is actually bounded."""
    seen = {}

    def _fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return _completed(0)

    monkeypatch.setattr("ogle.watch.subprocess.run", _fake_run)
    notify = make_command_notifier(["pager"], timeout=5.0)
    notify("body")
    assert seen["timeout"] == 5.0


def test_command_notifier_no_timeout_omits_the_kwarg(monkeypatch):
    """Default (no timeout): subprocess.run is called WITHOUT a timeout kwarg (wait forever)."""
    seen = {}

    def _fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return _completed(0)

    monkeypatch.setattr("ogle.watch.subprocess.run", _fake_run)
    notify = make_command_notifier(["pager"])
    notify("body")
    assert "timeout" not in seen  # unchanged call shape when the feature is off


def test_command_notifier_timeout_raises_notifyerror(monkeypatch):
    """A hung pager (TimeoutExpired) becomes a NotifyError — a transient, retryable failure."""

    def _hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr("ogle.watch.subprocess.run", _hang)
    notify = make_command_notifier(["pager", "--slow"], timeout=2.0)
    with pytest.raises(NotifyError) as ei:
        notify("body")
    assert "timed out after 2.0s" in str(ei.value)


def test_timeout_failure_is_retried_then_recovers(capsys):
    """A timed-out delivery is a NotifyError, so --notify-retries rides it out and recovers."""
    calls = {"n": 0}

    def _notify(_text):
        calls["n"] += 1
        if calls["n"] == 1:
            # first attempt "hangs" and is surfaced as the timeout NotifyError
            raise NotifyError("notifier ['pager'] timed out after 2.0s")
        # second attempt lands

    slept = []
    out = run_tick(
        [],
        notifier=_notify,
        notify_retries=1,
        sleeper=slept.append,
        run_check=_runner(EXIT_INCIDENT, stdout="HIGH drift"),
    )
    assert out.page_delivered is True   # recovered after the timeout blip
    assert calls["n"] == 2
    assert slept == [1]
    assert "PAGE DELIVERY FAILED" not in capsys.readouterr().err


def test_watch_notify_timeout_flag_parsed():
    parser = build_parser()
    ns = parser.parse_args(
        ["watch", "--notify-timeout", "3.5", "--", "--signatures", "s.json"]
    )
    assert ns.notify_timeout == 3.5
    ns2 = parser.parse_args(["watch", "--", "--signatures", "s.json"])
    assert ns2.notify_timeout is None  # default: no limit


def test_watch_nonpositive_timeout_rejected(tmp_path, capsys):
    """A zero/negative --notify-timeout is a usage error (exit 2), not a silent no-op."""
    store = tmp_path / "store.json"
    sig = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig])
    code = main(
        [
            "watch",
            "--notify-cmd",
            "true",
            "--notify-timeout",
            "0",
            "--",
            "--store",
            str(store),
            "--signatures",
            str(sigs),
        ]
    )
    assert code == EXIT_ERROR
    assert "--notify-timeout must be > 0" in capsys.readouterr().err


def test_watch_dry_run_human_line_says_would_page(tmp_path, capsys):
    """Human status under --dry-run on a real incident reads WOULD PAGE, not PAGED."""
    store = tmp_path / "store.json"
    sig1 = build_signature(CUSTOMERS_URN, schema_fields=[("id", "int")], row_count=10)
    sigs = _write_sigs(tmp_path / "s.json", [sig1])
    main(["watch", "--", "--store", str(store), "--signatures", str(sigs)])  # seed
    sig2 = build_signature(CUSTOMERS_URN, schema_fields=[], row_count=10)
    _write_sigs(tmp_path / "s.json", [sig2])
    capsys.readouterr()
    code = main(["watch", "--dry-run", "--", "--store", str(store), "--signatures", str(sigs)])
    out = capsys.readouterr().out
    assert code == 1
    assert "WOULD PAGE (dry-run" in out
    assert "PAGED (new incident)" not in out
