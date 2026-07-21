"""ogle watch — the scheduler-facing wrapper around a single `ogle check` tick.

Ogle's job is to run on a schedule (cron / a Windows Scheduled Task / an OpenClaw
Task) and page a human the *first* time drift appears — not on every tick. The pieces
that make that possible already live in the pipeline:

  * `ogle check` returns a strict **exit-code contract** — 0 healthy, 1 a NEW incident
    fired, 2 usage/input error;
  * a repeated *unchanged* drift **debounces to 0**, so a standing incident is paged
    once, not every run.

`ogle watch` is the thin glue that turns that contract into an action a scheduler can
lean on. One tick:

    run `ogle check <args>`  ->  read its exit code + captured narrative
        0  -> quiet OK      (log the tick, no page)
        1  -> PAGE          (hand the narrative to the notifier, exactly once)
        2  -> ERROR         (log; page only if --page-on-error / page_on_error=True)

The loop itself belongs to the scheduler (cron line / Task trigger), which is why this
is a single tick, not a `while True`. The notifier is injected so the public repo stays
free of any one messaging backend: the default prints a `PAGE:`-prefixed block to stderr,
and `--notify-cmd` shells an arbitrary command with the narrative on stdin — that is where
an operator wires their own pager (e.g. an OpenClaw `message send`).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

EXIT_HEALTHY = 0
EXIT_INCIDENT = 1
EXIT_ERROR = 2

# A notifier receives the already-rendered page text and delivers it somewhere.
Notifier = Callable[[str], None]
# A check-runner takes the argv AFTER the "check" token and returns an exit code.
CheckRunner = Callable[[Sequence[str]], int]


class NotifyError(RuntimeError):
    """A notifier could not deliver a page (bad exit code or an un-spawnable command).

    Raising — rather than swallowing — is the whole point: a page that never reached a
    human is a production failure at least as bad as the drift it was meant to report, so
    the tick must be able to SEE the delivery failure instead of reporting a false "PAGED".
    """


@dataclass
class TickOutcome:
    """The result of one watch tick — everything a scheduler or test needs to branch."""

    code: int          # the underlying `ogle check` exit code (contract is preserved)
    action: str        # "ok" | "page" | "error"
    paged: bool        # whether a page was DISPATCHED this tick (attempted)
    report_text: str   # narrative captured from stdout (the human-facing drift story)
    error_text: str    # anything the check wrote to stderr (input/live-walk failures)
    page_delivered: bool = True  # did the notifier actually succeed? (False on delivery failure)
    delivery_error: str = ""     # why delivery failed, when it did


def _default_check_runner(argv: Sequence[str]) -> int:
    """Run `ogle check <argv>` in-process via the CLI, returning its exit code.

    Imported lazily to avoid an import cycle (cli imports nothing from watch, but keeping
    the import local also lets tests swap this out without importing the CLI at all).
    """
    from .cli import main as cli_main

    return cli_main(["check", *argv])


def _stderr_notifier(text: str) -> None:
    """Default pager: a clearly-marked block on stderr, so a bare cron line still shows it."""
    sys.stderr.write("PAGE: ogle drift incident\n" + text + "\n")


def make_command_notifier(notify_cmd: Sequence[str]) -> Notifier:
    """Build a notifier that shells `notify_cmd`, handing the page text on stdin.

    This is the seam an operator uses to reach a real pager without Ogle depending on it:
    e.g. `--notify-cmd openclaw message send --channel whatsapp --target +1555…`.

    Delivery is verified: a command that can't be spawned (`OSError`, e.g. the pager binary
    isn't on PATH) or that exits non-zero raises `NotifyError`. `run_tick` catches that and
    falls back to a loud stderr page, so a broken pager surfaces as a *visible* delivery
    failure instead of a silently-dropped alert reported as "PAGED".
    """
    cmd = list(notify_cmd)

    def _notify(text: str) -> None:
        try:
            proc = subprocess.run(cmd, input=text, text=True, check=False)
        except OSError as exc:
            # Un-spawnable command (not found, not executable, …) — the classic silent-page
            # trap when a pager is misconfigured on a headless scheduler.
            raise NotifyError(f"could not run notifier {cmd!r}: {exc}") from exc
        if proc.returncode != 0:
            raise NotifyError(f"notifier {cmd!r} exited {proc.returncode}")

    return _notify


def _deliver(notifier: Notifier, text: str) -> "tuple[bool, str]":
    """Invoke `notifier`, guaranteeing the page lands *somewhere* even if it fails.

    Returns `(delivered, reason)`. On any notifier failure — a `NotifyError` from the
    command notifier, or an unexpected exception from a custom one — the page text is still
    written to stderr via the default notifier so it is never silently lost, and `delivered`
    is False with a human-readable reason.
    """
    try:
        notifier(text)
        return True, ""
    except NotifyError as exc:
        reason = str(exc)
    except Exception as exc:  # a custom notifier misbehaving must not swallow the page
        reason = f"notifier raised {type(exc).__name__}: {exc}"
    sys.stderr.write(f"PAGE DELIVERY FAILED ({reason}) — falling back to stderr:\n")
    _stderr_notifier(text)
    return False, reason


def run_tick(
    check_args: Sequence[str],
    *,
    notifier: Optional[Notifier] = None,
    page_on_error: bool = False,
    run_check: Optional[CheckRunner] = None,
) -> TickOutcome:
    """Run one `ogle check` and dispatch on its exit-code contract.

    `check_args` are the flags passed through to `ogle check` (e.g.
    `["--signatures", "sigs.json"]` or `["--gms", url, "--discover", "--write-back"]`).
    stdout (the narrative) and stderr (errors) are captured so the notifier can carry the
    story, and so the check's own console noise doesn't leak through the scheduler.
    """
    notifier = notifier or _stderr_notifier
    run_check = run_check or _default_check_runner

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = run_check(list(check_args))
    report_text = out.getvalue().rstrip()
    error_text = err.getvalue().rstrip()

    if code == EXIT_INCIDENT:
        delivered, derr = _deliver(
            notifier, report_text or "ogle: a new drift incident fired (no narrative captured)"
        )
        return TickOutcome(code, "page", True, report_text, error_text, delivered, derr)

    if code == EXIT_ERROR:
        if page_on_error:
            delivered, derr = _deliver(notifier, error_text or "ogle check failed (exit 2)")
            return TickOutcome(code, "error", True, report_text, error_text, delivered, derr)
        return TickOutcome(code, "error", False, report_text, error_text)

    # Healthy (0) or any unexpected non-contract code: treat as quiet, don't page.
    return TickOutcome(code, "ok", False, report_text, error_text)


def build_watch_args(parser_sub) -> None:
    """Register the `watch` subparser on an existing subparsers object (called from cli)."""
    watch = parser_sub.add_parser(
        "watch",
        help="One scheduler tick: run `ogle check`, page once on a NEW incident.",
        description=(
            "Run a single `ogle check` and act on its exit-code contract: page on a new "
            "incident (exit 1), stay quiet when healthy (0). Put this on a cron line or "
            "Scheduled Task; the pipeline debounces standing drift so you're paged once "
            "per incident, not every tick."
        ),
    )
    watch.add_argument(
        "--notify-cmd",
        nargs="*",
        metavar="ARG",
        help="Shell this command on a page, passing the narrative on stdin "
        "(e.g. --notify-cmd mail -s 'ogle drift' you@host). Default: print to stderr.",
    )
    watch.add_argument(
        "--page-on-error",
        action="store_true",
        help="Also page when the check exits 2 (input/live-walk failure). "
        "Default: log the error but do not page.",
    )
    watch.add_argument(
        "--json",
        action="store_true",
        help="Emit the tick outcome as JSON on stdout instead of the human status line "
        "(exit code preserved). Carries the delivery_failed signal a scheduler can gate on.",
    )
    # Everything after `--` is forwarded verbatim to `ogle check`.
    watch.add_argument(
        "check_args",
        nargs="*",
        metavar="CHECK_ARG",
        help="Args forwarded to `ogle check` (put them after `--`), "
        "e.g. `ogle watch -- --signatures sigs.json --serving URN`.",
    )
    watch.set_defaults(func=cmd_watch)


def cmd_watch(args) -> int:
    """CLI handler: run one tick, print a one-line status, and PRESERVE the check's exit code."""
    notifier = make_command_notifier(args.notify_cmd) if args.notify_cmd else None
    outcome = run_tick(
        args.check_args,
        notifier=notifier,
        page_on_error=args.page_on_error,
    )
    # A dispatched-but-undelivered page is an operational failure the scheduler must see:
    # the exit code still honors the `ogle check` contract, but a broken pager must not hide
    # behind a green "PAGED" status. In JSON mode the delivery_failed field carries it (stdout,
    # machine-readable); in human mode a loud stderr line does. Computed once, surfaced either way.
    delivery_failed = outcome.paged and not outcome.page_delivered

    if getattr(args, "json", False):
        # Structured twin of the human line for a scheduler/monitor wrapping `ogle watch`:
        # the exit-code contract folded into exit_rc, plus the delivery_failed signal that
        # otherwise only lives as scraped stderr text — so a wrapper can gate on a silently
        # dropped page (paged but never delivered) without parsing prose. The narrative and
        # captured stderr ride along so a JSON consumer can forward the drift story itself,
        # the same text the notifier would have carried. exit_rc mirrors `status --json`'s
        # folded verdict: it survives when the process exit code is lost over a log/message bus.
        sys.stdout.write(
            json.dumps(
                {
                    "watch": {
                        "action": outcome.action,      # ok | page | error
                        "exit_rc": outcome.code,       # the preserved `ogle check` exit code
                        "paged": outcome.paged,        # a page was dispatched this tick
                        "page_delivered": outcome.page_delivered,
                        "delivery_error": outcome.delivery_error,
                        # Folded production-failure signal: dispatched but never delivered.
                        # nonempty delivery_error iff this is true; distinct from a healthy
                        # tick that never paged (paged False → delivery_failed False).
                        "delivery_failed": delivery_failed,
                        "report_text": outcome.report_text,
                        "error_text": outcome.error_text,
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return outcome.code

    # A single machine-greppable status line for the scheduler's own logs.
    status = {"ok": "healthy", "page": "PAGED (new incident)", "error": "check error"}[
        outcome.action
    ]
    sys.stdout.write(f"ogle watch: {status} (exit {outcome.code})\n")
    if delivery_failed:
        sys.stderr.write(
            f"ogle watch: PAGE DELIVERY FAILED — {outcome.delivery_error}\n"
        )
    return outcome.code


__all__ = [
    "EXIT_HEALTHY",
    "EXIT_INCIDENT",
    "EXIT_ERROR",
    "NotifyError",
    "TickOutcome",
    "run_tick",
    "make_command_notifier",
    "build_watch_args",
    "cmd_watch",
]
