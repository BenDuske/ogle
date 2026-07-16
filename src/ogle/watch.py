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


@dataclass
class TickOutcome:
    """The result of one watch tick — everything a scheduler or test needs to branch."""

    code: int          # the underlying `ogle check` exit code (contract is preserved)
    action: str        # "ok" | "page" | "error"
    paged: bool        # whether the notifier was actually invoked this tick
    report_text: str   # narrative captured from stdout (the human-facing drift story)
    error_text: str    # anything the check wrote to stderr (input/live-walk failures)


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
    """

    def _notify(text: str) -> None:
        subprocess.run(list(notify_cmd), input=text, text=True, check=False)

    return _notify


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
        notifier(report_text or "ogle: a new drift incident fired (no narrative captured)")
        return TickOutcome(code, "page", True, report_text, error_text)

    if code == EXIT_ERROR:
        if page_on_error:
            notifier(error_text or "ogle check failed (exit 2)")
            return TickOutcome(code, "error", True, report_text, error_text)
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
    # A single machine-greppable status line for the scheduler's own logs.
    status = {"ok": "healthy", "page": "PAGED (new incident)", "error": "check error"}[
        outcome.action
    ]
    sys.stdout.write(f"ogle watch: {status} (exit {outcome.code})\n")
    return outcome.code


__all__ = [
    "EXIT_HEALTHY",
    "EXIT_INCIDENT",
    "EXIT_ERROR",
    "TickOutcome",
    "run_tick",
    "make_command_notifier",
    "build_watch_args",
    "cmd_watch",
]
