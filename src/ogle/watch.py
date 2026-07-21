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
import time
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
# A sleeper waits `seconds` between delivery retries (injected so tests don't really sleep).
Sleeper = Callable[[float], None]


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
    would_page: bool = False     # would this tick page a human? (the paging DECISION, decoupled
    #                              from dispatch — in a normal run would_page == paged; under
    #                              --dry-run a page is decided (would_page True) but suppressed
    #                              (paged False) so the wiring can be validated silently)


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


def _deliver(
    notifier: Notifier,
    text: str,
    *,
    retries: int = 0,
    sleeper: Sleeper = time.sleep,
) -> "tuple[bool, str]":
    """Invoke `notifier`, guaranteeing the page lands *somewhere* even if it fails.

    Returns `(delivered, reason)`. On any notifier failure — a `NotifyError` from the
    command notifier, or an unexpected exception from a custom one — the page text is still
    written to stderr via the default notifier so it is never silently lost, and `delivered`
    is False with a human-readable reason.

    `retries` re-attempts delivery after a `NotifyError`, with a linear backoff
    (`sleeper(1)`, `sleeper(2)`, …) between tries. A transient pager blip (a momentary 5xx,
    a DNS hiccup) shouldn't become a *permanent* dropped page — the module's whole thesis is
    that an undelivered page is a production failure, so a recoverable one deserves a retry.
    A non-`NotifyError` exception is a bug in a custom notifier, not a transient outage, so it
    is NOT retried — retrying a `ValueError` just repeats the same crash. The fallback stderr
    page still fires once, after the last attempt, and the reason names the attempts made.
    """
    attempts_made = 0
    reason = ""
    for attempt in range(retries + 1):
        attempts_made = attempt + 1
        try:
            notifier(text)
            return True, ""
        except NotifyError as exc:
            reason = str(exc)
        except Exception as exc:  # a custom notifier misbehaving must not swallow the page
            reason = f"notifier raised {type(exc).__name__}: {exc}"
            break  # a programming error won't self-heal on retry — stop and fall back now
        if attempt < retries:
            sleeper(attempt + 1)  # linear backoff before the next delivery attempt
    if attempts_made > 1:
        reason = f"{reason} (after {attempts_made} attempts)"
    sys.stderr.write(f"PAGE DELIVERY FAILED ({reason}) — falling back to stderr:\n")
    _stderr_notifier(text)
    return False, reason


def run_tick(
    check_args: Sequence[str],
    *,
    notifier: Optional[Notifier] = None,
    page_on_error: bool = False,
    dry_run: bool = False,
    notify_retries: int = 0,
    sleeper: Sleeper = time.sleep,
    run_check: Optional[CheckRunner] = None,
) -> TickOutcome:
    """Run one `ogle check` and dispatch on its exit-code contract.

    `check_args` are the flags passed through to `ogle check` (e.g.
    `["--signatures", "sigs.json"]` or `["--gms", url, "--discover", "--write-back"]`).
    stdout (the narrative) and stderr (errors) are captured so the notifier can carry the
    story, and so the check's own console noise doesn't leak through the scheduler.

    `dry_run` decides-but-suppresses: the check still runs and the paging decision is still
    made (`would_page`), but the notifier is never invoked, so an operator can validate a new
    watch cron line — does it fire? what narrative would it carry? — without paging a human.

    `notify_retries` re-attempts a failed delivery that many times before falling back to a
    loud stderr page, so a transient pager outage doesn't become a permanently dropped alert.
    """
    notifier = notifier or _stderr_notifier
    run_check = run_check or _default_check_runner

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = run_check(list(check_args))
    report_text = out.getvalue().rstrip()
    error_text = err.getvalue().rstrip()

    if code == EXIT_INCIDENT:
        if dry_run:
            # A page WOULD fire — but suppress delivery so validating the wiring is silent.
            return TickOutcome(code, "page", False, report_text, error_text, would_page=True)
        delivered, derr = _deliver(
            notifier,
            report_text or "ogle: a new drift incident fired (no narrative captured)",
            retries=notify_retries,
            sleeper=sleeper,
        )
        return TickOutcome(
            code, "page", True, report_text, error_text, delivered, derr, would_page=True
        )

    if code == EXIT_ERROR:
        if page_on_error:
            if dry_run:
                return TickOutcome(code, "error", False, report_text, error_text, would_page=True)
            delivered, derr = _deliver(
                notifier,
                error_text or "ogle check failed (exit 2)",
                retries=notify_retries,
                sleeper=sleeper,
            )
            return TickOutcome(
                code, "error", True, report_text, error_text, delivered, derr, would_page=True
            )
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
        "--dry-run",
        action="store_true",
        help="Run the check and decide whether it WOULD page, but never invoke the notifier "
        "— validate a new watch cron line without paging a human. Exit code is preserved.",
    )
    watch.add_argument(
        "--notify-retries",
        type=int,
        default=0,
        metavar="N",
        help="Re-attempt a FAILED page delivery up to N times (linear backoff) before "
        "falling back to a loud stderr page. Rides out a transient pager blip so a "
        "recoverable failure isn't reported as a permanently dropped alert. Default: 0.",
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
    dry_run = getattr(args, "dry_run", False)
    notify_retries = getattr(args, "notify_retries", 0)
    if notify_retries < 0:
        sys.stderr.write("ogle watch: --notify-retries must be >= 0\n")
        return EXIT_ERROR
    outcome = run_tick(
        args.check_args,
        notifier=notifier,
        page_on_error=args.page_on_error,
        dry_run=dry_run,
        notify_retries=notify_retries,
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
                        # The paging DECISION, decoupled from dispatch: True whenever the tick
                        # warranted a page, even under --dry-run where paged stays False. In a
                        # normal run would_page == paged; a wrapper validating a cron line gates
                        # on would_page, an alert router on paged/delivery_failed.
                        "would_page": outcome.would_page,
                        "dry_run": dry_run,
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
    # Under --dry-run a page was decided but suppressed — say so plainly rather than claim
    # a page went out (which never happened) or "healthy" (which hides the pending drift).
    if dry_run and outcome.would_page:
        status = "WOULD PAGE (dry-run, no page sent)"
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
