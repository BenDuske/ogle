"""Ogle CLI — the operator-facing entrypoint over the drift-check pipeline.

`ogle check` is the whole loop in one command:

    load baselines  ->  walk (live DataHub | offline signatures file)  ->  run_drift_check
                                                                                |
                                    save baselines  <-  render/alert  <---------+

Two input modes so the command is useful with OR without a live DataHub quickstart:

  * **Live walk** (`--gms` + `--models`/`--discover`) — pulls aspects through
    `ogle.walker.DataHubBackend`. Needs the `datahub` extra + a reachable GMS.
  * **Offline signatures** (`--signatures FILE`) — feeds pre-computed `DatasetSignature`s
    from JSON. No SDK, no Docker. This is how a scheduled job can hand Ogle signatures it
    pulled elsewhere, and how the CLI is unit-tested end-to-end.

Exit codes are chosen so a cron/Task wrapper can branch on them:
    0  healthy — no *new* incident to alert on (may include seeded/first-run datasets)
    1  a NEW incident fired (`DriftReport.should_alert`) — page Ben
    2  usage / input error
`--fail-on {low,medium,high}` tightens the 0/1 line for a CI gate: a new incident below
that severity floor is still reported (and tagged) but exits 0 instead of 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .llm import build_narrator
from .narrative import narrate
from .pipeline import DriftReport, run_drift_check
from .scorer import Severity, build_score_config
from .signature import DatasetSignature
from .store import BaselineStore

DEFAULT_STORE = "ogle-baselines.json"


def _emit(text: str, *, stream=None) -> None:
    """Print `text` without crashing on a legacy console encoding.

    The narrative carries emoji severity markers; a Windows cp1252 terminal raises
    UnicodeEncodeError on those. Encode to the stream's own encoding with errors="replace"
    so the command still runs (and its exit code stays trustworthy) on any console.
    """
    stream = stream or sys.stdout
    enc = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(enc, errors="replace").decode(enc) + "\n")


# ---------------------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------------------
def load_signatures_file(path: Path) -> Tuple[List[DatasetSignature], List[str]]:
    """Read signatures (and optional serving URNs) from a JSON file.

    Accepts either shape:
      * a bare list  ``[ <sig.to_dict()>, ... ]``  (no serving URNs), or
      * an object    ``{"signatures": [...], "serving_urns": [...]}``.

    Returns (signatures, serving_urns). Raises ``ValueError`` with an operator-readable
    message on any malformed input — the CLI turns that into exit code 2.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"signatures file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"signatures file is not valid JSON ({path}): {exc}")

    if isinstance(data, list):
        raw_sigs, serving = data, []
    elif isinstance(data, dict):
        raw_sigs = data.get("signatures", [])
        serving = list(data.get("serving_urns", []))
    else:
        raise ValueError("signatures file must be a JSON list or object")

    if not isinstance(raw_sigs, list):
        raise ValueError('"signatures" must be a JSON list')

    signatures: List[DatasetSignature] = []
    for i, item in enumerate(raw_sigs):
        if not isinstance(item, dict) or "urn" not in item:
            raise ValueError(f"signature #{i} is missing a urn / is not an object")
        signatures.append(DatasetSignature.from_dict(item))
    return signatures, serving


def _walk_live(gms: str, models: Sequence[str], discover: bool):
    """Run a live DataHub walk. Imported lazily so the SDK stays an optional extra.

    Returns (signatures, sorted_serving_urns, walk_result) — the full `WalkResult` rides
    along so the caller can hand it to `writeback.plan_writeback` without re-walking.
    """
    from .walker import DataHubBackend, walk_models

    backend = DataHubBackend(gms_server=gms)
    model_urns: List[str] = list(models)
    if discover:
        model_urns.extend(backend.discover_deployed_models())
    if not model_urns:
        raise ValueError(
            "no models to walk — pass --models URN [URN ...] or --discover"
        )
    # Dedup while preserving order so diagnostics list each model once.
    seen = set()
    ordered = [u for u in model_urns if not (u in seen or seen.add(u))]
    result = walk_models(backend, ordered)
    return result.signatures, sorted(result.serving_dataset_urns), result


def _do_writeback(findings, walk_result, gms: str, severity_tags: bool = False):
    """Live tag write-back. Imported lazily like the walker."""
    from .writeback import DataHubWritebackBackend, apply, plan_writeback

    backend = DataHubWritebackBackend(gms_server=gms)
    plan = plan_writeback(findings, walk_result, severity_tags=severity_tags)
    return plan, apply(plan, backend)


def _do_retract(recovered_urns, active_findings, walk_result, gms: str):
    """Live tag retraction — strip Ogle's tag off datasets whose drift cleared. Lazy import."""
    from .writeback import DataHubWritebackBackend, apply_retract, plan_retract

    backend = DataHubWritebackBackend(gms_server=gms)
    plan = plan_retract(recovered_urns, active_findings, walk_result)
    return plan, apply_retract(plan, backend)


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------
def render_report(report: DriftReport, *, as_json: bool) -> str:
    """Human markdown by default; machine JSON with --json."""
    if as_json:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)

    lines = [report.narrative.rstrip()]
    tail = []
    if report.new_urns:
        tail.append(f"seeded {len(report.new_urns)} new dataset(s) (first sighting)")
    if report.scored_urns:
        tail.append(f"checked {len(report.scored_urns)} dataset(s)")
    if report.suppressed_urns:
        tail.append(f"silenced {len(report.suppressed_urns)} muted dataset(s)")
    if tail:
        lines.append("")
        lines.append("_" + "; ".join(tail) + "._")
    return "\n".join(lines)


def gate_should_fail(report: DriftReport, fail_on: Optional[str]) -> bool:
    """CI exit-code gate: should this run exit non-zero (1) instead of 0?

    Pure — no I/O — so a wrapper's page/no-page decision is unit-testable on its own.

    * No *new* incident -> always False (exit 0), regardless of `fail_on`.
    * `fail_on is None` (default) -> any new incident fails the run: the page-on-drift
      contract every existing caller relies on.
    * `fail_on in {"low","medium","high"}` -> only a new incident whose OVERALL severity
      meets or exceeds that floor fails. A lower-severity new incident is still reported
      and still eligible for write-back, but the process exits 0 — so a CI gate can page
      on HIGH while merely logging medium/low drift.
    """
    if not report.should_alert:
        return False
    if fail_on is None:
        return True
    # should_alert is True only when there is an incident, so this is safe.
    return report.incident.overall_severity.rank >= Severity(fail_on).rank


# ---------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    # Build (and validate) the sensitivity config before any I/O so a bad threshold
    # fails fast with a usage error instead of after a live walk.
    try:
        cfg = build_score_config(
            volume_threshold=getattr(args, "volume_threshold", None),
            null_threshold=getattr(args, "null_threshold", None),
            escalate_when_serving=(
                False if getattr(args, "no_serving_escalation", False) else None
            ),
        )
    except ValueError as exc:
        print(f"ogle check: {exc}", file=sys.stderr)
        return 2

    store_path = Path(args.store)
    store = BaselineStore.load(store_path)
    if store.recovered_from_corruption:
        # A crash-looping check on a bad store would go silently blind to drift — warn loudly
        # (stderr, so JSON on stdout stays clean) and re-baseline this run rather than fail.
        _emit(
            f"ogle check: WARNING baseline store at {store_path} was unreadable "
            f"(corrupt/foreign); quarantined to {store.corrupt_backup_path} and "
            f"re-baselining from scratch — this run cannot detect drift against prior state.",
            stream=sys.stderr,
        )

    # Gather the current signatures + which of them feed a serving model.
    walk_result = None  # None in offline mode; a real WalkResult in live mode.
    try:
        if args.signatures:
            signatures, serving = load_signatures_file(Path(args.signatures))
        else:
            signatures, serving, walk_result = _walk_live(
                args.gms, args.models or [], args.discover
            )
        # An explicit --serving on the command line augments whatever the source reported.
        serving = sorted(set(serving) | set(args.serving or []))
    except ValueError as exc:
        print(f"ogle check: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # live-walk failure (SDK/network) — report, don't traceback
        print(f"ogle check: live walk failed: {exc}", file=sys.stderr)
        return 2

    report = run_drift_check(
        store,
        signatures,
        serving_urns=serving,
        cfg=cfg,
        update_baselines=not args.no_update,
    )

    _emit(render_report(report, as_json=args.json))

    # Optional LLM-phrased incident summary. Only when there is something to narrate and
    # not in JSON mode (markdown prose would corrupt the JSON payload). `narrate` itself
    # falls back to the deterministic report if the model is unreachable, so a build/parse
    # error is the only thing we surface here.
    if getattr(args, "narrate", None) and not args.json and report.findings:
        try:
            narrator = build_narrator(args.narrate)
        except ValueError as exc:
            print(f"ogle check: {exc}", file=sys.stderr)
            return 2
        _emit("\n---\n\n**Incident summary**\n")
        _emit(narrate(report.findings, llm=narrator))

    # Persist advanced baselines/incident memory unless asked for a read-only probe.
    if not args.no_update:
        # Self-clean the mute list: a lapsed snooze is dropped here so the store never
        # accumulates dead entries (and `ogle muted` stays honest).
        store.purge_expired_mutes(time.time())
        try:
            store.save(store_path)
        except Exception as exc:
            print(f"ogle check: warning — could not save store: {exc}", file=sys.stderr)

    # Optional outbound tag write-back — only when THIS run fired a new incident, so a
    # scheduler doesn't reapply the same tag on every tick.
    if args.write_back:
        if not report.should_alert:
            _emit("_write-back skipped: no new incident this run._")
        elif walk_result is None:
            print(
                "ogle check: --write-back requires a live walk (--gms/--models/--discover);"
                " offline mode has no way to reach DataHub.",
                file=sys.stderr,
            )
            return 2
        else:
            try:
                plan, wb_result = _do_writeback(
                    report.findings,
                    walk_result,
                    args.gms,
                    severity_tags=getattr(args, "write_back_severity", False),
                )
            except Exception as exc:
                print(f"ogle check: write-back failed: {exc}", file=sys.stderr)
                # Alert still fires — the check itself succeeded; the write-back is an
                # optional side-effect. Return 1 (new incident), not 2.
                return 1
            _render_writeback(plan, wb_result, as_json=args.json)

    # Optional retraction — clear Ogle's tag off assets that recovered. Unlike write-back
    # this runs on healthy runs too (recovery is exactly the no-new-incident case). Only
    # meaningful against a live graph; the read-before-write in apply_retract keeps it a
    # cheap no-op on entities Ogle never tagged.
    if getattr(args, "retract_cleared", False):
        if walk_result is None:
            print(
                "ogle check: --retract-cleared requires a live walk (--gms/--models/--discover);"
                " offline mode has no way to reach DataHub.",
                file=sys.stderr,
            )
            return 2
        drifted = {f.urn for f in report.findings}
        recovered = [u for u in report.scored_urns if u not in drifted]
        if not recovered:
            _emit("_retract: no recovered datasets this run._")
        else:
            try:
                r_plan, r_result = _do_retract(
                    recovered, report.findings, walk_result, args.gms
                )
            except Exception as exc:
                print(f"ogle check: retract failed: {exc}", file=sys.stderr)
                # Same contract as write-back: the check succeeded; retraction is an
                # optional side-effect, so don't upgrade the exit code on its account.
                return 1 if gate_should_fail(report, getattr(args, "fail_on", None)) else 0
            _render_retract(r_plan, r_result, as_json=args.json)

    # CI exit-code gate. Without --fail-on this is exactly the old "fail on any new
    # incident" contract; with it, a below-floor new incident is still reported (and
    # tagged) but exits 0. Announce that suppression so it's never a silent pass —
    # skipped in --json mode where prose would corrupt the payload.
    fail = gate_should_fail(report, getattr(args, "fail_on", None))
    if report.should_alert and not fail and not args.json:
        _emit(
            f"_new {report.incident.overall_severity.value} incident is below "
            f"--fail-on {args.fail_on} — reported, exit 0._"
        )
    return 1 if fail else 0


def _render_writeback(plan, result, *, as_json: bool) -> None:
    if as_json:
        _emit(
            json.dumps(
                {"plan": plan.to_dict(), "result": result.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not plan.actions:
        _emit("_write-back: nothing to tag._")
        return
    lines = ["", f"**Tagged {len(result.tagged_entities)} entity(ies) in DataHub:**"]
    for urn in result.tagged_entities:
        lines.append(f"- `{urn}`")
    if result.unchanged:
        lines.append(f"_({len(result.unchanged)} already tagged, skipped)_")
    if result.failed:
        lines.append(f"_({len(result.failed)} failed — see logs)_")
    _emit("\n".join(lines))


def _render_retract(plan, result, *, as_json: bool) -> None:
    if as_json:
        _emit(
            json.dumps(
                {"plan": plan.to_dict(), "result": result.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return
    # `tagged_entities` de-dupes the entities we actually changed — here it's the set we
    # UN-flagged (apply_retract reuses WritebackResult; `applied` = tags removed).
    if not result.tagged_entities:
        _emit("_retract: nothing to clear (no recovered asset carried an Ogle tag)._")
        return
    lines = ["", f"**Cleared Ogle's drift tag from {len(result.tagged_entities)} entity(ies):**"]
    for urn in result.tagged_entities:
        lines.append(f"- `{urn}`")
    if result.unchanged:
        lines.append(f"_({len(result.unchanged)} already clean, skipped)_")
    if result.failed:
        lines.append(f"_({len(result.failed)} failed — see logs)_")
    _emit("\n".join(lines))


# Repo-root examples/ (works from a clone: src/ogle/cli.py -> parents[2] == repo root).
_DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "demo"


def cmd_demo(args: argparse.Namespace) -> int:
    """Zero-setup, keyless proof: seed healthy baselines, then re-check drifted fixtures.

    Runs the *same* `run_drift_check` code path the live DataHub walk feeds — no SDK, no
    Docker, no API key — against the bundled `examples/demo/*.json` fixtures. First pass
    seeds and stays healthy (exit 0); second pass fires the HIGH serving-path incident that
    `examples/alerts/churn-orders-drift.md` captured. Exit 1 on that alert, matching a real
    `ogle check`, so a judge sees the whole loop in one command.
    """
    healthy = _DEMO_DIR / "healthy-signatures.json"
    drifted = _DEMO_DIR / "drifted-signatures.json"
    for f in (healthy, drifted):
        if not f.exists():
            print(f"ogle demo: bundled fixture not found: {f}", file=sys.stderr)
            return 2

    # In-memory store — the demo never touches the operator's cwd or a real baseline file.
    store = BaselineStore.load(_DEMO_DIR / "__demo_never_written__.json")

    try:
        h_sigs, h_serving = load_signatures_file(healthy)
        d_sigs, d_serving = load_signatures_file(drifted)
    except ValueError as exc:  # a corrupted bundled fixture — treat as input error
        print(f"ogle demo: {exc}", file=sys.stderr)
        return 2

    _emit("# Ogle offline demo — churn serving-path drift\n")
    _emit("_Keyless, no DataHub required; same drift-check code path as a live walk._\n")

    _emit("## 1. Seed baselines (healthy fixture)\n")
    seed = run_drift_check(store, h_sigs, serving_urns=h_serving, update_baselines=True)
    _emit(render_report(seed, as_json=False))

    _emit("\n## 2. Re-check the drifted fixture\n")
    drift = run_drift_check(store, d_sigs, serving_urns=d_serving, update_baselines=True)
    _emit(render_report(drift, as_json=False))

    # Optional feature-#2 showcase: the same LLM root-cause narrator `ogle check --narrate`
    # exposes, so a judge sees BOTH flagship features from the one keyless command. `narrate`
    # falls back to the deterministic summary when the model is unreachable, so a laptop with
    # no local Ollama still gets a clean section instead of an error — the demo stays keyless.
    if getattr(args, "narrate", None) and drift.findings:
        try:
            narrator = build_narrator(args.narrate)
        except ValueError as exc:
            print(f"ogle demo: {exc}", file=sys.stderr)
            return 2
        _emit("\n## 3. LLM root-cause summary\n")
        _emit(narrate(drift.findings, llm=narrator))

    _emit(
        "\n---\n_Reproduces `examples/alerts/churn-orders-drift.md`. "
        "Run against DataHub: `ogle check --gms http://localhost:8080 --discover`._"
    )
    return 1 if drift.should_alert else 0


def _fmt_expiry(exp: float) -> str:
    """Human-readable UTC expiry for a snooze (stable, timezone-explicit for tests)."""
    return datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# Duration suffixes shared by `--stale` parsing (and its help text). Kept in seconds so
# the largest sensible unit reads first when we build the error message.
_AGE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_age(text: str) -> Optional[float]:
    """Parse a compact duration like `3d`, `12h`, `30m`, `45s`, `2w` into seconds.

    Returns the positive number of seconds, or None if `text` isn't a positive integer
    followed by a single s/m/h/d/w unit. A bare number (no unit) is rejected on purpose:
    an ambiguous `--stale 3` shouldn't silently mean seconds *or* days. Zero/negative are
    also rejected — a staleness threshold of "0 ago" matches everything, which is never
    what an operator means.
    """
    if not text:
        return None
    raw = text.strip().lower()
    if len(raw) < 2:
        return None
    unit = raw[-1]
    mult = _AGE_UNITS.get(unit)
    if mult is None:
        return None
    try:
        amount = int(raw[:-1])
    except ValueError:
        return None
    if amount <= 0:
        return None
    return amount * mult


def _fmt_age(seconds: float) -> str:
    """A compact, human-readable relative age like `just now`, `5m`, `3h`, `2d`, `1w`.

    Picks the largest whole unit that fits so `ogle incidents` reads at a glance ("last
    seen 3h ago") without a wall of precision. Sub-minute ages collapse to `just now`;
    a negative age (clock skew / a future stamp) also reads `just now` rather than a
    nonsensical negative.
    """
    s = int(seconds)
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    if s < 604800:
        return f"{s // 86400}d"
    return f"{s // 604800}w"


def cmd_mute(args: argparse.Namespace) -> int:
    """Mark a dataset as a known false positive so its drift stops paging.

    Persists into the same store `ogle check` reads, so the next scheduled run silences it.
    Idempotent: muting an already-muted URN reports that rather than claiming a change.
    With `--for`/`--for-hours` the mute is a *snooze* that auto-expires, so a "quiet it for
    now" can never become a permanent blind spot.
    """
    days = getattr(args, "for_days", None)
    hours = getattr(args, "for_hours", None)
    if days is not None and hours is not None:
        print("ogle mute: use --for OR --for-hours, not both", file=sys.stderr)
        return 2
    until: Optional[float] = None
    if days is not None or hours is not None:
        amount = days if days is not None else hours
        if amount <= 0:
            print("ogle mute: snooze duration must be positive", file=sys.stderr)
            return 2
        seconds = amount * 86400 if days is not None else amount * 3600
        until = time.time() + seconds

    store_path = Path(args.store)
    store = BaselineStore.load(store_path)
    newly = store.mute(args.urn, until=until)
    try:
        store.save(store_path)
    except Exception as exc:
        print(f"ogle mute: could not save store: {exc}", file=sys.stderr)
        return 2
    if not newly:
        _emit(f"_already muted: {args.urn}_")
    elif until is not None:
        _emit(f"🔇 snoozed {args.urn} until {_fmt_expiry(until)}")
    else:
        _emit(f"🔇 muted {args.urn}")
    return 0


def cmd_unmute(args: argparse.Namespace) -> int:
    """Un-mute a dataset so its drift can page again."""
    store_path = Path(args.store)
    store = BaselineStore.load(store_path)
    was = store.unmute(args.urn)
    try:
        store.save(store_path)
    except Exception as exc:
        print(f"ogle unmute: could not save store: {exc}", file=sys.stderr)
        return 2
    _emit(
        f"🔔 unmuted {args.urn}" if was else f"_not muted: {args.urn}_"
    )
    return 0


def cmd_muted(args: argparse.Namespace) -> int:
    """List the datasets currently muted in the store (expired snoozes excluded)."""
    store = BaselineStore.load(Path(args.store))
    now = time.time()
    urns = store.muted(now)  # active mutes only — expired snoozes excluded
    # `--urns`: plain machine output — just each muted URN, one per line. The write-side
    # selector symmetric with `baselines --urns`/`incidents --fingerprints`: turns the mute
    # list into a pipe for bulk `unmute`/`show`. Overrides --json (this IS the scriptable
    # form) and stays SILENT on an empty set so a pipe gets a clean stream, not a prose line.
    if getattr(args, "urns", False):
        for urn in urns:
            _emit(urn)
        return 0
    if args.json:
        entries = []
        for urn in urns:
            exp = store.mute_expiry(urn)
            entries.append({"urn": urn, "until": exp})  # until=None -> permanent
        _emit(json.dumps({"muted": entries}, indent=2, sort_keys=True))
        return 0
    if not urns:
        _emit("_no muted datasets._")
        return 0
    _emit(f"**{len(urns)} muted dataset(s):**")
    for urn in urns:
        exp = store.mute_expiry(urn)
        suffix = f" — snoozed until {_fmt_expiry(exp)}" if exp is not None else ""
        _emit(f"- `{urn}`{suffix}")
    return 0


def _baseline_field_count(store: "BaselineStore", urn: str) -> int:
    """Schema-field count for a baseline, or -1 if it has no signature (sorts last)."""
    sig = store.get_baseline(urn)
    return len(sig.schema_fields) if sig else -1


def _baseline_row_count(store: "BaselineStore", urn: str) -> int:
    """Row count for a baseline, or -1 when unknown/missing (sorts last)."""
    sig = store.get_baseline(urn)
    if sig and sig.row_count is not None:
        return sig.row_count
    return -1


def cmd_baselines(args: argparse.Namespace) -> int:
    """List the datasets Ogle has a baseline signature for — the *other* half of the store.

    Read-only. Where `ogle incidents` shows remembered drift, this shows what Ogle is
    *watching*: every dataset it has captured a signature for, with the schema shape it will
    diff the next walk against. Lets an operator answer "is this dataset actually under
    Ogle's eye?" and "how many am I tracking?" without re-walking DataHub. `--urns` prints
    just the URNs (one per line) so the watch-list can feed a write-side command, e.g.
    `ogle baselines --grep serving --urns | xargs -n1 ogle mute`.
    """
    store = BaselineStore.load(Path(args.store))
    all_urns = store.urns()  # already sorted for stable output

    # `--grep`: substring match on the URN (case-insensitive), mirroring `incidents --grep`.
    # An all-whitespace needle matches NOTHING (a fat-fingered `--grep ""` is a slip, not a
    # wildcard), same as the incidents view.
    needle = getattr(args, "grep", None)
    filtered = needle is not None
    if needle is not None:
        low = needle.strip().lower()
        urns = [u for u in all_urns if low and low in u.lower()]
    else:
        urns = list(all_urns)

    # `--sort` picks the ordering axis (default `urn` = the alphabetical order the store
    # already returns; the stable baseline for scripting). `fields`/`rows` put the
    # highest-blast-radius datasets first — the widest schemas and highest-volume tables,
    # where silent drift matters most — with URN ascending as the deterministic tiebreak
    # (negate the metric so it descends while the URN stays ascending). A baseline with no
    # signature or unknown row_count sorts last (-1), mirroring how `incidents --sort` sinks
    # unknown severity. Applied here so --urns/--json/human views all share one order.
    sort_axis = getattr(args, "sort", None) or "urn"
    if sort_axis == "fields":
        urns = sorted(urns, key=lambda u: (-_baseline_field_count(store, u), u))
    elif sort_axis == "rows":
        urns = sorted(urns, key=lambda u: (-_baseline_row_count(store, u), u))

    # `--urns`: plain machine output — just each URN, one per line, honoring --grep. Turns
    # the watch-list into a selector for a write-side command (`mute`/`check --models`).
    # Overrides --json (this IS the scriptable form) and stays SILENT on an empty set so a
    # pipe gets a clean stream rather than a prose "no baselines" line.
    if getattr(args, "urns", False):
        for u in urns:
            _emit(u)
        return 0

    if args.json:
        entries = []
        for u in urns:
            sig = store.get_baseline(u)
            entries.append(
                {
                    "urn": u,
                    "fields": len(sig.schema_fields) if sig else 0,
                    "row_count": sig.row_count if sig else None,
                    "schema_hash": sig.schema_hash if sig else None,
                }
            )
        _emit(json.dumps({"baselines": entries}, indent=2, sort_keys=True))
        return 0

    if not urns:
        # Distinguish "nothing tracked" from "filter hid everything" so the operator knows
        # whether to widen --grep vs. that Ogle has no baselines at all (mirrors incidents).
        if filtered and all_urns:
            _emit(f"_no baselines match the filter ({len(all_urns)} tracked)._")
        else:
            _emit("_no baselines yet — run `ogle check` to capture some._")
        return 0

    _emit(f"**{len(urns)} tracked dataset(s):**")
    for u in urns:
        sig = store.get_baseline(u)
        nf = len(sig.schema_fields) if sig else 0
        rc = sig.row_count if sig else None
        rpart = f" · {rc} rows" if rc is not None else ""
        hpart = f"  `{sig.schema_hash[:12]}`" if sig else ""
        _emit(f"- `{u}` — {nf} field(s){rpart}{hpart}")
    return 0


def _mute_state(store: "BaselineStore", urn: str, now: float) -> dict:
    """The live mute state of one URN as a small dict: {muted, snoozed, until}.

    `muted` is the effective silence right now (permanent OR an unexpired snooze); `snoozed`
    distinguishes a timed mute from a permanent one; `until` is the snooze expiry (epoch
    seconds) or None for permanent/not-muted. Reuses the store's own `is_muted`/`mute_expiry`
    so `show` reports exactly what `ogle check` would honor on the next walk.
    """
    muted = store.is_muted(urn, now)
    until = store.mute_expiry(urn)
    return {"muted": muted, "snoozed": muted and until is not None, "until": until}


def cmd_show(args: argparse.Namespace) -> int:
    """Drill into ONE watched dataset — the full memorized signature plus its mute state.

    Read-only. Where `ogle baselines` lists the whole watch-list one summary line each, this
    opens a single URN and shows what no other view does: the exact field list (path + native
    type) Ogle memorized, each field's null fraction (the quality signal behind QUALITY
    drift), the row count, the FULL schema hash, the capture provenance, and whether the
    dataset is currently muted/snoozed. The natural next step after a page — "Ogle flagged
    `orders`; show me exactly what it has on it." Keys on an EXACT URN (the `--urns` selector
    emits them whole), so it composes with the watch-list: `ogle baselines --grep orders
    --urns | head -1 | xargs ogle show`.

    Incidents are keyed by drift-*event* fingerprint, not by URN, so a dataset's remembered
    drift lives in `ogle incidents --grep <name>`, not here — this view is strictly the
    baseline signature + mute state, the two facets the store holds per URN.

    Exit 0 when the dataset is watched, 1 when it isn't (a scriptable "no such baseline"),
    so `ogle show <urn> >/dev/null && …` branches cleanly.
    """
    store = BaselineStore.load(Path(args.store))
    urn = args.urn
    sig = store.get_baseline(urn)
    now = time.time()

    if sig is None:
        # Not on the watch-list. Distinguish an empty store from a plain miss so the operator
        # knows whether to run `check` first vs. that this specific URN just isn't tracked.
        if len(store) == 0:
            _emit(f"_not watched: `{urn}` — store is empty; run `ogle check` first._")
        else:
            _emit(f"_not watched: `{urn}` ({len(store)} dataset(s) tracked)._")
        return 1

    mute = _mute_state(store, urn, now)
    # Stable field order (schema_fields tuple order isn't guaranteed meaningful — the hash is
    # order-independent), so the same baseline always renders identically.
    fields = sorted(sig.schema_fields, key=lambda f: f.path)

    if args.json:
        entries = []
        for f in fields:
            entry = {"path": f.path, "native_type": f.native_type}
            if f.path in sig.field_null_fractions:
                entry["null_fraction"] = sig.field_null_fractions[f.path]
            entries.append(entry)
        _emit(
            json.dumps(
                {
                    "dataset": {
                        "urn": urn,
                        "fields": entries,
                        "field_count": len(fields),
                        "row_count": sig.row_count,
                        "schema_hash": sig.schema_hash,
                        "computed_at": sig.computed_at,
                        "muted": mute["muted"],
                        "muted_until": mute["until"],
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _emit(f"**dataset `{urn}`**")
    rpart = f"{sig.row_count} row(s)" if sig.row_count is not None else "rows unknown"
    _emit(f"- schema: {len(fields)} field(s) · {rpart}")
    _emit(f"- hash: `{sig.schema_hash}`")
    if sig.computed_at:
        _emit(f"- captured: {sig.computed_at}")
    if mute["snoozed"]:
        _emit(f"- 🔇 muted: snoozed until {_fmt_expiry(mute['until'])}")
    elif mute["muted"]:
        _emit("- 🔇 muted: permanent")
    else:
        _emit("- 🔔 muted: no")
    if fields:
        _emit("**fields:**")
        for f in fields:
            frac = sig.field_null_fractions.get(f.path)
            npart = f" — {frac * 100:.0f}% null" if frac is not None else ""
            _emit(f"- `{f.path}` : {f.native_type}{npart}")
    return 0


def _fmt_frac(v: Optional[float]) -> str:
    """A null fraction as a percent, or `unknown` when DataHub had no profile for it.

    A missing fraction is NOT 0% — it means "not measured". Rendering it as `unknown`
    keeps `ogle diff` from claiming a field went 0%→12% when really the baseline simply
    never had a null profile to compare against.
    """
    return f"{v * 100:.1f}%" if v is not None else "unknown"


def _diff_signatures(old: DatasetSignature, new: DatasetSignature) -> dict:
    """Pure structural diff of two signatures (old baseline vs new candidate).

    Returns the four field-level deltas plus row-count/schema-hash facets. Split out from
    the command so the comparison logic is unit-testable with no store or file I/O.

    Null-fraction changes are reported ONLY for fields present in both signatures — a
    field that was added or removed already carries its null story on the add/remove line,
    so re-reporting it here would be double-counting. A null fraction that appears or
    disappears on a *surviving* field (known↔unknown) is a real change and is kept. The
    0.1-percentage-point rounding gate drops float noise so a re-profiled-but-stable
    column doesn't show as drift.
    """
    old_fields = {f.path: f.native_type for f in old.schema_fields}
    new_fields = {f.path: f.native_type for f in new.schema_fields}

    added = [
        {"path": p, "native_type": new_fields[p]}
        for p in sorted(new_fields.keys() - old_fields.keys())
    ]
    removed = [
        {"path": p, "native_type": old_fields[p]}
        for p in sorted(old_fields.keys() - new_fields.keys())
    ]
    common = sorted(old_fields.keys() & new_fields.keys())
    type_changed = [
        {"path": p, "old_type": old_fields[p], "new_type": new_fields[p]}
        for p in common
        if old_fields[p] != new_fields[p]
    ]

    null_changed = []
    for p in common:
        o = old.field_null_fractions.get(p)
        n = new.field_null_fractions.get(p)
        if o is None and n is None:
            continue
        # A fraction appearing/disappearing is a change; two known fractions must move by
        # more than a rounded 0.1pp to count (drops re-profiling float jitter).
        if o is None or n is None or round(o * 100, 1) != round(n * 100, 1):
            null_changed.append({"path": p, "old": o, "new": n})

    row_changed = old.row_count != new.row_count
    row_delta = (
        new.row_count - old.row_count
        if old.row_count is not None and new.row_count is not None
        else None
    )
    hash_changed = old.schema_hash != new.schema_hash
    identical = not (
        added or removed or type_changed or null_changed or row_changed
    )

    return {
        "identical": identical,
        "fields_added": added,
        "fields_removed": removed,
        "fields_type_changed": type_changed,
        "null_fraction_changed": null_changed,
        "row_count": {
            "old": old.row_count,
            "new": new.row_count,
            "delta": row_delta,
            "changed": row_changed,
        },
        "schema_hash": {
            "old": old.schema_hash,
            "new": new.schema_hash,
            "changed": hash_changed,
        },
    }


def cmd_diff(args: argparse.Namespace) -> int:
    """Explain the drift on ONE dataset: stored baseline vs a candidate signature file.

    Read-only and side-effect-free — it never records an incident, advances a baseline, or
    touches the store. Where `ogle check` walks live, scores, and *remembers*, `diff`
    answers the narrower investigative question after a page: "what EXACTLY changed on this
    table?" It reads the same offline signatures file `check --signatures` consumes (so the
    dump you'd feed a dry-run doubles as the diff input) and prints the field-level delta —
    fields added / removed / retyped, null-fraction moves, row-count change, and whether the
    schema hash flipped.

    Exit codes are the scriptable drift verdict: 0 = identical to baseline (no drift),
    1 = differences found, 2 = can't compare (URN not watched, URN absent from the
    signatures file, or the file is malformed). Keeping preconditions on 2 leaves 0/1 as a
    clean `ogle diff <urn> --signatures new.json && echo unchanged` gate.
    """
    store = BaselineStore.load(Path(args.store))
    urn = args.urn

    old = store.get_baseline(urn)
    if old is None:
        if len(store) == 0:
            print(
                f"ogle diff: `{urn}` is not watched — store is empty; run `ogle check` first.",
                file=sys.stderr,
            )
        else:
            print(
                f"ogle diff: `{urn}` is not watched ({len(store)} dataset(s) tracked); "
                "nothing to diff against.",
                file=sys.stderr,
            )
        return 2

    try:
        signatures, _serving = load_signatures_file(Path(args.signatures))
    except ValueError as exc:
        print(f"ogle diff: {exc}", file=sys.stderr)
        return 2

    new = next((s for s in signatures if s.urn == urn), None)
    if new is None:
        print(
            f"ogle diff: `{urn}` is not present in {args.signatures} "
            f"({len(signatures)} signature(s) in the file).",
            file=sys.stderr,
        )
        return 2

    d = _diff_signatures(old, new)

    if args.json:
        _emit(json.dumps({"diff": {"urn": urn, **d}}, indent=2, sort_keys=True))
        return 0 if d["identical"] else 1

    if d["identical"]:
        _emit(f"**diff `{urn}`** — identical to baseline (no drift).")
        return 0

    _emit(f"**diff `{urn}`** — baseline → candidate")
    rc = d["row_count"]
    if rc["changed"]:
        if rc["delta"] is not None:
            sign = "+" if rc["delta"] >= 0 else ""
            _emit(f"- rows: {rc['old']} → {rc['new']} ({sign}{rc['delta']})")
        else:
            o = rc["old"] if rc["old"] is not None else "unknown"
            n = rc["new"] if rc["new"] is not None else "unknown"
            _emit(f"- rows: {o} → {n}")
    else:
        rpart = rc["old"] if rc["old"] is not None else "unknown"
        _emit(f"- rows: unchanged ({rpart})")
    sh = d["schema_hash"]
    if sh["changed"]:
        _emit(f"- schema hash: `{sh['old']}` → `{sh['new']}`")
    else:
        _emit(f"- schema hash: unchanged `{sh['old']}`")

    if d["fields_added"] or d["fields_removed"] or d["fields_type_changed"]:
        _emit("**schema:**")
        for f in d["fields_added"]:
            _emit(f"- ➕ `{f['path']}` : {f['native_type']}")
        for f in d["fields_removed"]:
            _emit(f"- ➖ `{f['path']}` : {f['native_type']}")
        for f in d["fields_type_changed"]:
            _emit(f"- 🔀 `{f['path']}` : {f['old_type']} → {f['new_type']}")

    if d["null_fraction_changed"]:
        _emit("**null fractions:**")
        for f in d["null_fraction_changed"]:
            _emit(f"- `{f['path']}` : {_fmt_frac(f['old'])} → {_fmt_frac(f['new'])} null")

    return 1


# Severity marks for the incident-memory view, keyed by the string the store persists
# (the store stays decoupled from the scorer's Severity enum, so the CLI maps here).
_INCIDENT_SEV_MARK = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e1"}


def _incident_severity_rank(rec: dict) -> int:
    """A record's `Severity.rank`, with unknown/legacy severity sorting last (-1)."""
    try:
        return Severity(rec.get("severity")).rank
    except (ValueError, TypeError):
        return -1


def _incident_sort_key(rec: dict) -> tuple:
    """Worst-severity-first, then most-recurred, then stable by fingerprint."""
    return (
        _incident_severity_rank(rec),
        int(rec.get("count", 0)),
        rec.get("fingerprint", ""),
    )


# `ogle incidents --sort` orderings. Every key is applied with reverse=True, so each puts
# the "most" of its primary axis first, then falls back to the other axes for a total,
# deterministic order (fingerprint last so equal rows never reshuffle between runs).
#   severity — worst severity first (default; the triage order), recurrence as tiebreak.
#   count    — most-recurring first (the chronic/flapping drift), severity as tiebreak.
#   datasets — broadest blast radius first (most datasets), severity as tiebreak.
_INCIDENT_SORTS = {
    "severity": _incident_sort_key,
    "count": lambda r: (
        int(r.get("count", 0)),
        _incident_severity_rank(r),
        r.get("fingerprint", ""),
    ),
    "datasets": lambda r: (
        int(r.get("datasets", 0)),
        _incident_severity_rank(r),
        r.get("fingerprint", ""),
    ),
    # recent — most-recently-seen first (freshest drift on top). A record with no
    # last_seen (legacy/untimed) sorts as -1 so it sinks under reverse=True, mirroring how
    # unknown severity/rows sink elsewhere; fingerprint is the deterministic tiebreak.
    "recent": lambda r: (
        r["last_seen"] if r.get("last_seen") is not None else -1.0,
        r.get("fingerprint", ""),
    ),
}


def _incident_matches_needle(rec: dict, needle: str) -> bool:
    """True if `needle` (case-insensitive) is a substring of the incident's title or
    fingerprint.

    The text axis behind `ogle incidents --grep`: find specific drift in a large memory by
    keyword. Matches the human-facing `title` (a dataset name, feature, or drift phrase) OR
    the `fingerprint` — so a fingerprint prefix works as a needle too, mirroring how
    `ogle resolve` accepts prefixes. An un-titled legacy record (no `title`) still matches on
    its fingerprint. An all-whitespace needle matches nothing meaningful, so it's treated as
    "no match" rather than "match everything" (an empty grep is a user slip, not a wildcard).
    """
    probe = needle.strip().lower()
    if not probe:
        return False
    title = (rec.get("title") or "").lower()
    fingerprint = (rec.get("fingerprint") or "").lower()
    return probe in title or probe in fingerprint


def _incident_passes(
    rec: dict,
    min_rank: Optional[int],
    serving_only: bool,
    min_count: Optional[int] = None,
    needle: Optional[str] = None,
    stale_before: Optional[float] = None,
) -> bool:
    """True if a remembered incident survives the `ogle incidents` triage filters.

    `min_rank` is a `Severity.rank` floor (None = no floor). A record whose severity is
    unknown/legacy ranks -1, so ANY `--min-severity` floor drops it — asking for a floor
    is asking to hide the un-triageable. `serving_only` keeps only serving-path incidents.
    `min_count` is a recurrence floor (None = no floor): keeps only incidents seen at least
    that many times, surfacing the chronic/flapping drift that keeps coming back. `needle`
    (None = no text filter) keeps only incidents whose title/fingerprint contains it
    (case-insensitive). `stale_before` (None = no staleness filter) keeps only incidents
    whose last_seen is KNOWN and older than that epoch cutoff — the drift Ogle hasn't seen
    recur lately, i.e. resolve/forget candidates. A record with no last_seen (legacy/untimed)
    can't be proven stale, so it's dropped by the filter rather than guessed. All filters
    are ANDed; passing none keeps everything.
    """
    if serving_only and not rec.get("serving"):
        return False
    if min_count is not None and int(rec.get("count", 0)) < min_count:
        return False
    if needle is not None and not _incident_matches_needle(rec, needle):
        return False
    if stale_before is not None:
        ls = rec.get("last_seen")
        if ls is None or ls >= stale_before:
            return False
    if min_rank is not None:
        try:
            rank = Severity(rec.get("severity")).rank
        except (ValueError, TypeError):
            rank = -1
        if rank < min_rank:
            return False
    return True


def _incidents_gate_fail(records: List[dict], fail_on: Optional[str]) -> bool:
    """True if any remembered incident meets/exceeds the `--fail-on` severity floor.

    Turns the read-only `ogle incidents` view into a CI/scheduled health gate. Where
    `check --fail-on` gates on *new* drift surfaced this run, this gates on whether Ogle's
    *memory* still holds open drift at/above a floor — so a nightly job can keep failing
    while any high-severity drift remains un-resolved, even on runs that surface nothing
    new (drift resolves only when `ogle resolve` forgets it or its fingerprint stops
    recurring). Evaluated against the already-filtered set, so it composes with
    `--min-severity`/`--serving-only`/`--min-count`, but is INDEPENDENT of `--limit`: a
    display cap must never change the pass/fail verdict. Unknown/legacy severities rank -1
    and never trip a floor (same rule as `--min-severity`).
    """
    if fail_on is None:
        return False
    floor = Severity(fail_on).rank
    for rec in records:
        try:
            rank = Severity(rec.get("severity")).rank
        except (ValueError, TypeError):
            rank = -1
        if rank >= floor:
            return True
    return False


def _incident_summary(records: List[dict]) -> dict:
    """Aggregate a set of remembered incidents into a triage rollup.

    Summarizes exactly the records handed in — the caller passes the already-filtered set,
    so `--summary` composes with `--min-severity`/`--serving-only`/`--min-count` (the rollup
    describes what the filter kept, not the whole store). `recurring` counts incidents seen
    at least twice (the flapping/chronic ones); `total_sightings` is the sum of every
    incident's observation count. Legacy/unknown severities land in the `unknown` bucket so
    the shape is stable regardless of what the store holds.
    """
    by_severity = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    serving = 0
    recurring = 0
    total_sightings = 0
    for r in records:
        sev = r.get("severity")
        key = sev if sev in ("high", "medium", "low") else "unknown"
        by_severity[key] += 1
        if r.get("serving"):
            serving += 1
        count = int(r.get("count", 0))
        total_sightings += count
        if count >= 2:
            recurring += 1
    return {
        "total": len(records),
        "by_severity": by_severity,
        "serving": serving,
        "recurring": recurring,
        "total_sightings": total_sightings,
    }


def _resolve_fingerprint(store: BaselineStore, needle: str) -> Tuple[Optional[str], List[str]]:
    """Look up a full fingerprint from a user-supplied token.

    Returns `(fp, candidates)`:
      * `(fp, [])`        — exact match OR unambiguous prefix match; caller can resolve `fp`.
      * `(None, [])`      — no match at all; caller reports the miss.
      * `(None, [a, b…])` — an ambiguous prefix (≥2 candidates); caller must refuse and list them.

    `ogle incidents` prints 16-hex fingerprints; typing all 16 is tedious. Accept any non-
    empty prefix and disambiguate — like a git short SHA — so the operator can paste 8 chars
    and move on. Prefix ambiguity is always an error (never a guess), so a partial match
    with two open incidents can't silently drop the wrong one.
    """
    known = list(store.seen_incidents.keys())
    if needle in store.seen_incidents:
        return needle, []
    if not needle:
        return None, []
    candidates = [fp for fp in known if fp.startswith(needle)]
    if len(candidates) == 1:
        return candidates[0], []
    if len(candidates) > 1:
        return None, sorted(candidates)
    return None, []


def _expand_stdin_fingerprints(tokens: List[str]) -> List[str]:
    """Expand a lone `-` token into whitespace-separated fingerprints read from stdin.

    Lets the documented selector pipe run natively on Windows, where `xargs` isn't a
    built-in:
        ogle incidents --serving-only --fingerprints | ogle resolve -
    `sys.stdin.read().split()` splits on any whitespace and drops empties, so trailing
    CRs and blank lines never become bogus tokens (same guarantee the per-token strip
    gives the `xargs` path). Stdin is read at most once even if `-` is repeated; the
    piped tokens are spliced in at the first `-` and any further `-` are dropped, and
    every non-`-` token passes through in place and order. A fingerprint is 16-hex, so a
    literal `-` is never a real token — no ambiguity with a value.
    """
    if "-" not in tokens:
        return tokens
    piped = sys.stdin.read().split()
    out: List[str] = []
    inserted = False
    for t in tokens:
        if t == "-":
            if not inserted:
                out.extend(piped)
                inserted = True
        else:
            out.append(t)
    return out


def cmd_resolve(args: argparse.Namespace) -> int:
    """Mark one or more remembered incidents as resolved (drops them from cross-run memory).

    The counterpart to `ogle incidents`: once the upstream drift is fixed in prod, the
    operator tells Ogle to stop tracking it. Dropping the fingerprint means the *next* time
    it appears (if the fix didn't hold), it pages as a fresh incident — resolve is not a
    mute. Accepts full 16-hex fingerprints or an unambiguous prefix (like a git short SHA).

    Reporting is per-token so a batch of resolves can partially succeed: hits print
    `✅ resolved <fp>`, misses print `_not remembered: <token>_` (not an error — the caller
    may be replaying a list). An ambiguous prefix is a usage error (exit 2): we refuse to
    guess and list the candidates so the operator can retype with more characters.

    `--dry-run` previews the SAME per-token resolution — hits print `👀 would resolve <fp>`,
    misses and ambiguity behave identically (ambiguity still exits 2) — but the store is
    never mutated or saved. Safe to run the documented batch pipe through it first:
    `ogle incidents --serving-only --fingerprints | xargs ogle resolve --dry-run` shows
    exactly what a real resolve would drop before you drop it.

    A lone `-` token reads fingerprints from stdin (whitespace-separated), so the same
    pipe works natively without `xargs` — key on Windows, where `xargs` isn't a built-in:
    `ogle incidents --serving-only --fingerprints | ogle resolve -`.
    """
    store_path = Path(args.store)
    store = BaselineStore.load(store_path)
    dry_run = getattr(args, "dry_run", False)
    resolved: List[str] = []
    for raw in _expand_stdin_fingerprints(args.fingerprint):
        # Strip surrounding whitespace so the documented pipe works cross-platform:
        # `ogle incidents --fingerprints | xargs ogle resolve` — on Windows the emitted lines
        # carry a trailing CR, and a fingerprint never has surrounding whitespace, so trimming
        # is always safe. An all-whitespace token collapses to "" → a reportable miss below
        # (never a mass wipe), preserving the empty-token guard.
        token = raw.strip()
        fp, candidates = _resolve_fingerprint(store, token)
        if candidates:
            print(
                f"ogle resolve: '{token}' is ambiguous — matches "
                f"{len(candidates)} incidents: {', '.join(candidates)}",
                file=sys.stderr,
            )
            return 2
        if fp is None:
            _emit(f"_not remembered: {token}_")
            continue
        # --dry-run resolves the token to a fingerprint (so the preview is exact) but never
        # forgets it — the store is left untouched and the save below is skipped.
        if not dry_run:
            store.forget_incident(fp)
        resolved.append(fp)
        _emit(f"👀 would resolve `{fp}`" if dry_run else f"✅ resolved `{fp}`")
    if resolved and not dry_run:
        try:
            store.save(store_path)
        except Exception as exc:
            print(f"ogle resolve: could not save store: {exc}", file=sys.stderr)
            return 2
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    """Drop one or more datasets from the watch-list (their baseline signature + mute state).

    The write-side counterpart to `ogle baselines`: once a dataset is decommissioned in
    DataHub, its signature would otherwise sit in the watch-list forever. `forget` prunes it
    so `ogle baselines` and the next `ogle check` walk stay honest. Unlike `resolve` (which
    drops a drift *event* by fingerprint), `forget` drops the *dataset* by URN — and also
    clears any mute/snooze on it, since a mute pointing at a gone dataset is dead weight.

    Reporting is per-token so a batch can partially succeed: hits print `✅ forgot <urn>`,
    misses print `_not watched: <urn>_` (not an error — the caller may be replaying a list).
    URNs are matched exactly (they aren't hash prefixes; the documented pipe emits them
    whole), so there's no prefix ambiguity to guard against.

    A lone `-` token reads URNs from stdin (whitespace-separated), so the watch-list selector
    pipes natively without `xargs` — key on Windows:
        ogle baselines --grep decommissioned --urns | ogle forget -

    `--dry-run` previews the SAME per-token outcome — hits print `👀 would forget <urn>` —
    but the store is never mutated or saved, so a batch pipe can be checked before it commits.
    """
    store_path = Path(args.store)
    store = BaselineStore.load(store_path)
    dry_run = getattr(args, "dry_run", False)
    forgotten: List[str] = []
    for raw in _expand_stdin_fingerprints(args.urn):
        # Trim surrounding whitespace so the cross-platform pipe works (Windows lines carry a
        # trailing CR); a URN never has surrounding whitespace. An all-whitespace token
        # collapses to "" → a reportable miss below, never a mass wipe.
        urn = raw.strip()
        if not urn or urn not in store:
            _emit(f"_not watched: {urn}_")
            continue
        # --dry-run reports the exact outcome but leaves the store untouched (save skipped).
        if not dry_run:
            store.forget_baseline(urn)
        forgotten.append(urn)
        _emit(f"👀 would forget `{urn}`" if dry_run else f"✅ forgot `{urn}`")
    if forgotten and not dry_run:
        try:
            store.save(store_path)
        except Exception as exc:
            print(f"ogle forget: could not save store: {exc}", file=sys.stderr)
            return 2
    return 0


def cmd_incidents(args: argparse.Namespace) -> int:
    """List the incidents Ogle currently remembers — its cross-run drift memory.

    Read-only: surfaces the same `seen_incidents` memory that debounces `ogle check` so an
    operator can see what open drift Ogle is tracking, how often each has recurred, and
    which touch a serving path — without re-walking DataHub. An incident stays remembered
    until its drift resolves (its fingerprint stops recurring) or it is explicitly forgotten.
    """
    store = BaselineStore.load(Path(args.store))
    # `--sort` picks the ordering axis (default: worst-severity-first, the triage order).
    # It shapes the list AND what `--limit` calls the "top N"; --summary/--fail-on ignore
    # order (a rollup and a floor gate don't depend on it).
    sort_key = _INCIDENT_SORTS[getattr(args, "sort", None) or "severity"]
    all_records = sorted(store.incidents(), key=sort_key, reverse=True)

    # Triage filters (mirror `check --fail-on`): a floor on severity, recurrence, and/or
    # serving-only.
    min_rank = Severity(args.min_severity).rank if args.min_severity else None
    serving_only = getattr(args, "serving_only", False)
    min_count = getattr(args, "min_count", None)
    needle = getattr(args, "grep", None)

    # `--stale AGE`: keep only drift last seen longer ago than AGE (e.g. `--stale 7d`) — the
    # resolve/forget candidates that stopped recurring. Parsed against a single `now` so the
    # cutoff and the age display below share one clock. A bad duration is a hard error (exit
    # 2) rather than a silent no-op that would read as "nothing is stale".
    now = time.time()
    stale_raw = getattr(args, "stale", None)
    stale_before: Optional[float] = None
    if stale_raw is not None:
        age = _parse_age(stale_raw)
        if age is None:
            _emit("_--stale wants a duration like 7d, 12h, 30m, or 2w._")
            return 2
        stale_before = now - age

    filtered = (
        getattr(args, "min_severity", None) is not None
        or serving_only
        or min_count is not None
        or needle is not None
        or stale_before is not None
    )
    records = [
        r
        for r in all_records
        if _incident_passes(r, min_rank, serving_only, min_count, needle, stale_before)
    ]

    limit = getattr(args, "limit", None)
    if limit is not None and limit < 1:
        _emit("_--limit must be a positive integer._")
        return 2

    # CI/scheduled health gate. Evaluated on the whole filtered set (NOT the --limit cap)
    # so a display cap can never flip the verdict; 0 when no --fail-on is given. Every
    # "shown" path below returns `gate_rc` instead of a bare 0.
    gate_rc = 1 if _incidents_gate_fail(records, getattr(args, "fail_on", None)) else 0

    # `--fingerprints`: plain machine output — just each surviving incident's fingerprint,
    # one per line, in --sort order and honoring every filter + --limit. Turns the read-side
    # `ogle incidents` into a selector that feeds the write-side `ogle resolve`, e.g.
    #   ogle incidents --serving-only --min-severity high --fingerprints | xargs ogle resolve
    # Deliberately overrides --summary/--json (a rollup has no per-incident ids; this IS the
    # scriptable form) and stays SILENT on an empty set so a pipe gets nothing to act on
    # rather than a prose "no incidents" line. Still returns gate_rc so it composes with
    # --fail-on as a health gate.
    if getattr(args, "fingerprints", False):
        capped = records[:limit] if limit is not None else records
        for r in capped:
            _emit(r["fingerprint"])
        return gate_rc

    # `--summary`: an aggregate rollup of the (filtered) set instead of the per-incident list.
    # `--limit` is deliberately NOT applied here: the rollup describes the whole filtered set,
    # so capping it would under-count severity/serving/recurring totals.
    if getattr(args, "summary", False):
        summary = _incident_summary(records)
        if args.json:
            _emit(json.dumps({"summary": summary}, indent=2, sort_keys=True))
            return gate_rc
        if not records:
            # Same empty-vs-filtered distinction as the list view so a hidden set never
            # reads as an empty store.
            if filtered and all_records:
                _emit(f"_no incidents match the filter ({len(all_records)} remembered)._")
            else:
                _emit("_no incidents remembered yet._")
            return gate_rc
        sev = summary["by_severity"]
        _emit(f"**Incident memory summary — {summary['total']} remembered:**")
        _emit(
            f"- 🔴 high: {sev['high']} · 🟠 medium: {sev['medium']} · "
            f"🟡 low: {sev['low']} · • unknown: {sev['unknown']}"
        )
        _emit(f"- ⚠️ serving-path: {summary['serving']}")
        _emit(f"- 🔁 recurring (seen ≥2×): {summary['recurring']}")
        _emit(f"- total sightings: {summary['total_sightings']}")
        if gate_rc and not args.json:
            _emit(f"_open drift at/above --fail-on {args.fail_on} remembered — exit 1._")
        return gate_rc

    # `--limit`: cap to the top N after sort+filter (records are already worst-first).
    capped = records[:limit] if limit is not None else records

    if args.json:
        _emit(json.dumps({"incidents": capped}, indent=2, sort_keys=True))
        return gate_rc
    if not records:
        # Distinguish "memory is empty" from "filters hid everything" so the operator
        # knows whether to widen the filter vs. that there's genuinely nothing tracked.
        if filtered and all_records:
            _emit(f"_no incidents match the filter ({len(all_records)} remembered)._")
        else:
            _emit("_no incidents remembered yet._")
        return gate_rc

    # When --limit hides some, say so ("Top N of M") so a capped view never reads as the
    # full remembered set.
    if len(capped) < len(records):
        _emit(f"**Top {len(capped)} of {len(records)} remembered incident(s):**")
    else:
        _emit(f"**{len(records)} remembered incident(s):**")
    for r in capped:
        sev = r.get("severity") or "unknown"
        mark = _INCIDENT_SEV_MARK.get(sev, "•")  # bullet for unknown/legacy
        count = int(r.get("count", 0))
        seen = f"seen {count}×"  # e.g. "seen 3×"
        nd = int(r.get("datasets", 0))
        dpart = f" · {nd} dataset(s)" if nd else ""
        serv = " · ⚠️ serving" if r.get("serving") else ""
        # Relative age of the most recent sighting, when Ogle has a timestamp for it.
        # Legacy/untimed records simply omit it rather than fake an age. "just now" reads
        # on its own; older ages take the "… ago" suffix.
        ls = r.get("last_seen")
        if ls is None:
            apart = ""
        else:
            age = _fmt_age(now - ls)
            apart = f" · last seen {age}" if age == "just now" else f" · last seen {age} ago"
        title = r.get("title") or "(drift)"
        _emit(
            f"- {mark} **{sev}** — {title} · {seen}{dpart}{serv}{apart}  `{r['fingerprint']}`"
        )
    if gate_rc and not args.json:
        _emit(f"_open drift at/above --fail-on {args.fail_on} remembered — exit 1._")
    return gate_rc


def _baseline_totals(store: "BaselineStore") -> dict:
    """Aggregate the watch-list into blast-radius totals for the status rollup.

    `fields`/`rows` sum only over baselines that carry a signature with a known value;
    `unknown_rows` counts baselines whose `row_count` is None (never captured / not tracked),
    so a small `rows` total next to a large `unknown_rows` reads as "coverage gap", not
    "low volume". Mirrors how `baselines --sort rows` sinks unknown-row datasets last.
    """
    urns = store.urns()
    total_fields = 0
    total_rows = 0
    unknown_rows = 0
    for u in urns:
        sig = store.get_baseline(u)
        if not sig:
            unknown_rows += 1
            continue
        total_fields += len(sig.schema_fields)
        if sig.row_count is None:
            unknown_rows += 1
        else:
            total_rows += sig.row_count
    return {
        "watching": len(urns),
        "fields": total_fields,
        "rows": total_rows,
        "unknown_rows": unknown_rows,
    }


def cmd_status(args: argparse.Namespace) -> int:
    """One-glance health snapshot of the whole store — both halves plus mutes.

    Read-only. `baselines` shows the watch-list, `incidents` shows remembered drift, and
    `muted` shows snoozes — this unifies all three into a single rollup so an operator (or a
    scheduled wrapper) can answer "what is Ogle holding right now?" in one call, without
    re-walking DataHub. Reuses `_incident_summary` so the severity/serving/recurring counts
    match what `incidents --summary` reports on the same store.
    """
    store = BaselineStore.load(Path(args.store))
    totals = _baseline_totals(store)
    incident_records = store.incidents()
    inc = _incident_summary(incident_records)
    muted = store.muted(time.time())  # active snoozes only (expired excluded)

    # CI/scheduled health gate. Turns the whole-store rollup into an exit-code check so a
    # cron/CI wrapper that runs `ogle status` to answer "what is Ogle holding right now?" can
    # also PAGE on it. Evaluated against every remembered incident (status has no filters), so
    # it fails while any drift at/above the floor is still un-resolved — the same drift-memory
    # semantics as `incidents --fail-on`, just over the unfiltered set. 0 when --fail-on is
    # unset, so the default snapshot stays exit 0. Independent of --json / the empty-store path.
    gate_rc = (
        1 if _incidents_gate_fail(incident_records, getattr(args, "fail_on", None)) else 0
    )

    if args.json:
        _emit(
            json.dumps(
                {
                    "status": {
                        "store": str(args.store),
                        "baselines": totals,
                        "incidents": inc,
                        "muted": len(muted),
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )
        return gate_rc

    # Nothing captured, nothing remembered, nothing muted → first-run / empty store. Say so
    # plainly rather than printing a wall of zeros that reads like a populated-but-quiet store.
    # (gate_rc is necessarily 0 here — an empty store holds no incident to trip the floor.)
    if totals["watching"] == 0 and inc["total"] == 0 and not muted:
        _emit(f"_store `{args.store}` is empty — run `ogle check` to start watching._")
        return gate_rc

    sev = inc["by_severity"]
    _emit(f"**Ogle store status — `{args.store}`**")
    rpart = f"{totals['rows']} row(s)"
    if totals["unknown_rows"]:
        rpart += f" ({totals['unknown_rows']} unknown)"
    _emit(
        f"- 📊 watching: {totals['watching']} dataset(s) · "
        f"{totals['fields']} field(s) · {rpart}"
    )
    _emit(
        f"- 🧠 incidents remembered: {inc['total']} "
        f"(🔴 {sev['high']} · 🟠 {sev['medium']} · 🟡 {sev['low']} · • {sev['unknown']})"
    )
    _emit(
        f"- ⚠️ serving-path: {inc['serving']} · 🔁 recurring: {inc['recurring']} · "
        f"total sightings: {inc['total_sightings']}"
    )
    _emit(f"- 🔇 muted: {len(muted)} active")
    if gate_rc:
        _emit(f"_open drift at/above --fail-on {args.fail_on} remembered — exit 1._")
    return gate_rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogle",
        description="Ogle — ML-lineage drift agent for DataHub.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"ogle {__version__}")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser(
        "check",
        help="Run a drift check: walk lineage, diff against baselines, alert on new drift.",
    )
    check.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"Baseline/incident store JSON (default: {DEFAULT_STORE}).",
    )
    src = check.add_argument_group("input (choose live walk OR an offline signatures file)")
    src.add_argument(
        "--signatures",
        metavar="FILE",
        help="Offline: read DatasetSignatures from JSON instead of a live DataHub walk.",
    )
    src.add_argument(
        "--gms",
        default="http://localhost:8080",
        help="Live: DataHub GMS server URL (default: http://localhost:8080).",
    )
    src.add_argument(
        "--models",
        nargs="*",
        metavar="URN",
        help="Live: explicit mlModel URNs to walk.",
    )
    src.add_argument(
        "--discover",
        action="store_true",
        help="Live: also auto-discover every IN_SERVICE model and walk it.",
    )
    check.add_argument(
        "--serving",
        nargs="*",
        metavar="URN",
        help="Extra dataset URNs to treat as serving (severity-escalated).",
    )
    tune = check.add_argument_group("sensitivity (tune per deployment)")
    tune.add_argument(
        "--volume-threshold",
        type=float,
        metavar="FRAC",
        help="Relative row-count change that counts as volume drift (default: 0.30 = ±30%%).",
    )
    tune.add_argument(
        "--null-threshold",
        type=float,
        metavar="FRAC",
        help="Absolute null-fraction increase that counts as quality drift (default: 0.20).",
    )
    tune.add_argument(
        "--no-serving-escalation",
        action="store_true",
        help="Do not bump severity for sources feeding a deployed model (default: escalate).",
    )
    check.add_argument(
        "--fail-on",
        choices=["low", "medium", "high"],
        metavar="SEV",
        help=(
            "CI gate: exit 1 only when a NEW incident's overall severity is at least SEV "
            "(low|medium|high). Below-floor incidents are still reported and tagged but "
            "exit 0. Default: any new incident exits 1."
        ),
    )
    check.add_argument(
        "--no-update",
        action="store_true",
        help="Read-only probe: do not advance baselines or incident memory.",
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit the DriftReport as JSON instead of markdown.",
    )
    check.add_argument(
        "--write-back",
        action="store_true",
        help=(
            "On a new incident (exit 1), stamp affected datasets and their downstream "
            "mlModels with `urn:li:tag:ogle-drift-flagged` in DataHub. Requires --gms."
        ),
    )
    check.add_argument(
        "--write-back-severity",
        action="store_true",
        help=(
            "With --write-back, ALSO stamp a per-severity tag "
            "(`urn:li:tag:ogle-drift-high|medium|low`) so DataHub can be filtered to the "
            "worst drift. A model inherits the worst severity of the datasets feeding it."
        ),
    )
    check.add_argument(
        "--retract-cleared",
        action="store_true",
        help=(
            "Close the loop: on this run, REMOVE Ogle's drift tags from datasets that were "
            "checked and are now healthy (and their downstream mlModels no longer fed by any "
            "still-drifting dataset), so a recovered asset stops carrying a stale flag. Runs "
            "even when there is no new incident. Requires a live walk (--gms). Idempotent — "
            "skips entities that aren't Ogle-tagged."
        ),
    )
    check.add_argument(
        "--narrate",
        nargs="?",
        const="ollama",
        metavar="SPEC",
        help=(
            "After the report, add an LLM-phrased incident summary grounded in the "
            "computed facts. Optional SPEC picks the model (default: 'ollama' = local "
            "qwen3:latest); also 'ollama:<model>' or '<...>@http://host:11434'. Falls "
            "back to the deterministic report if the model is unreachable. Ignored with "
            "--json."
        ),
    )
    check.set_defaults(func=cmd_check)

    # `ogle demo` — one-command, keyless judge repro over the bundled fixtures.
    demo = sub.add_parser(
        "demo",
        help="Zero-setup offline drift demo (no DataHub/API key) — seeds, then alerts.",
    )
    demo.add_argument(
        "--narrate",
        nargs="?",
        const="ollama",
        metavar="SPEC",
        help=(
            "Also show the LLM root-cause summary (feature #2) after the alert. Optional "
            "SPEC picks the model (default: 'ollama' = local qwen3:latest); also "
            "'ollama:<model>' or '<...>@http://host:11434'. Falls back to the deterministic "
            "summary if the model is unreachable, so the demo stays keyless."
        ),
    )
    demo.set_defaults(func=cmd_demo)

    # `ogle mute|unmute|muted` — manage the known-false-positive list (feature #3). A muted
    # dataset is still tracked but never pages, so a chronically-noisy asset stops crying wolf.
    mute = sub.add_parser(
        "mute",
        help="Mark a dataset as a known false positive so its drift stops paging.",
    )
    mute.add_argument("urn", help="Dataset URN to mute.")
    mute.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    mute.add_argument(
        "--for",
        dest="for_days",
        type=float,
        metavar="DAYS",
        help="Snooze for DAYS instead of muting permanently — auto-expires so it can't "
        "become a permanent blind spot.",
    )
    mute.add_argument(
        "--for-hours",
        dest="for_hours",
        type=float,
        metavar="HOURS",
        help="Snooze for HOURS instead of DAYS (mutually exclusive with --for).",
    )
    mute.set_defaults(func=cmd_mute)

    unmute = sub.add_parser("unmute", help="Un-mute a dataset so its drift can page again.")
    unmute.add_argument("urn", help="Dataset URN to un-mute.")
    unmute.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    unmute.set_defaults(func=cmd_unmute)

    muted = sub.add_parser("muted", help="List datasets currently muted in the store.")
    muted.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    muted.add_argument("--json", action="store_true", help="Emit the list as JSON.")
    muted.add_argument(
        "--urns",
        action="store_true",
        help="Print ONLY the muted URNs (one per line) — pipe into a write-side command, "
        "e.g. `ogle muted --urns | xargs -n1 ogle unmute` to lift a whole false-positive "
        "campaign at once. Overrides --json; silent on an empty set.",
    )
    muted.set_defaults(func=cmd_muted)

    # `ogle status` — one-glance rollup of the whole store (read-only): the watch-list, the
    # incident memory, and active mutes in a single view. The top-level "what is Ogle holding
    # right now?" that unifies `baselines` + `incidents` + `muted`.
    status = sub.add_parser(
        "status",
        help="One-glance health snapshot of the store (watch-list + incidents + mutes).",
    )
    status.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    status.add_argument("--json", action="store_true", help="Emit the snapshot as JSON.")
    status.add_argument(
        "--fail-on",
        choices=["low", "medium", "high"],
        default=None,
        help="Exit 1 if any remembered incident is at/above this severity (CI/scheduled "
        "whole-store health gate). Gates on every remembered incident; independent of --json.",
    )
    status.set_defaults(func=cmd_status)

    # `ogle baselines` — inspect the OTHER half of the store (read-only): the datasets Ogle
    # has a signature for and will diff the next walk against. `incidents` shows remembered
    # drift; this shows what's under watch.
    baselines = sub.add_parser(
        "baselines",
        help="List datasets Ogle has a baseline signature for (what it's watching).",
    )
    baselines.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    baselines.add_argument(
        "--grep",
        metavar="TEXT",
        default=None,
        help="Only show datasets whose URN contains TEXT (case-insensitive).",
    )
    baselines.add_argument(
        "--sort",
        choices=["urn", "fields", "rows"],
        default="urn",
        help="Ordering axis: urn (default, alphabetical), fields (widest schema first), "
        "or rows (highest volume first) — surface the highest-blast-radius watched datasets. "
        "Honored by --urns/--json too.",
    )
    baselines.add_argument("--json", action="store_true", help="Emit the list as JSON.")
    baselines.add_argument(
        "--urns",
        action="store_true",
        help="Print ONLY the URNs (one per line), honoring --grep — pipe into a write-side "
        "command, e.g. `ogle baselines --grep serving --urns | xargs -n1 ogle mute`. "
        "Overrides --json.",
    )
    baselines.set_defaults(func=cmd_baselines)

    # `ogle show` — drill into ONE watched dataset (read-only): the full memorized signature
    # (fields + types + null fractions + row count + hash + provenance) and its mute state.
    # `baselines` lists the watch-list a line each; this opens a single URN — the detail no
    # other view shows. Keys on an exact URN, so it pairs with `baselines --urns`.
    show = sub.add_parser(
        "show",
        help="Show one dataset's full memorized signature + mute state (drill into a baseline).",
    )
    show.add_argument(
        "urn",
        help="Dataset URN to inspect (exact, as emitted by `ogle baselines --urns`).",
    )
    show.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    show.add_argument("--json", action="store_true", help="Emit the full signature as JSON.")
    show.set_defaults(func=cmd_show)

    # `ogle diff` — read-only field-level diff of one dataset: stored baseline vs a candidate
    # signatures file. The investigative "what exactly changed?" step after a page; unlike
    # `check` it records nothing and advances no baseline.
    diff = sub.add_parser(
        "diff",
        help="Diff one dataset's stored baseline against a candidate signatures file (read-only).",
    )
    diff.add_argument(
        "urn",
        help="Dataset URN to diff (exact, as emitted by `ogle baselines --urns`).",
    )
    diff.add_argument(
        "--signatures",
        required=True,
        metavar="FILE",
        help="Candidate signatures JSON (same shape `ogle check --signatures` reads); the "
        "URN's current signature is compared against its stored baseline.",
    )
    diff.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    diff.add_argument("--json", action="store_true", help="Emit the diff as JSON.")
    diff.set_defaults(func=cmd_diff)

    # `ogle incidents` — inspect Ogle's cross-run incident memory (read-only): what drift
    # it's tracking, recurrence counts, serving impact. The visible side of the same
    # `seen_incidents` memory that debounces `ogle check`.
    incidents = sub.add_parser(
        "incidents",
        help="List the incidents Ogle remembers (its cross-run drift memory).",
    )
    incidents.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    incidents.add_argument(
        "--min-severity",
        choices=["low", "medium", "high"],
        default=None,
        help="Only show incidents at or above this severity (drops unknown/legacy severity).",
    )
    incidents.add_argument(
        "--serving-only",
        action="store_true",
        help="Only show incidents that touch a serving path.",
    )
    incidents.add_argument(
        "--min-count",
        type=int,
        metavar="N",
        default=None,
        help="Only show incidents seen at least N times (surfaces chronic/flapping drift).",
    )
    incidents.add_argument(
        "--grep",
        metavar="TEXT",
        default=None,
        help="Only show incidents whose title or fingerprint contains TEXT "
        "(case-insensitive) — find specific drift in a large memory.",
    )
    incidents.add_argument(
        "--stale",
        metavar="AGE",
        default=None,
        help="Only show incidents last seen longer ago than AGE (e.g. 7d, 12h, 30m, 2w) — "
        "drift that stopped recurring, i.e. resolve/forget candidates. Skips legacy "
        "records with no recorded age.",
    )
    incidents.add_argument(
        "--sort",
        choices=["severity", "count", "datasets", "recent"],
        default="severity",
        help="Ordering axis: severity (default, worst first), count (most-recurring "
        "first), datasets (broadest blast radius first), or recent (freshest sighting "
        "first). Also defines --limit's top N.",
    )
    incidents.add_argument(
        "--summary",
        action="store_true",
        help="Show an aggregate rollup (counts by severity, serving, recurring) instead of the list.",
    )
    incidents.add_argument(
        "--limit",
        type=int,
        metavar="N",
        default=None,
        help="Show only the top N incidents (by the chosen --sort). Ignored by --summary.",
    )
    incidents.add_argument(
        "--fail-on",
        choices=["low", "medium", "high"],
        default=None,
        help="Exit 1 if any remembered incident is at/above this severity (CI/scheduled "
        "drift-memory health gate). Composes with the filters; independent of --limit.",
    )
    incidents.add_argument("--json", action="store_true", help="Emit the list as JSON.")
    incidents.add_argument(
        "--fingerprints",
        action="store_true",
        help="Print ONLY the fingerprints (one per line), honoring the filters/--sort/--limit "
        "— pipe into `ogle resolve`, e.g. `ogle incidents --serving-only --fingerprints | "
        "xargs ogle resolve`. Overrides --summary/--json.",
    )
    incidents.set_defaults(func=cmd_incidents)

    # `ogle resolve` — the operator's counterpart to `ogle incidents`: drop a fixed drift
    # from cross-run memory so it stops appearing in `ogle incidents` and pages fresh if it
    # recurs. Accepts full fingerprints or an unambiguous prefix (git-short-SHA style).
    resolve = sub.add_parser(
        "resolve",
        help="Drop a remembered incident (its drift is fixed) — takes fingerprints or prefixes.",
    )
    resolve.add_argument(
        "fingerprint",
        nargs="+",
        help="Incident fingerprint(s) from `ogle incidents` (full 16-hex or unambiguous "
        "prefix). A lone `-` reads them from stdin, so `ogle incidents --fingerprints | "
        "ogle resolve -` works without xargs (native on Windows).",
    )
    resolve.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    resolve.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which incidents would be resolved without dropping anything from "
        "memory (store is never modified). Pipe-safe: dry-run a batch before committing it.",
    )
    resolve.set_defaults(func=cmd_resolve)

    # `ogle forget` — the write-side counterpart to `ogle baselines`: prune a decommissioned
    # dataset from the watch-list (drops its baseline signature + any mute state). Mirrors
    # `resolve` (stdin `-`, --dry-run) but keys on URN, not fingerprint.
    forget = sub.add_parser(
        "forget",
        help="Drop a dataset from the watch-list (its baseline is gone/decommissioned).",
    )
    forget.add_argument(
        "urn",
        nargs="+",
        help="Dataset URN(s) from `ogle baselines --urns` (matched exactly). A lone `-` reads "
        "them from stdin, so `ogle baselines --grep old --urns | ogle forget -` works without "
        "xargs (native on Windows).",
    )
    forget.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    forget.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which datasets would be forgotten without dropping anything from the "
        "store (never modified). Pipe-safe: dry-run a batch before committing it.",
    )
    forget.set_defaults(func=cmd_forget)

    # `ogle watch` — one scheduler tick that pages once on a new incident.
    from .watch import build_watch_args

    build_watch_args(sub)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
