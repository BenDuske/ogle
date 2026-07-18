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
    urns = store.muted(now)
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
}


def _incident_passes(
    rec: dict,
    min_rank: Optional[int],
    serving_only: bool,
    min_count: Optional[int] = None,
) -> bool:
    """True if a remembered incident survives the `ogle incidents` triage filters.

    `min_rank` is a `Severity.rank` floor (None = no floor). A record whose severity is
    unknown/legacy ranks -1, so ANY `--min-severity` floor drops it — asking for a floor
    is asking to hide the un-triageable. `serving_only` keeps only serving-path incidents.
    `min_count` is a recurrence floor (None = no floor): keeps only incidents seen at least
    that many times, surfacing the chronic/flapping drift that keeps coming back. All filters
    are ANDed; passing none keeps everything.
    """
    if serving_only and not rec.get("serving"):
        return False
    if min_count is not None and int(rec.get("count", 0)) < min_count:
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
    """
    store_path = Path(args.store)
    store = BaselineStore.load(store_path)
    resolved: List[str] = []
    for token in args.fingerprint:
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
        store.forget_incident(fp)
        resolved.append(fp)
        _emit(f"✅ resolved `{fp}`")
    if resolved:
        try:
            store.save(store_path)
        except Exception as exc:
            print(f"ogle resolve: could not save store: {exc}", file=sys.stderr)
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
    filtered = (
        getattr(args, "min_severity", None) is not None
        or serving_only
        or min_count is not None
    )
    records = [
        r for r in all_records if _incident_passes(r, min_rank, serving_only, min_count)
    ]

    limit = getattr(args, "limit", None)
    if limit is not None and limit < 1:
        _emit("_--limit must be a positive integer._")
        return 2

    # CI/scheduled health gate. Evaluated on the whole filtered set (NOT the --limit cap)
    # so a display cap can never flip the verdict; 0 when no --fail-on is given. Every
    # "shown" path below returns `gate_rc` instead of a bare 0.
    gate_rc = 1 if _incidents_gate_fail(records, getattr(args, "fail_on", None)) else 0

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
        title = r.get("title") or "(drift)"
        _emit(f"- {mark} **{sev}** — {title} · {seen}{dpart}{serv}  `{r['fingerprint']}`")
    if gate_rc and not args.json:
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
    muted.set_defaults(func=cmd_muted)

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
        "--sort",
        choices=["severity", "count", "datasets"],
        default="severity",
        help="Ordering axis: severity (default, worst first), count (most-recurring "
        "first), or datasets (broadest blast radius first). Also defines --limit's top N.",
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
        help="Incident fingerprint(s) from `ogle incidents` (full 16-hex or unambiguous prefix).",
    )
    resolve.add_argument(
        "--store", default=DEFAULT_STORE, help=f"Store JSON (default: {DEFAULT_STORE})."
    )
    resolve.set_defaults(func=cmd_resolve)

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
