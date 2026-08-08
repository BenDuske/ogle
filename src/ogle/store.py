"""Baseline store — Ogle's memory between runs.

Drift detection is a *diff*: it needs to remember what each dataset looked like last time so
it can tell what changed. This module is that memory. It persists two things:

  * BASELINES     — the last `DatasetSignature` seen per dataset URN. Next run scores the
                    fresh signature against this to surface schema/volume/quality drift.
  * SEEN INCIDENTS — the set of incident fingerprints Ogle has already reported, with an
                    observation count. Lets a scheduled run tell a *new* problem from one it
                    already alerted on, so Ben isn't paged every 10 minutes for the same drift.
  * MUTED URNS     — datasets an operator has marked as known-noisy false positives ("this
                    dashboard bounces every Monday, ignore"). Their drift is still tracked
                    (baselines advance) but never pages — the difference from dedup is that a
                    muted asset stays silent even when it flaps with a *fresh* fingerprint.

The on-disk format is a single JSON file, written atomically (temp + replace) so a crash
mid-write can't corrupt the baseline. That file is the concrete "Aegis memory" backing for
Ogle: when Aegis's salience memory is wired in W3, `BaselineStore` is the seam that swaps a
JSON path for an Aegis-backed key/value without the scorer or pipeline knowing.

Everything here is pure and clock-free: the store never stamps a timestamp of its own (any
`computed_at` provenance rides along inside the signature the caller built). That keeps a
run reproducible and the file diffable.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

from .signature import DatasetSignature

# Bump when the on-disk shape changes so a stale file can be detected rather than misread.
STORE_VERSION = 1


def _finite_mute_time(value: object, *, key: str, urn: str) -> float:
    """Coerce a mute timestamp (epoch seconds) to a *finite* float, rejecting NaN/Infinity.

    `json.loads` accepts the non-standard `NaN`/`Infinity`/`-Infinity` tokens by default, so a
    hand-edited or truncated store can carry a non-finite mute time that a bare `float(...)`
    happily preserves. That is silently wrong for a suppression list — the same class of bug the
    `muted_urns` list-guard was written to stop:

      * a NaN `muted_until` makes `is_muted`'s `exp > now` always False, so the snooze VANISHES
        and the dataset it was meant to silence starts paging again — while `purge_expired_mutes`'
        `exp <= now` is also False, so the dead entry never clears (a zombie snooze);
      * an Infinity `muted_until` can NEVER expire, masquerading as a permanent mute through the
        timed-snooze channel that the `muted_urns` set is supposed to own.

    Reject both with the same ValueError the sibling shape guards raise, so `load()` quarantines
    the foreign file rather than misreading it. (A non-numeric value already raises
    ValueError/TypeError from `float(...)`, which `load()` catches identically.)

    A JSON boolean is the one non-number that slips past `float(...)`: `bool` is a subclass of
    `int`, so `float(True)` is a silent, *finite* `1.0` (and `float(False)` a `0.0`) — both pass
    `math.isfinite`. A hand-edited/truncated `"muted_until": {urn: true}` would then read as epoch
    second 1.0 (Jan 1 1970), which `is_muted`'s `exp > now` treats as long-expired: the snooze
    VANISHES and the dataset it was meant to silence starts paging again — the exact silent-
    suppression-loss this guard exists to stop, just wearing a bool instead of a NaN. Reject it
    explicitly (before `float(...)` can launder it) with the same ValueError contract.
    """
    return _finite_epoch(value, context=f"'{key}' time for {urn!r}")


def _finite_epoch(value: object, *, context: str) -> float:
    """Coerce an epoch-seconds value to a *finite* float, rejecting bool/NaN/Infinity.

    The shared core behind every store timestamp guard: `json.loads` accepts the non-standard
    `NaN`/`Infinity`/`-Infinity` tokens, and `bool` is an `int` subclass so `float(True)` is a
    silent finite `1.0` — both launder a hand-edited/truncated store past a bare `float(...)`.
    A non-finite or bool-derived epoch reads as garbage everywhere a time is compared (age math,
    recency/longevity sorts, stale/fresh windows), so reject it with the same ValueError contract
    the sibling shape guards raise and let `load()` quarantine the foreign file. `context` names
    which field/URN for the operator; a non-numeric value already raises from `float(...)`, which
    `load()` catches identically. See `_finite_mute_time` for the mute-side rationale in full."""
    if isinstance(value, bool):
        raise ValueError(
            f"baseline store {context} must be a number, got bool "
            f"(refusing to misread a truncated/foreign file)"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(
            f"baseline store {context} must be finite, got {f} "
            f"(refusing to misread a truncated/foreign file)"
        )
    return f


def _nonneg_int(value: object, *, context: str) -> int:
    """Coerce a persisted incident counter to a non-negative int, rejecting the bool case.

    `count` (observation tally) and `datasets` (assets in the incident) are non-negative ints.
    `bool` is an `int` subclass, so a bare `int(True)` is a silent `1` — a truncated/foreign
    `{"count": true}` would forge a real "seen once" / "1 dataset" reading with NO exception, the
    same silent-misread class `_finite_epoch` (timestamps) and signature.py's `row_count`/
    `_require_finite_real` guards close. A str/None/NaN already raises ValueError/TypeError from
    `int(...)`, which `load()`'s recovery net catches identically; only the in-type bool (and a
    negative, which is visible nonsense but still not a valid tally) slip through, so reject both
    with the same ValueError contract the sibling shape guards raise and let `load()` quarantine
    the file. `context` names which field for the operator."""
    if isinstance(value, bool):
        raise ValueError(
            f"baseline store {context} must be an integer, got bool "
            f"(refusing to misread a truncated/foreign file)"
        )
    n = int(value)
    if n < 0:
        raise ValueError(
            f"baseline store {context} must be >= 0, got {n} "
            f"(refusing to misread a truncated/foreign file)"
        )
    return n


def _fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so a completed rename is durable.

    `os.replace` swaps the temp file into place atomically, but on POSIX the *directory
    entry* that now points at the new file is itself only persisted once the parent
    directory is fsync'd. Without this, a crash right after the rename can lose the rename
    even though the file's bytes were already flushed — the store reverts to its pre-save
    name (or the entry vanishes). This closes that last gap in the atomic-write guarantee.

    Best-effort by design: directory fsync is not portable. On Windows (and some network
    filesystems) opening a directory for fsync raises, and rename durability there rests on
    the platform's own semantics — so those errors are swallowed rather than failing a save
    whose data is already on disk.
    """
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return  # platform/filesystem doesn't allow opening a dir for fsync (e.g. Windows)
    try:
        os.fsync(dir_fd)
    except OSError:
        pass  # data is already durable; a failed dir-fsync must not sink the save
    finally:
        os.close(dir_fd)


@dataclass
class _IncidentRecord:
    """What Ogle remembers about one incident fingerprint across runs.

    `count` (recurrence) is the load-bearing dedup field. The rest is human-facing
    provenance so `ogle incidents` can show WHAT the remembered drift was rather than an
    opaque 16-hex fingerprint — it reflects the *latest* sighting, since a recurring
    incident's shape can shift. All provenance is additive: a record written by an older
    Ogle carries only `count` and loads with the rest as defaults (no STORE_VERSION bump).
    """

    count: int = 0
    severity: Optional[str] = None   # overall severity at last sighting ("high"/"medium"/"low")
    title: Optional[str] = None      # incident headline at last sighting
    datasets: int = 0                # number of datasets in the incident at last sighting
    serving: bool = False            # whether a serving path was impacted at last sighting
    last_seen: Optional[float] = None  # epoch-seconds of the most recent sighting (None = legacy/untimed)
    first_seen: Optional[float] = None  # epoch-seconds of the FIRST sighting — the incident's longevity/standing (None = legacy/untimed)
    kinds: List[str] = field(default_factory=list)  # drift dimensions in the incident at last sighting (schema/volume/quality/distribution/freshness); [] = legacy/unknown
    owners: List[str] = field(default_factory=list)  # owner display names (union across the incident's datasets) at last sighting — the "who to page" attribution; [] = unowned/legacy
    urns: List[str] = field(default_factory=list)  # dataset URNs the incident touched at last sighting — the "which assets" provenance behind per-asset history queries; [] = legacy/unknown

    def to_dict(self) -> dict:
        # Serialize provenance only when set so an old bare-count record round-trips
        # unchanged and the on-disk file stays minimal/diffable.
        d: dict = {"count": self.count}
        if self.severity is not None:
            d["severity"] = self.severity
        if self.title is not None:
            d["title"] = self.title
        if self.datasets:
            d["datasets"] = self.datasets
        if self.serving:
            d["serving"] = True
        if self.last_seen is not None:
            d["last_seen"] = self.last_seen
        if self.first_seen is not None:
            d["first_seen"] = self.first_seen
        if self.kinds:
            d["kinds"] = list(self.kinds)
        if self.owners:
            d["owners"] = list(self.owners)
        if self.urns:
            d["urns"] = list(self.urns)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "_IncidentRecord":
        # Each seen_incidents VALUE must be a JSON object. The store-level guard (3844ade)
        # proves the `seen_incidents` *section* is a dict, but a per-entry value can still be
        # a valid-JSON-but-non-object (a list/scalar from a truncated or hand-mangled file
        # that still parses AND carries the right version): `{"seen_incidents": {"fp": [1,2]}}`.
        # That value would reach `data.get(...)` and raise AttributeError — which is NOT in
        # `BaselineStore.load`'s corrupt-recovery net (ValueError/KeyError/TypeError), so a
        # scheduled `ogle check` would crash-loop and go blind on exactly the bad file the net
        # exists for. Normalize to the same ValueError the section/version guards raise so it
        # quarantines (default) or surfaces cleanly (strict) like any other foreign file.
        if not isinstance(data, dict):
            raise ValueError(
                f"incident record must be a JSON object, got {type(data).__name__} "
                f"(refusing to misread a truncated/foreign file)"
            )
        ls = data.get("last_seen")
        fs = data.get("first_seen")
        raw_sev = data.get("severity")
        raw_title = data.get("title")
        raw_kinds = data.get("kinds")
        raw_owners = data.get("owners")
        raw_urns = data.get("urns")
        return cls(
            # count/datasets are non-negative ints. Route through the shared guard so a
            # bool-laundered `{"count": true}` (int(True) == 1) quarantines via load() instead
            # of silently reading as a real 1-observation / 1-dataset record — the counter twin
            # of the last_seen/first_seen finite-epoch guard just below.
            count=_nonneg_int(data.get("count", 0), context="incident record 'count'"),
            # severity/title are human-facing provenance strings. A truncated/foreign file that
            # still parses AND carries the right version can hand back a NON-string here (e.g.
            # `{"title": 42}` or `{"severity": ["high"]}`) — and unlike the numeric fields there
            # is no earlier guard, so the bad type would load clean (load()'s recovery net never
            # trips) and then crash a DOWNSTREAM consumer: `ogle incidents --grep` does
            # `(rec.get("title") or "").lower()`, and `.lower()` on an int raises AttributeError,
            # which is NOT in load()'s catch net — the exact "survives one tier past the guard and
            # crash-loops on a foreign file" failure the sibling shape guards exist to stop. Coerce
            # to str-when-present (None stays None so the to_dict "emit only when set" contract and
            # the legacy/untimed round-trip are preserved), mirroring the str() coercion the
            # kinds/owners/urns lists get just below.
            severity=str(raw_sev) if raw_sev is not None else None,
            title=str(raw_title) if raw_title is not None else None,
            datasets=_nonneg_int(data.get("datasets", 0), context="incident record 'datasets'"),
            serving=bool(data.get("serving", False)),
            # last_seen/first_seen drive every age display, the recent/standing sorts, and the
            # --stale/--fresh/--standing windows + age gauges. A bare float() here would launder a
            # NaN/Infinity/bool epoch from a truncated/foreign file the same way the mute-time path
            # did before _finite_mute_time — and a NaN last_seen is the nastiest: `now - NaN` is
            # False against BOTH the stale and fresh cutoffs, so the incident silently vanishes from
            # both filtered views (and sorts unstably). Route through the shared finite guard so such
            # a file quarantines via load() instead of misreading. None stays None (legacy/untimed).
            last_seen=_finite_epoch(ls, context="incident record 'last_seen'") if ls is not None else None,
            first_seen=_finite_epoch(fs, context="incident record 'first_seen'") if fs is not None else None,
            kinds=[str(k) for k in raw_kinds] if isinstance(raw_kinds, list) else [],
            owners=[str(o) for o in raw_owners] if isinstance(raw_owners, list) else [],
            urns=[str(u) for u in raw_urns] if isinstance(raw_urns, list) else [],
        )


@dataclass
class BaselineStore:
    """Persistable baselines + incident dedup memory.

    Construct with a `path` to enable `save()`/`load()`; an in-memory store (no path) is
    handy for tests and dry runs. Mutating methods do NOT auto-persist — call `save()` when
    a run is complete so a half-finished walk never overwrites good baselines.
    """

    path: Optional[Path] = None
    baselines: Dict[str, DatasetSignature] = field(default_factory=dict)
    seen_incidents: Dict[str, _IncidentRecord] = field(default_factory=dict)
    muted_urns: Set[str] = field(default_factory=set)
    # Timed ("snoozed") mutes: urn -> epoch-seconds expiry. A snooze auto-expires so a
    # "mute this for now" never silently becomes a permanent blind spot. Permanent mutes
    # live in `muted_urns` and always win over a snooze for the same URN.
    muted_until: Dict[str, float] = field(default_factory=dict)
    # Optional human rationale per muted URN ("dashboard bounces every Monday, ignore").
    # A mute with no note is a mystery weeks later — this is where the "why" lives so
    # `ogle muted`/`ogle show` can explain a silence rather than just report it. Keyed by
    # URN, cleared alongside the mute by unmute/forget/expiry, so a reason never outlives
    # the mute it annotates.
    mute_reasons: Dict[str, str] = field(default_factory=dict)
    # When each URN's *current* silence began (epoch seconds). A permanent mute that has been
    # standing for weeks is a bigger blind spot than one set an hour ago, so this is the age
    # axis behind `ogle muted`'s "muted 3d ago" — the accountability twin of mute_reasons'
    # "why". Stamped once when a URN becomes muted and preserved for the life of that
    # continuous silence (a re-annotate/no-op keeps the original start), then cleared alongside
    # the mute by unmute/forget/expiry so a stamp never outlives the mute it dates.
    muted_at: Dict[str, float] = field(default_factory=dict)

    # Runtime-only recovery status (NOT persisted — excluded from to_dict, and from
    # dataclass eq/repr so two stores with identical data still compare equal). Set by
    # `load()` when it had to quarantine a corrupt/foreign file and start fresh, so a
    # caller (e.g. `ogle check`) can warn loudly instead of silently re-baselining blind.
    recovered_from_corruption: bool = field(default=False, compare=False, repr=False)
    corrupt_backup_path: Optional[Path] = field(default=None, compare=False, repr=False)

    # ---- baselines -----------------------------------------------------------------
    def get_baseline(self, urn: str) -> Optional[DatasetSignature]:
        """The last signature seen for `urn`, or None if this dataset is new to Ogle."""
        return self.baselines.get(urn)

    def put_baseline(self, signature: DatasetSignature) -> None:
        """Upsert the current signature as the new baseline for its URN."""
        self.baselines[signature.urn] = signature

    def urns(self) -> List[str]:
        """All dataset URNs Ogle currently has a baseline for (sorted for stable output)."""
        return sorted(self.baselines)

    def forget_baseline(self, urn: str) -> bool:
        """Drop a dataset from the watch-list entirely — its baseline signature and any
        mute/snooze state for it.

        The counterpart to `put_baseline` for a decommissioned dataset: once a table is gone
        from DataHub, its signature would otherwise linger in the watch-list forever (and its
        mute would be an orphan pointing at nothing). Clearing both keeps `baselines`/`muted`
        honest.

        Incidents are keyed by fingerprint (a drift *event*), not by URN, so they are
        intentionally left untouched — a remembered incident outlives the dataset row and is
        dropped via `forget_incident`/`resolve`, not here.

        Returns True if a baseline was actually removed, False if this URN wasn't being
        watched (so a CLI can report a miss rather than claim an action).
        """
        existed = self.baselines.pop(urn, None) is not None
        self.muted_urns.discard(urn)
        self.muted_until.pop(urn, None)
        self.mute_reasons.pop(urn, None)
        self.muted_at.pop(urn, None)
        return existed

    def __len__(self) -> int:
        return len(self.baselines)

    def __contains__(self, urn: object) -> bool:
        return urn in self.baselines

    # ---- incident dedup ------------------------------------------------------------
    def has_seen(self, fingerprint: str) -> bool:
        """True if this exact incident (fingerprint) was recorded on an earlier run."""
        return fingerprint in self.seen_incidents

    def record_incident(
        self,
        fingerprint: str,
        *,
        severity: Optional[str] = None,
        title: Optional[str] = None,
        datasets: int = 0,
        serving: bool = False,
        kinds: Optional[Iterable[str]] = None,
        owners: Optional[Iterable[str]] = None,
        urns: Optional[Iterable[str]] = None,
        now: Optional[float] = None,
    ) -> int:
        """Record one observation of an incident; return its running observation count.

        First sighting returns 1. Callers should check `has_seen()` *before* recording to
        decide whether an alert is new vs a repeat.

        The optional provenance (severity/title/datasets/serving/kinds/owners) is stored for
        later display + filtering by `ogle incidents`. It's refreshed to the current sighting
        only when the caller supplies it — a bare `record_incident(fp)` never blanks provenance
        an earlier rich call captured, so a metadata-less dedup ping can't erase the record's
        human context. `kinds` (the drift dimensions present this sighting) refreshes with the
        rest of the provenance block, since a recurring incident's dimension set can shift.

        `owners` (the "who to page" display names, unioned across the incident's datasets) is
        provenance in exactly the same sense: it reflects the latest sighting and is stored for
        display, never for the dedup fingerprint (re-assigning an owner is not drift — that
        invariant lives in `narrative.incident_fingerprint`, which never sees owners). Like
        `kinds` it refreshes only when supplied; `owners=None` leaves an earlier capture intact,
        so an owner-less offline (`--signatures`) run can't erase the attribution a live walk
        recorded.

        `urns` (the dataset URNs this incident touched) is provenance in the same latest-sighting
        sense — stored so a per-asset history query (`prior_incident_history`) can tell whether an
        asset is chronically unstable. It refreshes only when supplied; `urns=None` leaves an
        earlier capture intact. A legacy/pre-`urns` record simply carries [] and is invisible to
        the per-asset query (the same never-guess rule `kinds`/`owners` filters follow).

        `now` (epoch seconds) stamps `last_seen` for this sighting, giving the incident a
        temporal axis (`ogle incidents` age display + `--stale` staleness hunt). It's
        always refreshed when supplied — last_seen means *most recent* sighting — but a
        `now=None` call never clears a timestamp an earlier call set, so an untimed dedup
        ping can't erase age history (mirrors the provenance-refresh rule above).

        The same `now` stamps `first_seen` *set-if-absent* — the incident's longevity axis
        (`ogle incidents` "first seen X ago"). Unlike last_seen it is written once and never
        moved, so it measures the whole standing life of the drift (first detection → now),
        not the latest sighting. It's dropped only when the incident is forgotten/resolved.
        A first timed sighting on an incident whose earlier sightings were untimed backfills
        first_seen to that clock (best available), never overwriting a stamp already set.
        """
        rec = self.seen_incidents.get(fingerprint)
        if rec is None:
            rec = _IncidentRecord(count=0)
            self.seen_incidents[fingerprint] = rec
        rec.count += 1
        if severity is not None or title is not None:
            rec.severity = severity
            rec.title = title
            rec.datasets = datasets
            rec.serving = serving
            # Refresh the drift-dimension set with the rest of the provenance (same
            # latest-sighting semantics). Deduped + sorted for a stable, diffable record;
            # None leaves an earlier set intact, [] here means "supplied, but empty".
            if kinds is not None:
                rec.kinds = sorted({str(k) for k in kinds})
            # Same latest-sighting refresh as kinds: deduped + sorted for a stable, diffable
            # record; None leaves an earlier capture intact, [] means "supplied, but unowned".
            if owners is not None:
                rec.owners = sorted({str(o) for o in owners})
            # Same latest-sighting refresh as kinds/owners: deduped + sorted for a stable,
            # diffable record; None leaves an earlier capture intact, [] means "supplied,
            # but no URNs recorded".
            if urns is not None:
                rec.urns = sorted({str(u) for u in urns})
        if now is not None:
            rec.last_seen = now
            if rec.first_seen is None:
                rec.first_seen = now
        return rec.count

    def forget_incident(self, fingerprint: str) -> None:
        """Drop an incident from memory (e.g. once the underlying drift is resolved)."""
        self.seen_incidents.pop(fingerprint, None)

    def incidents_confined_to(self, urns: Iterable[str]) -> List[str]:
        """Fingerprints of remembered incidents whose every touched URN is in `urns`.

        The write-side companion to `forget_baseline` for a decommissioned dataset: once a
        table is gone from DataHub it stops being walked, so an open incident that touches
        ONLY that table can never self-resolve (its fingerprint won't recur) and would sit in
        `ogle incidents` forever. This finds exactly those orphans so a caller can prune them.

        An incident is returned only when its recorded `urns` provenance is non-empty AND is a
        subset of `urns` — i.e. the whole incident is explained by the given set. An incident
        that also touches a URN outside the set is preserved (it still has a live asset behind
        it). A legacy/offline record with no captured URNs is never returned: it can't be
        *proven* to be confined, so the same never-guess rule the `--owner`/`--kind` filters
        follow keeps it out. Returns the matching fingerprints (sorted for stable output).
        """
        want = {str(u) for u in urns}
        if not want:
            return []
        return sorted(
            fp
            for fp, rec in self.seen_incidents.items()
            if rec.urns and want.issuperset(rec.urns)
        )

    def incidents(self) -> List[dict]:
        """Every remembered incident as a plain dict (its provenance + `fingerprint`).

        The read-only view behind `ogle incidents` — Ogle's cross-run drift memory made
        inspectable. Ordering and severity ranking are the caller's job so the store stays
        free of the scorer's `Severity` enum (it only ever knows severity as a string).
        """
        out: List[dict] = []
        for fp, rec in self.seen_incidents.items():
            d = rec.to_dict()
            d["fingerprint"] = fp
            out.append(d)
        return out

    def prior_incident_history(
        self, urns: Iterable[str], exclude_fingerprint: Optional[str] = None
    ) -> int:
        """How many DISTINCT remembered incidents touched any of the given dataset URNs.

        The cross-run signal behind an alert's "recurring instability" annotation: an asset
        with prior incidents on record is chronically unstable, not a first-time blip. Pass
        `exclude_fingerprint` (usually the incident being reported this run) so an incident
        never counts itself.

        Only incidents whose recorded `urns` provenance overlaps `urns` count — a legacy or
        offline record that never captured its URNs can't be *proven* to touch the asset, so
        it is not counted (the same never-guess rule the `--owner`/`--kind` filters follow).
        Returns the match count (0 when none / all un-URN'd).
        """
        want = {str(u) for u in urns}
        if not want:
            return 0
        return sum(
            1
            for fp, rec in self.seen_incidents.items()
            if fp != exclude_fingerprint and want.intersection(rec.urns)
        )

    # ---- muting (known false positives) --------------------------------------------
    def mute(
        self,
        urn: str,
        until: Optional[float] = None,
        reason: Optional[str] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Mark a dataset as a known false positive so its drift never pages.

        `until` is an epoch-seconds expiry for a *snooze* (temporary mute); omit it for a
        permanent mute. A permanent mute supersedes any existing snooze for the same URN,
        and re-snoozing an already-permanent URN is a no-op (the stronger state stands).

        `reason` is an optional human note explaining the mute. It's recorded whenever
        supplied — even for an already-muted URN, so `ogle mute foo --reason ...` can
        annotate a silence after the fact — and a `reason=None` call never blanks a note an
        earlier mute set (mirrors the provenance-refresh rule on `record_incident`). A
        re-snooze/no-op that leaves the mute state unchanged can still update the note.

        `now` (epoch seconds) dates the silence: it stamps `muted_at[urn]` the first time a
        URN becomes muted and is *preserved* while that silence stays continuous — a
        re-annotate, a re-snooze no-op, or a snooze→permanent escalation all keep the original
        start, so `ogle muted`'s "muted 3d ago" measures the whole standing blind spot, not
        the last touch. A `now=None` call never stamps or clears (keeps the method usable from
        tests/legacy callers without a clock); the stamp is cleared only when the mute ends.

        Returns True if this changed the mute *state*, False if it was already covered by an
        equal-or-stronger mute (so a CLI can say "already muted" rather than claim an action).
        Setting a reason / stamping the age are side effects and never affect this return value.
        """
        if reason is not None:
            # Only annotate a URN that is (or is becoming) muted, so no orphan note lingers
            # for an unmuted dataset. The stronger-state guards below can still return False,
            # but the URL ends this call muted either way, so recording the note is correct.
            self.mute_reasons[urn] = reason
        if now is not None and urn not in self.muted_at:
            # Set-if-absent: the stamp marks when this continuous silence began, so it survives
            # re-annotation and snooze→permanent escalation. It's absent again only after the
            # mute is fully cleared (unmute/forget/expiry), so the next mute re-dates from now.
            self.muted_at[urn] = now
        if until is None:
            # Permanent mute wins over any snooze.
            self.muted_until.pop(urn, None)
            if urn in self.muted_urns:
                return False
            self.muted_urns.add(urn)
            return True
        # Timed snooze. A permanent mute is stronger — leave it alone.
        if urn in self.muted_urns:
            return False
        self.muted_until[urn] = until
        return True

    def unmute(self, urn: str) -> bool:
        """Stop suppressing a dataset's drift (clears both permanent and timed mutes).

        Returns True if it had been muted in either form.
        """
        had = urn in self.muted_urns or urn in self.muted_until
        self.muted_urns.discard(urn)
        self.muted_until.pop(urn, None)
        self.mute_reasons.pop(urn, None)
        self.muted_at.pop(urn, None)
        return had

    def is_muted(self, urn: str, now: Optional[float] = None) -> bool:
        """True if drift on this dataset should be tracked but not alerted on.

        Permanent mutes always count. A snooze counts only until it expires: pass `now`
        (epoch seconds) to enforce expiry — the paging path does this so a lapsed snooze
        pages again automatically. With `now` omitted, a snooze reads as "configured muted"
        (useful for `is this in the mute list at all`).
        """
        if urn in self.muted_urns:
            return True
        exp = self.muted_until.get(urn)
        if exp is None:
            return False
        return True if now is None else exp > now

    def mute_expiry(self, urn: str) -> Optional[float]:
        """The snooze expiry for `urn` (epoch seconds), or None if permanent / not muted."""
        return self.muted_until.get(urn)

    def mute_reason(self, urn: str) -> Optional[str]:
        """The human rationale recorded for `urn`'s mute, or None if none was given."""
        return self.mute_reasons.get(urn)

    def mute_since(self, urn: str) -> Optional[float]:
        """When `urn`'s current silence began (epoch seconds), or None if not dated.

        Only mutes created with a `now` clock carry a stamp — a legacy/hand-edited file or a
        mute set without one reads as None (age *unknown*, not zero), so a caller shows "muted
        <unknown> ago" honestly rather than inventing an age.
        """
        return self.muted_at.get(urn)

    def purge_expired_mutes(self, now: float) -> List[str]:
        """Drop snoozes that have expired as of `now`; return the URNs freed (sorted).

        Keeps the on-disk store from accumulating dead snoozes. Permanent mutes are untouched.
        """
        expired = sorted(urn for urn, exp in self.muted_until.items() if exp <= now)
        for urn in expired:
            self.muted_until.pop(urn, None)
            self.mute_reasons.pop(urn, None)  # drop the note with the snooze it annotated
            self.muted_at.pop(urn, None)      # and its age stamp — the silence has ended
        return expired

    def muted(self, now: Optional[float] = None) -> List[str]:
        """All currently muted dataset URNs (sorted). With `now`, expired snoozes are excluded."""
        active = set(self.muted_urns)
        for urn, exp in self.muted_until.items():
            if now is None or exp > now:
                active.add(urn)
        return sorted(active)

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": STORE_VERSION,
            "baselines": {urn: sig.to_dict() for urn, sig in self.baselines.items()},
            "seen_incidents": {fp: r.to_dict() for fp, r in self.seen_incidents.items()},
            "muted_urns": sorted(self.muted_urns),
            "muted_until": {urn: self.muted_until[urn] for urn in sorted(self.muted_until)},
            # Persist a note only for a URN that is actually muted (permanent or snoozed) so
            # a stray reason can never linger past its mute or bloat the file.
            "mute_reasons": {
                urn: self.mute_reasons[urn]
                for urn in sorted(self.mute_reasons)
                if urn in self.muted_urns or urn in self.muted_until
            },
            # Age stamps, persisted only for a URN still muted in either form — same guard as
            # mute_reasons so a stamp can't linger past its mute or bloat the file.
            "muted_at": {
                urn: self.muted_at[urn]
                for urn in sorted(self.muted_at)
                if urn in self.muted_urns or urn in self.muted_until
            },
        }

    @classmethod
    def from_dict(cls, data: dict, path: Optional[Path] = None) -> "BaselineStore":
        # A store file must be a JSON *object*. A valid-JSON-but-non-object payload (a
        # top-level array, number, string, or null — e.g. a truncated/partially-overwritten
        # file that still parses) would otherwise reach `data.get(...)` and raise
        # AttributeError, which `load`'s corrupt-recovery net does NOT catch — so a scheduled
        # `ogle check` would crash-loop and go blind to drift on exactly the class of bad file
        # the net exists for. Normalize it to the same ValueError the version/shape guards
        # raise, so it quarantines (default) or surfaces cleanly (strict) like any other
        # foreign file.
        if not isinstance(data, dict):
            raise ValueError(
                f"baseline store must be a JSON object, got {type(data).__name__} "
                f"(refusing to misread a truncated/foreign file)"
            )
        version = data.get("version")
        if version != STORE_VERSION:
            raise ValueError(
                f"baseline store version {version!r} != supported {STORE_VERSION} "
                f"(refusing to misread a stale/foreign file)"
            )
        # The nested `baselines`/`seen_incidents` sections must each be a JSON *object* too.
        # A valid-JSON-but-non-object value here (a list/scalar from a truncated or
        # hand-mangled file that still parses AND carries the right `version`) would reach
        # `.items()` and raise AttributeError — which is NOT in `load`'s corrupt-recovery net
        # (ValueError/KeyError/TypeError), so a scheduled `ogle check` would crash-loop and go
        # blind on exactly the bad file the net exists for. Normalize to the same ValueError
        # the top-level and version guards raise so it quarantines/surfaces like any foreign file.
        raw_baselines = data.get("baselines", {})
        if not isinstance(raw_baselines, dict):
            raise ValueError(
                f"baseline store 'baselines' must be a JSON object, got "
                f"{type(raw_baselines).__name__} (refusing to misread a truncated/foreign file)"
            )
        raw_seen = data.get("seen_incidents", {})
        if not isinstance(raw_seen, dict):
            raise ValueError(
                f"baseline store 'seen_incidents' must be a JSON object, got "
                f"{type(raw_seen).__name__} (refusing to misread a truncated/foreign file)"
            )
        baselines = {
            urn: DatasetSignature.from_dict(raw)
            for urn, raw in raw_baselines.items()
        }
        seen = {
            fp: _IncidentRecord.from_dict(raw)
            for fp, raw in raw_seen.items()
        }
        # muted_urns is additive (introduced after v1 shipped): a store written by an older
        # Ogle simply lacks the key and loads with nothing muted, so no version bump is needed.
        # It must be a JSON *list* when present, though. Unlike the sibling mute sections below
        # (which go through `dict(...)` — a string/scalar there raises ValueError/TypeError that
        # `load` catches to quarantine), `set(...)` degrades SILENTLY on a non-list: a hand-
        # edited/truncated `"muted_urns": "urn:li:dataset:x"` char-splits into a set of single
        # characters ({'u','r','n',':',...}) with no exception, so the operator's intended mute
        # vanishes and the dataset it was meant to silence starts paging again — quietly wrong,
        # the worst failure for a suppression list. Normalize a present-but-non-list value to the
        # same ValueError the other shape guards raise so it quarantines/surfaces like any foreign
        # file (mirrors the `serving_urns` guard in cli.load_signatures_file).
        raw_muted = data.get("muted_urns", [])
        if not isinstance(raw_muted, list):
            raise ValueError(
                f"baseline store 'muted_urns' must be a JSON list, got "
                f"{type(raw_muted).__name__} (refusing to misread a truncated/foreign file)"
            )
        # Every muted_urns ELEMENT must be a string URN. The list-guard above only proves the
        # *container* is a list; a hand-edited/truncated file can still carry a non-string element
        # (`"muted_urns": ["urn:li:dataset:x", 123, true]`). Unlike the sibling incident-record
        # lists (`kinds`/`owners`/`urns`), which `str()`-coerce each element on load, this set is
        # built raw — and a non-string entry can NEVER equal a real string URN in `is_muted`'s set
        # lookup (`"urn:..." in {123}` is False), so the intended mute silently VANISHES and the
        # dataset it was meant to silence starts paging again. That is the same silent-suppression-
        # loss the container guard was written to stop, one level deeper (element vs container).
        # Coercing to str wouldn't help — a bogus `"123"` still can't match a real URN — so reject
        # with the same ValueError contract and let `load()` quarantine the foreign file.
        for entry in raw_muted:
            if not isinstance(entry, str):
                raise ValueError(
                    f"baseline store 'muted_urns' entries must be strings, got "
                    f"{type(entry).__name__} (refusing to misread a truncated/foreign file)"
                )
        muted = set(raw_muted)

        # The three mute-metadata fields below are all JSON objects (urn -> value maps). A
        # non-dict value (truncated/foreign/hand-edited file) must be *rejected*, not coerced:
        # `dict(["12","34"])` silently char-pairs a list of 2-char strings into `{"1":"2",...}`
        # — the exact silent-misread the `muted_urns` guard above was written to prevent. Raise
        # the same clean ValueError contract so `load()` quarantines it like any foreign file.
        def _require_map(key: str) -> dict:
            raw = data.get(key, {})
            if not isinstance(raw, dict):
                raise ValueError(
                    f"baseline store '{key}' must be a JSON object, got "
                    f"{type(raw).__name__} (refusing to misread a truncated/foreign file)"
                )
            return raw

        # muted_until (timed snoozes) is likewise additive; older files lack it. Guard against
        # a URN being both permanent and snoozed (permanent wins) so state stays coherent.
        muted_until = {
            urn: _finite_mute_time(exp, key="muted_until", urn=urn)
            for urn, exp in _require_map("muted_until").items()
            if urn not in muted
        }
        # mute_reasons (introduced after muted_until) is likewise additive; older files lack
        # it. Keep only notes whose URN is still muted in either form, so a hand-edited or
        # legacy file can't resurrect an orphan reason for an unmuted dataset.
        mute_reasons = {
            urn: str(reason)
            for urn, reason in _require_map("mute_reasons").items()
            if urn in muted or urn in muted_until
        }
        # muted_at (age stamps, introduced after mute_reasons) is likewise additive; older
        # files lack it and their mutes simply read as undated. Keep only stamps whose URN is
        # still muted, so a legacy/hand-edited file can't resurrect an orphan stamp.
        muted_at = {
            urn: _finite_mute_time(ts, key="muted_at", urn=urn)
            for urn, ts in _require_map("muted_at").items()
            if urn in muted or urn in muted_until
        }
        return cls(
            path=path,
            baselines=baselines,
            seen_incidents=seen,
            muted_urns=muted,
            muted_until=muted_until,
            mute_reasons=mute_reasons,
            muted_at=muted_at,
        )

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Atomically write the store to disk. Returns the path written.

        Writes to a temp file in the same directory then `os.replace`s it into place, so a
        concurrent reader (or a crash) never sees a partial file.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to (construct with path= or pass one)")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        blob = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".ogle-store-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
                # Force the temp file's bytes to disk BEFORE the rename. os.replace makes the
                # swap atomic, but atomicity of the rename is not durability of the data: a
                # crash/power-loss right after the rename becomes durable can otherwise surface
                # a truncated or zero-length store under the real name. Flushing + fsync here is
                # what makes the module-header "a crash mid-write can't corrupt the baseline"
                # guarantee true at the data level, not just the rename level.
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
            # The rename is atomic, but its durability lives in the parent directory entry:
            # fsync the directory so a crash right after the swap can't lose the rename and
            # revert the store to its pre-save name. Best-effort (no-op where unsupported).
            _fsync_dir(target.parent)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.path = target
        return target

    @staticmethod
    def _quarantine_corrupt(p: Path) -> Path:
        """Move an unreadable store file aside so the next `save()` can write a clean one,
        while preserving the bad file for forensics. Returns the backup path.

        Deterministic naming (`<name>.corrupt`, then `.corrupt.1`, `.corrupt.2`, ... if a
        prior recovery already claimed the slot) — never clobbers an earlier forensic copy,
        and stays test-reproducible (no timestamp/random in the name). The move is an atomic
        same-directory `os.replace`.
        """
        target = p.with_name(p.name + ".corrupt")
        n = 1
        while target.exists():
            target = p.with_name(p.name + f".corrupt.{n}")
            n += 1
        os.replace(p, target)
        return target

    @classmethod
    def load(cls, path: Union[str, Path], *, recover_corrupt: bool = True) -> "BaselineStore":
        """Load a store from disk. A missing file yields a fresh empty store (first run).

        A corrupt or foreign file (invalid JSON, wrong version, or malformed shape) is a
        real production hazard: a scheduled `ogle check` that crash-loops on a bad store
        goes silently blind to drift. So by default (`recover_corrupt=True`) an unreadable
        file is *quarantined* aside (see `_quarantine_corrupt`) and this returns a fresh
        empty store that re-baselines on the next walk — `recovered_from_corruption` and
        `corrupt_backup_path` are set on it so the caller can warn instead of failing silent.
        Pass `recover_corrupt=False` for a strict caller (or a test) that wants the raw error.
        """
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls.from_dict(data, path=p)
        except (ValueError, KeyError, TypeError):
            # ValueError covers json.JSONDecodeError (a subclass) + from_dict's version
            # guard + bad numeric coercions; KeyError/TypeError cover a shape that parsed
            # as JSON but isn't a store. All mean "this file is not a usable baseline store".
            if not recover_corrupt:
                raise
            backup = cls._quarantine_corrupt(p)
            store = cls(path=p)
            store.recovered_from_corruption = True
            store.corrupt_backup_path = backup
            return store

    def put_many(self, signatures: Iterable[DatasetSignature]) -> None:
        """Convenience: upsert a batch of baselines (what a full DataHub walk produces)."""
        for sig in signatures:
            self.put_baseline(sig)
