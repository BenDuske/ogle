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
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

from .signature import DatasetSignature

# Bump when the on-disk shape changes so a stale file can be detected rather than misread.
STORE_VERSION = 1


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
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "_IncidentRecord":
        ls = data.get("last_seen")
        return cls(
            count=int(data.get("count", 0)),
            severity=data.get("severity"),
            title=data.get("title"),
            datasets=int(data.get("datasets", 0)),
            serving=bool(data.get("serving", False)),
            last_seen=float(ls) if ls is not None else None,
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
        now: Optional[float] = None,
    ) -> int:
        """Record one observation of an incident; return its running observation count.

        First sighting returns 1. Callers should check `has_seen()` *before* recording to
        decide whether an alert is new vs a repeat.

        The optional provenance (severity/title/datasets/serving) is stored for later
        display by `ogle incidents`. It's refreshed to the current sighting only when the
        caller supplies it — a bare `record_incident(fp)` never blanks provenance an
        earlier rich call captured, so a metadata-less dedup ping can't erase the record's
        human context.

        `now` (epoch seconds) stamps `last_seen` for this sighting, giving the incident a
        temporal axis (`ogle incidents` age display + `--stale` staleness hunt). It's
        always refreshed when supplied — last_seen means *most recent* sighting — but a
        `now=None` call never clears a timestamp an earlier call set, so an untimed dedup
        ping can't erase age history (mirrors the provenance-refresh rule above).
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
        if now is not None:
            rec.last_seen = now
        return rec.count

    def forget_incident(self, fingerprint: str) -> None:
        """Drop an incident from memory (e.g. once the underlying drift is resolved)."""
        self.seen_incidents.pop(fingerprint, None)

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

    # ---- muting (known false positives) --------------------------------------------
    def mute(self, urn: str, until: Optional[float] = None, reason: Optional[str] = None) -> bool:
        """Mark a dataset as a known false positive so its drift never pages.

        `until` is an epoch-seconds expiry for a *snooze* (temporary mute); omit it for a
        permanent mute. A permanent mute supersedes any existing snooze for the same URN,
        and re-snoozing an already-permanent URN is a no-op (the stronger state stands).

        `reason` is an optional human note explaining the mute. It's recorded whenever
        supplied — even for an already-muted URN, so `ogle mute foo --reason ...` can
        annotate a silence after the fact — and a `reason=None` call never blanks a note an
        earlier mute set (mirrors the provenance-refresh rule on `record_incident`). A
        re-snooze/no-op that leaves the mute state unchanged can still update the note.

        Returns True if this changed the mute *state*, False if it was already covered by an
        equal-or-stronger mute (so a CLI can say "already muted" rather than claim an action).
        Setting a reason is a side effect and never affects this return value.
        """
        if reason is not None:
            # Only annotate a URN that is (or is becoming) muted, so no orphan note lingers
            # for an unmuted dataset. The stronger-state guards below can still return False,
            # but the URL ends this call muted either way, so recording the note is correct.
            self.mute_reasons[urn] = reason
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

    def purge_expired_mutes(self, now: float) -> List[str]:
        """Drop snoozes that have expired as of `now`; return the URNs freed (sorted).

        Keeps the on-disk store from accumulating dead snoozes. Permanent mutes are untouched.
        """
        expired = sorted(urn for urn, exp in self.muted_until.items() if exp <= now)
        for urn in expired:
            self.muted_until.pop(urn, None)
            self.mute_reasons.pop(urn, None)  # drop the note with the snooze it annotated
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
        }

    @classmethod
    def from_dict(cls, data: dict, path: Optional[Path] = None) -> "BaselineStore":
        version = data.get("version")
        if version != STORE_VERSION:
            raise ValueError(
                f"baseline store version {version!r} != supported {STORE_VERSION} "
                f"(refusing to misread a stale/foreign file)"
            )
        baselines = {
            urn: DatasetSignature.from_dict(raw)
            for urn, raw in data.get("baselines", {}).items()
        }
        seen = {
            fp: _IncidentRecord.from_dict(raw)
            for fp, raw in data.get("seen_incidents", {}).items()
        }
        # muted_urns is additive (introduced after v1 shipped): a store written by an older
        # Ogle simply lacks the key and loads with nothing muted, so no version bump is needed.
        muted = set(data.get("muted_urns", []))
        # muted_until (timed snoozes) is likewise additive; older files lack it. Guard against
        # a URN being both permanent and snoozed (permanent wins) so state stays coherent.
        muted_until = {
            urn: float(exp)
            for urn, exp in dict(data.get("muted_until", {})).items()
            if urn not in muted
        }
        # mute_reasons (introduced after muted_until) is likewise additive; older files lack
        # it. Keep only notes whose URN is still muted in either form, so a hand-edited or
        # legacy file can't resurrect an orphan reason for an unmuted dataset.
        mute_reasons = {
            urn: str(reason)
            for urn, reason in dict(data.get("mute_reasons", {})).items()
            if urn in muted or urn in muted_until
        }
        return cls(
            path=path,
            baselines=baselines,
            seen_incidents=seen,
            muted_urns=muted,
            muted_until=muted_until,
            mute_reasons=mute_reasons,
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
            os.replace(tmp, target)
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
