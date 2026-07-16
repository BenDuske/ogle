"""Baseline store — Ogle's memory between runs.

Drift detection is a *diff*: it needs to remember what each dataset looked like last time so
it can tell what changed. This module is that memory. It persists two things:

  * BASELINES     — the last `DatasetSignature` seen per dataset URN. Next run scores the
                    fresh signature against this to surface schema/volume/quality drift.
  * SEEN INCIDENTS — the set of incident fingerprints Ogle has already reported, with an
                    observation count. Lets a scheduled run tell a *new* problem from one it
                    already alerted on, so Ben isn't paged every 10 minutes for the same drift.

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
from typing import Dict, Iterable, List, Optional, Union

from .signature import DatasetSignature

# Bump when the on-disk shape changes so a stale file can be detected rather than misread.
STORE_VERSION = 1


@dataclass
class _IncidentRecord:
    """What Ogle remembers about one incident fingerprint across runs."""

    count: int = 0

    def to_dict(self) -> dict:
        return {"count": self.count}

    @classmethod
    def from_dict(cls, data: dict) -> "_IncidentRecord":
        return cls(count=int(data.get("count", 0)))


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

    def __len__(self) -> int:
        return len(self.baselines)

    def __contains__(self, urn: object) -> bool:
        return urn in self.baselines

    # ---- incident dedup ------------------------------------------------------------
    def has_seen(self, fingerprint: str) -> bool:
        """True if this exact incident (fingerprint) was recorded on an earlier run."""
        return fingerprint in self.seen_incidents

    def record_incident(self, fingerprint: str) -> int:
        """Record one observation of an incident; return its running observation count.

        First sighting returns 1. Callers should check `has_seen()` *before* recording to
        decide whether an alert is new vs a repeat.
        """
        rec = self.seen_incidents.get(fingerprint)
        if rec is None:
            rec = _IncidentRecord(count=0)
            self.seen_incidents[fingerprint] = rec
        rec.count += 1
        return rec.count

    def forget_incident(self, fingerprint: str) -> None:
        """Drop an incident from memory (e.g. once the underlying drift is resolved)."""
        self.seen_incidents.pop(fingerprint, None)

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": STORE_VERSION,
            "baselines": {urn: sig.to_dict() for urn, sig in self.baselines.items()},
            "seen_incidents": {fp: r.to_dict() for fp, r in self.seen_incidents.items()},
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
        return cls(path=path, baselines=baselines, seen_incidents=seen)

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

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineStore":
        """Load a store from disk. A missing file yields a fresh empty store (first run)."""
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data, path=p)

    def put_many(self, signatures: Iterable[DatasetSignature]) -> None:
        """Convenience: upsert a batch of baselines (what a full DataHub walk produces)."""
        for sig in signatures:
            self.put_baseline(sig)
