"""Drift-check pipeline — the end-to-end seam a DataHub walk plugs into.

This is where the four W2 cores meet:

    signature (fingerprint)  ->  store (baseline diff memory)
                                     |
                                     v
                          scorer (per-dataset drift findings)
                                     |
                                     v
                        narrative (findings -> incident + text)

`run_drift_check` is deliberately I/O-free: it takes the *current* signatures the walker
already pulled and a `BaselineStore`, and returns a `DriftReport`. The live DataHub client
(W2b remainder, needs the Docker quickstart) only has to produce `DatasetSignature`s and
hand them here — every decision below is pure and unit-testable without a quickstart or a
real model.

New datasets (no baseline yet) are never scored — you can't diff against nothing — they are
just seeded as baselines so the *next* run can. Incident dedup runs against the store so a
scheduled loop reports a drift once, not every tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

from .narrative import Incident, build_incident, narrate
from .scorer import DriftFinding, ScoreConfig, score_dataset
from .signature import DatasetSignature
from .store import BaselineStore


@dataclass(frozen=True)
class DriftReport:
    """The outcome of one drift-check run — everything a caller needs to alert or stay quiet."""

    findings: List[DriftFinding]
    incident: Optional[Incident]
    narrative: str
    # URNs seen this run, split by how they were handled.
    scored_urns: List[str] = field(default_factory=list)      # had a baseline -> diffed
    new_urns: List[str] = field(default_factory=list)         # first sighting -> seeded only
    suppressed_urns: List[str] = field(default_factory=list)  # drifted but muted -> not paged
    # True when there IS an incident AND it wasn't reported on an earlier run.
    is_new_incident: bool = False
    # Running observation count of this incident (1 = first time). 0 when no incident.
    incident_count: int = 0

    @property
    def should_alert(self) -> bool:
        """Alert only on a genuinely new incident — the debounce a scheduled loop relies on."""
        return self.incident is not None and self.is_new_incident

    def to_dict(self) -> dict:
        return {
            "narrative": self.narrative,
            "incident": self.incident.to_dict() if self.incident else None,
            "scored_urns": list(self.scored_urns),
            "new_urns": list(self.new_urns),
            "suppressed_urns": list(self.suppressed_urns),
            "is_new_incident": self.is_new_incident,
            "incident_count": self.incident_count,
            "should_alert": self.should_alert,
        }


def run_drift_check(
    store: BaselineStore,
    current: Sequence[DatasetSignature],
    serving_urns: Iterable[str] = (),
    cfg: Optional[ScoreConfig] = None,
    llm: Optional[Callable[[str], str]] = None,
    update_baselines: bool = True,
    now: Optional[float] = None,
) -> DriftReport:
    """Score a batch of fresh signatures against stored baselines and narrate the result.

    Args:
        store: baselines + incident dedup memory (see `BaselineStore`).
        current: freshly-pulled signatures for the datasets being monitored.
        serving_urns: URNs that feed a deployed model — their findings are severity-escalated.
        cfg: scoring thresholds (defaults are quiet-on-noise, loud-on-breakage).
        llm: optional callable to phrase the incident; falls back to deterministic markdown.
        update_baselines: when True (default) the current signatures become the new baselines
            so the next run diffs against the latest state. Pass False for a read-only probe.
        now: epoch seconds used to expire snoozed mutes (defaults to the wall clock). Inject a
            fixed value in tests for deterministic snooze-expiry behaviour.

    Returns:
        A `DriftReport`. `should_alert` is the single field a scheduled loop needs.
    """
    serving = set(serving_urns)
    cfg = cfg or ScoreConfig()
    now = time.time() if now is None else now

    all_findings: List[DriftFinding] = []
    scored_urns: List[str] = []
    new_urns: List[str] = []
    suppressed_urns: List[str] = []

    for sig in current:
        baseline = store.get_baseline(sig.urn)
        if baseline is None:
            # First time Ogle has seen this dataset — nothing to diff, just seed it.
            new_urns.append(sig.urn)
            continue
        findings = score_dataset(baseline, sig, cfg, serving=sig.urn in serving)
        scored_urns.append(sig.urn)
        # A muted dataset is still diffed (so its baseline advances and it can be unmuted
        # later against fresh state), but its findings are held out of the incident so a
        # known-noisy asset never pages — even when it flaps with a brand-new fingerprint.
        if findings and store.is_muted(sig.urn, now):
            suppressed_urns.append(sig.urn)
            continue
        all_findings.extend(findings)

    # Rank the merged findings so the narrative leads with the worst across all datasets.
    all_findings.sort(key=lambda f: f.severity.rank, reverse=True)

    incident = build_incident(all_findings)
    is_new = False
    count = 0
    if incident is not None:
        is_new = not store.has_seen(incident.fingerprint)
        count = store.record_incident(incident.fingerprint)

    text = narrate(all_findings, llm=llm)

    # Advance baselines only after scoring, so a mid-batch failure can't half-update state.
    if update_baselines:
        store.put_many(current)

    return DriftReport(
        findings=all_findings,
        incident=incident,
        narrative=text,
        scored_urns=sorted(scored_urns),
        new_urns=sorted(new_urns),
        suppressed_urns=sorted(suppressed_urns),
        is_new_incident=is_new,
        incident_count=count,
    )
