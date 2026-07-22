"""Narrative writer — turns scorer findings into a human-readable incident.

`score_dataset` produces a flat list of `DriftFinding`s; a human (or a Slack/WhatsApp
alert) wants one coherent story: *what broke, how bad, which datasets, what to check.*
This module is that story.

Two layers, cleanly separated so the useful part never depends on an LLM being reachable:

  * **Deterministic core** — `build_incident` folds findings into an `Incident`
    (overall severity, per-dataset sections, ranked recommended actions, a stable
    `fingerprint` for Aegis dedup). `render_markdown` prints it. Pure: same findings ->
    same bytes, so it is unit-testable with no live DataHub and no model.
  * **LLM polish (optional)** — `narrate(findings, llm=...)` builds a grounded prompt
    from the deterministic facts and lets a model phrase it. If no `llm` is given, or the
    call raises, it returns the deterministic markdown. The model only ever *rewords*
    facts Ogle already computed — it is never the source of truth for severity.

The `fingerprint` is what lets Aegis's salience memory dedup a recurring incident across
scheduled runs (same datasets + same drift kinds/severities = the same open issue, not a
new one) and learn which incidents Ben acts on.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .scorer import DriftFinding, DriftKind, Severity

# What a human should actually do about each drift kind. Kept terse and imperative — this
# is the "next click" line in an alert, not a runbook.
_ACTIONS: Dict[DriftKind, str] = {
    DriftKind.SCHEMA: (
        "check the upstream transform that renamed/dropped the column, and any feature "
        "reading it before the next training run"
    ),
    DriftKind.VOLUME: (
        "check whether the upstream load job stopped, backfilled, or double-wrote before "
        "trusting downstream features"
    ),
    DriftKind.QUALITY: (
        "check the source for a partial/failed load — a null spike usually means an "
        "upstream join or extract broke, not real data"
    ),
    DriftKind.DISTRIBUTION: (
        "check the upstream transform for a stuck default or a fan-out join — a collapsed "
        "distinct-value count means the feature lost signal or rows got duplicated"
    ),
    DriftKind.FRESHNESS: (
        "check whether the upstream load/profile job is still running — a stale timestamp "
        "with unchanged rows usually means the feed silently stopped, so every retrain is "
        "learning yesterday's data"
    ),
    DriftKind.MEAN: (
        "check the source for a unit/scale change, sensor recalibration, or a genuine "
        "population shift — a moved mean with intact schema and row count is covariate "
        "drift that quietly degrades the model until it's retrained on the new values"
    ),
    DriftKind.STDEV: (
        "check the source for a stuck sensor, a clipped/saturated range, or a noisier feed — "
        "a collapsed or exploded spread with an intact mean and schema is scale drift that "
        "moves feature variance under the model without touching its average"
    ),
    DriftKind.RANGE: (
        "check the source for an overflow, a unit bug on a subset of rows, or a new outlier "
        "regime — values escaping the historical min/max envelope while the mean and spread "
        "hold are out-of-bounds features that can silently break a model's input assumptions"
    ),
}

_SEV_MARK: Dict[Severity, str] = {
    Severity.HIGH: "\U0001f534",   # red circle
    Severity.MEDIUM: "\U0001f7e0", # orange circle
    Severity.LOW: "\U0001f7e1",    # yellow circle
}


def _normalize_owners(
    owners: Optional[Dict[str, List[str]]], keep_urns: List[str]
) -> Dict[str, List[str]]:
    """Clean an owner map down to what a report can trust.

    DataHub ownership arrives as free-form strings (a corpuser urn, a group name, an
    email). We keep them as given but: restrict to URNs actually in this incident (a
    stray owner for an unrelated dataset never leaks into the alert), strip whitespace,
    drop empties, and dedup while preserving order (an asset owned by the same person via
    two ownership types shows once). A URN with no usable owner is omitted entirely so
    `render_markdown` can cleanly skip the line rather than print an empty "owner:".
    """
    if not owners:
        return {}
    keep = set(keep_urns)
    cleaned: Dict[str, List[str]] = {}
    for urn, names in owners.items():
        if urn not in keep or not names:
            continue
        seen: List[str] = []
        for name in names:
            n = (name or "").strip()
            if n and n not in seen:
                seen.append(n)
        if seen:
            cleaned[urn] = seen
    return cleaned


def short_name(urn: str) -> str:
    """Pull a readable dataset name out of a DataHub dataset URN.

    `urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)` -> `b2fd91.customers`.
    Falls back to the raw URN if it doesn't match the expected shape (never raises — a
    display helper must not be able to break a report).
    """
    inner = urn
    if urn.startswith("urn:li:dataset:(") and urn.endswith(")"):
        inner = urn[len("urn:li:dataset:(") : -1]
    parts = [p for p in inner.split(",")]
    # (platformUrn, name, fabric) — the name is the middle segment when present.
    if len(parts) >= 3:
        return parts[-2].strip()
    return urn


@dataclass(frozen=True)
class Incident:
    """A grouped, ranked view of one scoring run's findings."""

    findings: List[DriftFinding]
    overall_severity: Severity
    serving_impacted: bool
    urns: List[str] = field(default_factory=list)
    fingerprint: str = ""
    # urn -> owner display strings (from DataHub's Ownership aspect). Presentation only:
    # deliberately NOT part of `fingerprint`, because re-assigning an owner is not drift
    # and must never re-page a still-open incident.
    owners: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def title(self) -> str:
        n = len(self.urns)
        noun = "dataset" if n == 1 else "datasets"
        tag = " on a serving path" if self.serving_impacted else ""
        return (
            f"{self.overall_severity.value.upper()} drift across {n} {noun}{tag}"
        )

    @property
    def summary_line(self) -> str:
        """One-line, at-a-glance blast radius for the top of an alert.

        Counts findings by severity (worst first) so an on-call engineer — or a judge
        skimming the demo — sees scope before parsing any per-dataset section. Grounded:
        every number is derived from `findings`, never invented.
        """
        nf = len(self.findings)
        nd = len(self.urns)
        breakdown = ", ".join(
            f"{c} {_SEV_MARK[s]} {s.value}"
            for s in (Severity.HIGH, Severity.MEDIUM, Severity.LOW)
            if (c := sum(1 for f in self.findings if f.severity is s))
        )
        line = (
            f"**{nf} finding{'' if nf == 1 else 's'}** across "
            f"{nd} dataset{'' if nd == 1 else 's'} — {breakdown}"
        )
        if self.serving_impacted:
            line += " · ⚠️ serving path impacted"
        return line

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "overall_severity": self.overall_severity.value,
            "serving_impacted": self.serving_impacted,
            "urns": list(self.urns),
            "fingerprint": self.fingerprint,
            "owners": {u: list(v) for u, v in self.owners.items()},
            "findings": [f.to_dict() for f in self.findings],
        }


def incident_fingerprint(findings: List[DriftFinding]) -> str:
    """Stable id for an incident: the *set* of (urn, kind, severity) triples.

    Order-independent so two runs that surface the same problems in a different order
    dedup to one incident in Aegis memory. Changes when a drift resolves, worsens, or a
    new dataset joins — i.e. exactly when Ben would consider it a different situation.
    """
    triples = sorted(
        (f.urn, f.kind.value, f.severity.value) for f in findings
    )
    blob = "|".join(f"{u}::{k}::{s}" for u, k, s in triples)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_incident(
    findings: List[DriftFinding],
    owners: Optional[Dict[str, List[str]]] = None,
) -> Optional[Incident]:
    """Fold findings into a ranked `Incident`. Returns None when there is nothing to say.

    `owners` is an optional urn -> owner-names map (from DataHub's Ownership aspect). It is
    attached for the narrative's "who to page" line and does not affect severity or the
    dedup fingerprint.
    """
    if not findings:
        return None

    overall = max((f.severity for f in findings), key=lambda s: s.rank)
    serving = any(f.details.get("serving") for f in findings)
    # Preserve first-seen order of URNs (findings arrive most-severe-first per dataset).
    urns: List[str] = []
    for f in findings:
        if f.urn not in urns:
            urns.append(f.urn)

    ordered = sorted(findings, key=lambda f: f.severity.rank, reverse=True)
    return Incident(
        findings=ordered,
        overall_severity=overall,
        serving_impacted=serving,
        urns=urns,
        fingerprint=incident_fingerprint(findings),
        owners=_normalize_owners(owners, urns),
    )


def _recommended_actions(findings: List[DriftFinding]) -> List[str]:
    """Deduped, severity-ordered action lines — one per drift kind actually present."""
    worst: Dict[DriftKind, Severity] = {}
    for f in findings:
        cur = worst.get(f.kind)
        if cur is None or f.severity.rank > cur.rank:
            worst[f.kind] = f.severity
    kinds = sorted(worst, key=lambda k: worst[k].rank, reverse=True)
    return [_ACTIONS[k] for k in kinds]


def render_markdown(incident: Incident) -> str:
    """Deterministic markdown report. This is the ground truth an LLM only rephrases."""
    mark = _SEV_MARK[incident.overall_severity]
    lines: List[str] = [f"## {mark} {incident.title}", "", incident.summary_line, ""]

    by_urn: Dict[str, List[DriftFinding]] = {}
    for f in incident.findings:
        by_urn.setdefault(f.urn, []).append(f)

    for urn in incident.urns:
        lines.append(f"### {short_name(urn)}")
        owners = incident.owners.get(urn)
        if owners:
            label = "owner" if len(owners) == 1 else "owners"
            lines.append(f"- \U0001f464 {label}: {', '.join(owners)}")
        for f in by_urn[urn]:
            lines.append(f"- {_SEV_MARK[f.severity]} **{f.kind.value}** — {f.message}")
        lines.append("")

    actions = _recommended_actions(incident.findings)
    if actions:
        lines.append("**What to check**")
        for a in actions:
            lines.append(f"- {a}")
        lines.append("")

    lines.append(f"_incident {incident.fingerprint}_")
    return "\n".join(lines).rstrip() + "\n"


def build_llm_prompt(incident: Incident) -> str:
    """Grounded prompt handed to an LLM to phrase the incident.

    Pure and testable. It hands the model the already-computed facts and *forbids*
    inventing severity or datasets — the model's only job is a tight, plain-English
    summary a busy engineer can act on. Kept model-agnostic (works for Aegis-local Qwen
    or Anthropic fallback).
    """
    facts = render_markdown(incident)
    return (
        "You are Ogle, an ML-lineage monitoring agent. Below are drift findings Ogle "
        "already computed for datasets feeding production ML models. Write a short "
        "incident summary (3-5 sentences) an on-call engineer can act on.\n"
        "Rules: use ONLY the facts given; do not invent datasets, numbers, severity, or "
        "owners; lead with the most severe, serving-path impact first; if an owner is "
        "listed for an affected dataset, name who to page; end with the single most "
        "important next check. Do not restate the markdown verbatim.\n\n"
        "FACTS:\n"
        f"{facts}"
    )


def narrate(
    findings: List[DriftFinding],
    llm: Optional[Callable[[str], str]] = None,
    owners: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Produce the human-facing narrative for a scoring run.

    With no `llm`, returns the deterministic markdown. With an `llm` callable, hands it the
    grounded prompt and returns its text — but any exception (model down, timeout) falls
    back to the deterministic report, because an alert must always go out. Empty findings
    yield a clean "no drift" line so callers can send a heartbeat without special-casing.
    `owners` (urn -> owner names) is surfaced as a "who to page" line when provided.
    """
    incident = build_incident(findings, owners=owners)
    if incident is None:
        return "✅ No drift detected across monitored datasets.\n"

    if llm is None:
        return render_markdown(incident)

    prompt = build_llm_prompt(incident)
    try:
        text = llm(prompt)
    except Exception:
        return render_markdown(incident)
    if not text or not text.strip():
        return render_markdown(incident)
    return text.strip() + "\n"
