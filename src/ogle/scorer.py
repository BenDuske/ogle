"""Anomaly scorer — turns a pair of dataset signatures into drift findings.

Given a *baseline* signature (last-known-good, persisted from a prior run) and the
*current* signature, `score_dataset` returns the list of `DriftFinding`s that a human
should look at. This is the analytic heart of Ogle's W2 story; the narrative writer (LLM)
and tag write-back (W3) both consume these findings.

Design rules:
  * Pure and deterministic — no clock, no network, no LLM. Same inputs -> same findings.
  * Never guess. A dimension with no data on one side is skipped, not flagged.
  * Severity is earned by magnitude, and a source feeding a *deployed* model escalates
    (an ML model in production is the whole reason drift matters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .signature import DatasetSignature


class DriftKind(str, Enum):
    SCHEMA = "schema"
    VOLUME = "volume"
    QUALITY = "quality"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


@dataclass(frozen=True)
class DriftFinding:
    urn: str
    kind: DriftKind
    severity: Severity
    message: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "urn": self.urn,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ScoreConfig:
    """Thresholds. Defaults chosen to be quiet on noise, loud on real breakage."""

    # Relative row-count change that counts as volume drift (0.30 = ±30%).
    volume_rel_threshold: float = 0.30
    # A collapse to (near) empty is always serious regardless of the relative rule.
    volume_collapse_floor: int = 1
    # Absolute increase in a field's null fraction that counts as quality drift.
    null_fraction_abs_threshold: float = 0.20
    # If the source feeds a deployed model, bump every finding one severity step.
    escalate_when_serving: bool = True


def build_score_config(
    volume_threshold: Optional[float] = None,
    null_threshold: Optional[float] = None,
    escalate_when_serving: Optional[bool] = None,
) -> ScoreConfig:
    """Validated `ScoreConfig` builder — the single place CLI/config tuning lands.

    Each argument is optional; a `None` keeps the default. Sensitivity is a real
    per-deployment knob (a noisy dimension table wants a looser volume band than a
    stable serving-path source), so operators tune it — but a nonsensical threshold
    would silently break scoring (a zero/negative band divides by ~0 in
    `_severity_from_ratio`; a null band outside [0, 1] can never trip). Reject those
    here, up front, rather than emit garbage findings.
    """
    vol = ScoreConfig.volume_rel_threshold if volume_threshold is None else volume_threshold
    nul = (
        ScoreConfig.null_fraction_abs_threshold
        if null_threshold is None
        else null_threshold
    )
    esc = (
        ScoreConfig.escalate_when_serving
        if escalate_when_serving is None
        else bool(escalate_when_serving)
    )

    if not (vol > 0):
        raise ValueError(
            f"volume threshold must be > 0 (got {vol}); it is a relative row-count band."
        )
    if not (0 < nul <= 1):
        raise ValueError(
            f"null threshold must be in (0, 1] (got {nul}); it is an absolute "
            "null-fraction increase."
        )
    return ScoreConfig(
        volume_rel_threshold=float(vol),
        null_fraction_abs_threshold=float(nul),
        escalate_when_serving=esc,
    )


def _severity_from_ratio(ratio: float, threshold: float) -> Severity:
    """Map how far past the threshold a relative change is onto a severity band."""
    over = ratio / threshold if threshold > 0 else float("inf")
    if over >= 3.0:
        return Severity.HIGH
    if over >= 1.5:
        return Severity.MEDIUM
    return Severity.LOW


def _bump(sev: Severity) -> Severity:
    return {Severity.LOW: Severity.MEDIUM, Severity.MEDIUM: Severity.HIGH, Severity.HIGH: Severity.HIGH}[sev]


def score_schema(baseline: DatasetSignature, current: DatasetSignature) -> Optional[DriftFinding]:
    if baseline.schema_hash == current.schema_hash:
        return None

    base_types = {f.path: f.native_type for f in baseline.schema_fields}
    cur_types = {f.path: f.native_type for f in current.schema_fields}

    added = sorted(cur_types.keys() - base_types.keys())
    removed = sorted(base_types.keys() - cur_types.keys())
    retyped = sorted(
        p for p in base_types.keys() & cur_types.keys() if base_types[p] != cur_types[p]
    )

    # Removing or retyping a column that a feature reads is the breaking case; adding a
    # column is usually additive/safe.
    if removed or retyped:
        severity = Severity.HIGH
    else:
        severity = Severity.LOW

    parts = []
    if removed:
        parts.append(f"removed {removed}")
    if retyped:
        parts.append(f"retyped {[f'{p}:{base_types[p]}->{cur_types[p]}' for p in retyped]}")
    if added:
        parts.append(f"added {added}")

    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.SCHEMA,
        severity=severity,
        message="schema changed: " + "; ".join(parts),
        details={"added": added, "removed": removed, "retyped": retyped},
    )


def score_volume(
    baseline: DatasetSignature, current: DatasetSignature, cfg: ScoreConfig
) -> Optional[DriftFinding]:
    if baseline.row_count is None or current.row_count is None:
        return None
    base, cur = baseline.row_count, current.row_count

    # Collapse: was populated, now (near) empty. Pipeline broke — always HIGH.
    if base > 0 and cur < cfg.volume_collapse_floor:
        return DriftFinding(
            urn=current.urn,
            kind=DriftKind.VOLUME,
            severity=Severity.HIGH,
            message=f"row count collapsed {base} -> {cur} (source likely stopped loading)",
            details={"baseline": base, "current": cur, "rel_change": -1.0},
        )

    if base == 0:
        # No baseline volume to compare against; growth from empty isn't drift.
        return None

    rel_change = (cur - base) / base
    if abs(rel_change) < cfg.volume_rel_threshold:
        return None

    severity = _severity_from_ratio(abs(rel_change), cfg.volume_rel_threshold)
    direction = "grew" if rel_change > 0 else "shrank"
    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.VOLUME,
        severity=severity,
        message=f"row count {direction} {base} -> {cur} ({rel_change:+.0%})",
        details={"baseline": base, "current": cur, "rel_change": rel_change},
    )


def score_quality(
    baseline: DatasetSignature, current: DatasetSignature, cfg: ScoreConfig
) -> Optional[DriftFinding]:
    worsened: Dict[str, Dict[str, float]] = {}
    for path, cur_frac in current.field_null_fractions.items():
        base_frac = baseline.field_null_fractions.get(path)
        if base_frac is None:
            continue
        delta = cur_frac - base_frac
        if delta >= cfg.null_fraction_abs_threshold:
            worsened[path] = {"baseline": base_frac, "current": cur_frac, "delta": delta}

    if not worsened:
        return None

    max_delta = max(v["delta"] for v in worsened.values())
    severity = _severity_from_ratio(max_delta, cfg.null_fraction_abs_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["delta"], reverse=True)
    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.QUALITY,
        severity=severity,
        message=(
            "null rate spiked on "
            + ", ".join(f"{p} ({worsened[p]['baseline']:.0%}->{worsened[p]['current']:.0%})" for p in fields)
        ),
        details={"fields": worsened},
    )


def score_dataset(
    baseline: DatasetSignature,
    current: DatasetSignature,
    cfg: Optional[ScoreConfig] = None,
    serving: bool = False,
) -> List[DriftFinding]:
    """Score one dataset across all drift dimensions.

    `serving=True` means this dataset feeds a deployed (IN_SERVICE) model; findings are
    escalated one severity step because drift on a serving path is production-affecting.
    Returns findings sorted most-severe first.
    """
    if baseline.urn != current.urn:
        raise ValueError(
            f"cannot score across datasets: {baseline.urn!r} vs {current.urn!r}"
        )
    cfg = cfg or ScoreConfig()

    findings: List[DriftFinding] = []
    for finding in (
        score_schema(baseline, current),
        score_volume(baseline, current, cfg),
        score_quality(baseline, current, cfg),
    ):
        if finding is None:
            continue
        if serving and cfg.escalate_when_serving:
            finding = DriftFinding(
                urn=finding.urn,
                kind=finding.kind,
                severity=_bump(finding.severity),
                message=finding.message + " [serving]",
                details={**finding.details, "serving": True},
            )
        findings.append(finding)

    findings.sort(key=lambda f: f.severity.rank, reverse=True)
    return findings
