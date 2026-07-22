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

from .signature import DatasetSignature, parse_iso_epoch


class DriftKind(str, Enum):
    SCHEMA = "schema"
    VOLUME = "volume"
    QUALITY = "quality"
    DISTRIBUTION = "distribution"
    FRESHNESS = "freshness"


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
    # Absolute *drop* in a field's distinct-value fraction that counts as distribution
    # drift (a categorical collapsing to one value, or a key losing uniqueness). Only a
    # drop is flagged — cardinality *rising* is usually benign (more variety, not breakage).
    unique_fraction_drop_threshold: float = 0.30
    # Data-freshness SLA in seconds. When set, a dataset whose profile timestamp
    # (`computed_at`) is older than this relative to the walk's `now` flags freshness drift —
    # the classic silent-stall failure (ETL stopped, rows unchanged so volume looks fine, but
    # the data is stale). Default None = OFF: freshness is opt-in per deployment (a nightly
    # table and a streaming source have very different SLAs), so it never pages uninvited.
    freshness_max_age_seconds: Optional[float] = None
    # If the source feeds a deployed model, bump every finding one severity step.
    escalate_when_serving: bool = True


def build_score_config(
    volume_threshold: Optional[float] = None,
    null_threshold: Optional[float] = None,
    unique_drop_threshold: Optional[float] = None,
    freshness_max_age_seconds: Optional[float] = None,
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
    uniq = (
        ScoreConfig.unique_fraction_drop_threshold
        if unique_drop_threshold is None
        else unique_drop_threshold
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
    if not (0 < uniq <= 1):
        raise ValueError(
            f"unique-drop threshold must be in (0, 1] (got {uniq}); it is an absolute "
            "distinct-value-fraction decrease."
        )
    # None keeps the freshness dimension OFF (the default). A supplied SLA must be a positive
    # duration — a zero/negative age would flag every timestamped dataset as stale, and
    # `_severity_from_ratio` divides by it. Reject up front, same as the other bands.
    fresh = (
        ScoreConfig.freshness_max_age_seconds
        if freshness_max_age_seconds is None
        else freshness_max_age_seconds
    )
    if fresh is not None and not (fresh > 0):
        raise ValueError(
            f"freshness max-age must be > 0 seconds (got {fresh}); it is a staleness SLA."
        )
    return ScoreConfig(
        volume_rel_threshold=float(vol),
        null_fraction_abs_threshold=float(nul),
        unique_fraction_drop_threshold=float(uniq),
        freshness_max_age_seconds=None if fresh is None else float(fresh),
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


def score_distribution(
    baseline: DatasetSignature, current: DatasetSignature, cfg: ScoreConfig
) -> Optional[DriftFinding]:
    """Distinct-value-fraction collapse — the cardinality half of distribution drift.

    Two real, silent failure modes share one signal, a *drop* in a field's unique fraction
    (distinct values / rows):
      * a categorical/feature column stuck on one value (upstream defaulting, a frozen
        enum) — the model keeps training but the feature carries no signal;
      * an id/join key that lost uniqueness — a fan-out join now duplicates rows.

    Only a drop is flagged. A *rise* in cardinality is usually benign (more variety), so
    flagging it would be noise on the serving path we're trying to keep quiet. A field with
    no unique fraction on either side is skipped, never guessed.
    """
    worsened: Dict[str, Dict[str, float]] = {}
    for path, cur_frac in current.field_unique_fractions.items():
        base_frac = baseline.field_unique_fractions.get(path)
        if base_frac is None:
            continue
        drop = base_frac - cur_frac
        if drop >= cfg.unique_fraction_drop_threshold:
            worsened[path] = {"baseline": base_frac, "current": cur_frac, "drop": drop}

    if not worsened:
        return None

    max_drop = max(v["drop"] for v in worsened.values())
    severity = _severity_from_ratio(max_drop, cfg.unique_fraction_drop_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["drop"], reverse=True)
    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.DISTRIBUTION,
        severity=severity,
        message=(
            "distinct-value fraction dropped on "
            + ", ".join(
                f"{p} ({worsened[p]['baseline']:.0%}->{worsened[p]['current']:.0%})"
                for p in fields
            )
        ),
        details={"fields": worsened},
    )


def score_freshness(
    current: DatasetSignature, cfg: ScoreConfig, now: Optional[float]
) -> Optional[DriftFinding]:
    """Data-freshness SLA — the silent-stall dimension the other four can't see.

    A source whose ETL quietly stopped keeps its rows, schema, null and unique fractions
    unchanged, so schema/volume/quality/distribution all stay green — yet the data is stale
    and every model retraining on it is learning yesterday's world. The one signal that moves
    is the profile timestamp (`computed_at`), which stops advancing. This dimension compares
    that stamp's age against a configured SLA.

    Opt-in and clock-injected, to keep the scorer pure and quiet by default:
      * `cfg.freshness_max_age_seconds is None` -> not scored (the default).
      * `now is None` -> not scored (no reference instant; never guess "now").
      * unparseable/absent `computed_at` -> not scored (age unknown; "never guess" holds).
    A future stamp (clock skew) clamps to age 0 rather than reading negative, mirroring the
    CLI's `_baseline_age_seconds`.
    """
    if cfg.freshness_max_age_seconds is None or now is None:
        return None
    epoch = parse_iso_epoch(current.computed_at)
    if epoch is None:
        return None
    age = max(0.0, now - epoch)
    if age < cfg.freshness_max_age_seconds:
        return None

    severity = _severity_from_ratio(age, cfg.freshness_max_age_seconds)
    age_h = age / 3600.0
    sla_h = cfg.freshness_max_age_seconds / 3600.0
    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.FRESHNESS,
        severity=severity,
        message=(
            f"data is stale: last profiled {current.computed_at} "
            f"({age_h:.1f}h ago, SLA {sla_h:.1f}h) — feed likely stalled"
        ),
        details={
            "computed_at": current.computed_at,
            "age_seconds": age,
            "max_age_seconds": cfg.freshness_max_age_seconds,
        },
    )


def score_dataset(
    baseline: DatasetSignature,
    current: DatasetSignature,
    cfg: Optional[ScoreConfig] = None,
    serving: bool = False,
    now: Optional[float] = None,
) -> List[DriftFinding]:
    """Score one dataset across all drift dimensions.

    `serving=True` means this dataset feeds a deployed (IN_SERVICE) model; findings are
    escalated one severity step because drift on a serving path is production-affecting.
    `now` (epoch seconds) is the reference instant the freshness dimension measures staleness
    against; None (the default) or an unconfigured SLA leaves freshness unscored.
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
        score_distribution(baseline, current, cfg),
        score_freshness(current, cfg, now),
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
