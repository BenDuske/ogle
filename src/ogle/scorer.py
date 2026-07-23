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

import math
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
    MEAN = "mean"
    STDEV = "stdev"
    RANGE = "range"


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
    # Relative shift in a numeric field's mean that counts as covariate/feature drift
    # (0.25 = ±25% move vs the baseline mean). Both directions matter — a feature that
    # doubled or halved has moved under the model either way — unlike the drop-only
    # distribution rule. A field whose baseline mean is ~0 is skipped (relative shift is
    # undefined there; see `mean_zero_floor`), never guessed.
    mean_rel_threshold: float = 0.25
    # Baseline-mean magnitude below this is treated as "effectively zero": a relative shift
    # against it explodes (÷~0) and would page on trivial absolute wiggle, so the field is not
    # scored for mean drift rather than flagged on noise.
    mean_zero_floor: float = 1e-9
    # Relative shift in a numeric field's standard deviation that counts as spread/scale drift
    # (0.25 = ±25% move vs the baseline stdev). Both directions matter — a feature whose spread
    # collapsed (sensor stuck on one reading) or exploded (gone noisy) has moved its scale under
    # the model either way, a covariate shift the mean rule (location only) cannot see. A field
    # whose baseline stdev is ~0 is skipped (relative shift is undefined there; see
    # `stdev_zero_floor`), never guessed.
    stdev_rel_threshold: float = 0.25
    # Baseline-stdev magnitude below this is treated as "effectively zero": a relative shift
    # against it explodes (÷~0) and would page on trivial absolute wiggle, so the field is not
    # scored for spread drift rather than flagged on noise. (A field that was genuinely constant
    # — stdev 0 — going noisy is real, but scoring it relatively is undefined; it surfaces via
    # mean/distribution drift instead, and this stays quiet rather than dividing by zero.)
    stdev_zero_floor: float = 1e-9
    # Fraction of the baseline value-range (max - min) that a field's observed min/max may
    # escape before it counts as range/bounds drift (0.25 = the min or max broke out by a
    # quarter of the historical span). The breach is measured *relative to the field's own
    # baseline span*, so the band self-scales to each feature's natural range. Both directions
    # count — a max that shot up OR a min that dropped below the historical floor. A field whose
    # baseline span is ~0 (a constant column) is skipped: a relative breach against a zero span
    # is undefined, and that constant-goes-variable case already surfaces via stdev drift.
    range_rel_threshold: float = 0.25
    # Baseline span (max - min) below this is treated as "effectively zero": a relative breach
    # against it is undefined (division by ~0), so the field is not scored for range drift here
    # rather than paging on noise. Mirrors `mean_zero_floor` / `stdev_zero_floor`.
    range_zero_floor: float = 1e-9
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
    mean_threshold: Optional[float] = None,
    stdev_threshold: Optional[float] = None,
    range_threshold: Optional[float] = None,
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
    mean = (
        ScoreConfig.mean_rel_threshold if mean_threshold is None else mean_threshold
    )
    stdev = (
        ScoreConfig.stdev_rel_threshold if stdev_threshold is None else stdev_threshold
    )
    rng = (
        ScoreConfig.range_rel_threshold if range_threshold is None else range_threshold
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
    # A relative band with no upper cap (a 300% mean move is a legitimate threshold), just
    # strictly positive — zero/negative would flag every field and divide by ~0 in
    # `_severity_from_ratio`, same failure the volume band guards against.
    if not (mean > 0):
        raise ValueError(
            f"mean threshold must be > 0 (got {mean}); it is a relative mean-shift band."
        )
    # Same shape as the mean band: a relative spread band, strictly positive, no upper cap.
    # Zero/negative would flag every field and divide by ~0 in `_severity_from_ratio`.
    if not (stdev > 0):
        raise ValueError(
            f"stdev threshold must be > 0 (got {stdev}); it is a relative spread-shift band."
        )
    # Same shape as the mean/stdev bands: a relative breach band, strictly positive, no upper
    # cap. Zero/negative would flag every field and divide by ~0 in `_severity_from_ratio`.
    if not (rng > 0):
        raise ValueError(
            f"range threshold must be > 0 (got {rng}); it is a relative bounds-breach band."
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
        mean_rel_threshold=float(mean),
        stdev_rel_threshold=float(stdev),
        range_rel_threshold=float(rng),
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


def _cohens_d(
    base_mean: float,
    cur_mean: float,
    base_stdev: Optional[float],
    cur_stdev: Optional[float],
    zero_floor: float,
) -> Optional[float]:
    """Standardized mean difference (Cohen's d) — the first two-sample signal Ogle takes.

    The relative-shift rule that gates mean drift answers "how far did the center move,
    as a fraction of itself?" It is blind to the data's own spread: a 25% mean move on a
    razor-tight feature is a screaming covariate shift, while the same 25% on a fat-tailed
    field can be ordinary week-to-week noise. Cohen's d rescales the raw move by the pooled
    spread of the two samples, so the magnitude is expressed in standard deviations — the
    language of statistical significance, and the first concrete step toward the two-sample
    tests on Ogle's roadmap.

        d = (mean_cur - mean_base) / sqrt((s_base^2 + s_current^2) / 2)

    Signed: positive means the mean rose, negative means it fell, mirroring the raw move.
    Purely enrichment — it never gates the finding (that stays the relative-shift rule, so
    a field with no stdev is still flagged); it only annotates *how many sigma* the move is
    so a human can triage. Returns None when either side lacks a stdev (nothing to pool) or
    the pooled spread is below `zero_floor` (both samples ~constant — a standardized move is
    undefined there, and the constant-goes-variable case surfaces via stdev/range drift).
    """
    if base_stdev is None or cur_stdev is None:
        return None
    pooled = ((base_stdev ** 2 + cur_stdev ** 2) / 2.0) ** 0.5
    if pooled < zero_floor:
        return None
    return (cur_mean - base_mean) / pooled


def _effect_magnitude(d: float) -> str:
    """Classify a Cohen's d value into its conventional interpretation band.

    A raw `d=+2.3` on the alert is only actionable to someone who already carries Cohen's
    thresholds in their head. This maps |d| onto the standard bands so the narrative can say
    "large" next to the number and an operator can triage without the convention memorized:

        |d| < 0.2   negligible   (within-noise; the relative rule fired but the shift is small
                                  vs the field's own spread — a likely false-page candidate)
        0.2..0.5    small
        0.5..0.8    medium
        |d| >= 0.8  large        (a genuine population move, not spread-scaled noise)

    Sign-independent — a rise and a fall of equal standardized size read the same magnitude.
    Bands are the widely-cited Cohen (1988) cutoffs; a value on a boundary lands in the
    higher band. Pure labeling: it never gates the finding, it only makes the number legible.
    """
    mag = abs(d)
    if mag < 0.2:
        return "negligible"
    if mag < 0.5:
        return "small"
    if mag < 0.8:
        return "medium"
    return "large"


def _prob_superiority(d: float) -> float:
    """Common-language effect size: P(a random *current* value exceeds a random *baseline* one).

    Cohen's d says how far apart the two means are in pooled sigma, but "d=+2.3, large" is
    still statistician's shorthand. This converts the same number into the one quantity an
    operator reads without any convention memorized: the probability that a value drawn from
    the new distribution lands above a value drawn from the old one (McGraw & Wong's CLES).

    Under the same normal, equal-spread approximation Cohen's d already assumes, the
    difference of two independent draws is normal with mean (mu_cur - mu_base) and variance
    2*sigma^2, so:

        P(X_cur > X_base) = Phi(d / sqrt(2)) = 0.5 * (1 + erf(d / 2))

    d=0 -> 0.5 (distributions coincide, a coin flip); d=+0.8 (large) -> ~0.71; a big positive
    d saturates toward 1.0, a big negative d toward 0.0. Signed through d: it mirrors the
    direction of the mean move. Pure enrichment computed from d alone — no sample size, no new
    signature data — so it is defined exactly when the effect size is, and never gates a
    finding. Value is in [0, 1].
    """
    return 0.5 * (1.0 + math.erf(d / 2.0))


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


def score_mean(
    baseline: DatasetSignature, current: DatasetSignature, cfg: ScoreConfig
) -> Optional[DriftFinding]:
    """Numeric-mean shift — covariate/feature drift the other dimensions are blind to.

    A feature column can keep its schema, row count, null rate and cardinality while its
    *values* drift out from under a deployed model — sensor recalibrated, currency switched,
    a unit changed upstream, or the population genuinely moved. Schema/volume/quality/
    distribution all stay green; the one signal that moves is the field's mean. This scores
    the relative shift of each numeric field's mean against its baseline.

    Both directions are flagged (a feature that doubled *or* halved has moved), unlike the
    drop-only distribution rule. A field with no mean on either side is skipped (never
    guessed), and a baseline mean whose magnitude is below `cfg.mean_zero_floor` is skipped
    too — a relative shift against ~0 explodes and would page on trivial wiggle.
    """
    worsened: Dict[str, Dict[str, float]] = {}
    for path, cur_mean in current.field_means.items():
        base_mean = baseline.field_means.get(path)
        if base_mean is None:
            continue
        if abs(base_mean) < cfg.mean_zero_floor:
            continue
        rel_shift = abs(cur_mean - base_mean) / abs(base_mean)
        if rel_shift >= cfg.mean_rel_threshold:
            entry = {
                "baseline": base_mean,
                "current": cur_mean,
                "rel_shift": rel_shift,
            }
            # Enrich with the standardized effect size when both sides carry a stdev — how
            # many pooled sigma the mean moved. Never gates the finding; a field without a
            # stdev is still flagged on the relative rule, it just carries no effect size.
            effect = _cohens_d(
                base_mean,
                cur_mean,
                baseline.field_stdevs.get(path),
                current.field_stdevs.get(path),
                cfg.stdev_zero_floor,
            )
            if effect is not None:
                entry["effect_size"] = effect
                entry["effect_magnitude"] = _effect_magnitude(effect)
                # Same normal approximation d already carries, expressed as the one number an
                # operator reads cold: the chance a new row outranks an old one.
                entry["prob_superiority"] = _prob_superiority(effect)
            worsened[path] = entry

    if not worsened:
        return None

    max_shift = max(v["rel_shift"] for v in worsened.values())
    severity = _severity_from_ratio(max_shift, cfg.mean_rel_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["rel_shift"], reverse=True)

    def _annotate(p: str) -> str:
        base = (
            f"{p} ({worsened[p]['baseline']:g}->{worsened[p]['current']:g}, "
            f"{(worsened[p]['current'] - worsened[p]['baseline']) / abs(worsened[p]['baseline']):+.0%}"
        )
        eff = worsened[p].get("effect_size")
        if eff is not None:
            base += f", d={eff:+.1f} {worsened[p]['effect_magnitude']}"
            base += f", P(new>old)={worsened[p]['prob_superiority']:.0%}"
        return base + ")"

    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.MEAN,
        severity=severity,
        message="numeric mean shifted on " + ", ".join(_annotate(p) for p in fields),
        details={"fields": worsened},
    )


def score_stdev(
    baseline: DatasetSignature, current: DatasetSignature, cfg: ScoreConfig
) -> Optional[DriftFinding]:
    """Numeric-spread shift — the scale half of covariate drift the mean rule can't see.

    A feature can hold its schema, row count, null rate, cardinality *and its mean* while its
    *spread* moves out from under a deployed model: a sensor stuck on one reading (variance
    collapses toward 0), a source gone noisy (variance explodes), or a genuine change in
    population dispersion. Mean drift sees a shift in location; this sees a shift in scale — a
    symmetric spread change leaves the mean untouched, so only the standard deviation moves.
    This scores the relative shift of each numeric field's stdev against its baseline.

    Both directions are flagged (a spread that collapsed *or* blew up has moved), like the mean
    rule. A field with no stdev on either side is skipped (never guessed), and a baseline stdev
    whose magnitude is below `cfg.stdev_zero_floor` is skipped too — a relative shift against ~0
    explodes and would page on trivial wiggle.
    """
    worsened: Dict[str, Dict[str, float]] = {}
    for path, cur_stdev in current.field_stdevs.items():
        base_stdev = baseline.field_stdevs.get(path)
        if base_stdev is None:
            continue
        if abs(base_stdev) < cfg.stdev_zero_floor:
            continue
        rel_shift = abs(cur_stdev - base_stdev) / abs(base_stdev)
        if rel_shift >= cfg.stdev_rel_threshold:
            worsened[path] = {
                "baseline": base_stdev,
                "current": cur_stdev,
                "rel_shift": rel_shift,
            }

    if not worsened:
        return None

    max_shift = max(v["rel_shift"] for v in worsened.values())
    severity = _severity_from_ratio(max_shift, cfg.stdev_rel_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["rel_shift"], reverse=True)
    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.STDEV,
        severity=severity,
        message=(
            "numeric spread shifted on "
            + ", ".join(
                f"{p} (stdev {worsened[p]['baseline']:g}->{worsened[p]['current']:g}, "
                f"{(worsened[p]['current'] - worsened[p]['baseline']) / abs(worsened[p]['baseline']):+.0%})"
                for p in fields
            )
        ),
        details={"fields": worsened},
    )


def score_range(
    baseline: DatasetSignature, current: DatasetSignature, cfg: ScoreConfig
) -> Optional[DriftFinding]:
    """Numeric min/max bounds breach — the envelope drift the moment rules can't see.

    Mean drift sees the center move; stdev drift sees the spread move. Both are *aggregate
    moments*: a handful of out-of-bounds values — an integer overflow, a unit bug on a subset
    of rows, a new outlier regime — barely nudges either on a large table, yet those extremes
    push the field's observed min or max clean out of its historical envelope. This scores how
    far each numeric field's [min, max] escaped its baseline band, measured as a fraction of the
    baseline span (max - min) so the band self-scales to each feature's natural range.

    Both directions count (a max that shot up *or* a min that dropped below the historical
    floor). A field is scored only when it carries a full min AND max on BOTH sides — a partial
    envelope is skipped, never guessed. A baseline span below `cfg.range_zero_floor` (a constant
    column) is skipped too: a relative breach against a ~0 span is undefined, and that
    constant-goes-variable case already surfaces via stdev/distribution drift.
    """
    worsened: Dict[str, Dict[str, float]] = {}
    for path, cur_max in current.field_maxes.items():
        cur_min = current.field_mins.get(path)
        base_max = baseline.field_maxes.get(path)
        base_min = baseline.field_mins.get(path)
        if cur_min is None or base_max is None or base_min is None:
            continue
        span = base_max - base_min
        if span < cfg.range_zero_floor:
            continue
        breach_above = max(0.0, cur_max - base_max)
        breach_below = max(0.0, base_min - cur_min)
        breach = (breach_above + breach_below) / span
        if breach >= cfg.range_rel_threshold:
            worsened[path] = {
                "baseline_min": base_min,
                "baseline_max": base_max,
                "current_min": cur_min,
                "current_max": cur_max,
                "breach": breach,
            }

    if not worsened:
        return None

    max_breach = max(v["breach"] for v in worsened.values())
    severity = _severity_from_ratio(max_breach, cfg.range_rel_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["breach"], reverse=True)
    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.RANGE,
        severity=severity,
        message=(
            "numeric range breached on "
            + ", ".join(
                f"{p} ([{worsened[p]['baseline_min']:g}, {worsened[p]['baseline_max']:g}]"
                f"->[{worsened[p]['current_min']:g}, {worsened[p]['current_max']:g}], "
                f"+{worsened[p]['breach']:.0%} of span)"
                for p in fields
            )
        ),
        details={"fields": worsened},
    )


def score_freshness(
    current: DatasetSignature, cfg: ScoreConfig, now: Optional[float]
) -> Optional[DriftFinding]:
    """Data-freshness SLA — the silent-stall dimension the others can't see.

    A source whose ETL quietly stopped keeps its rows, schema, null and unique fractions
    unchanged, so schema/volume/quality/distribution/mean all stay green — yet the data is stale
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
        score_mean(baseline, current, cfg),
        score_stdev(baseline, current, cfg),
        score_range(baseline, current, cfg),
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
