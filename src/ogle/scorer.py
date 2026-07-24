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
from typing import Dict, List, Optional, Sequence, Tuple

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


def _gaussian_hellinger(
    base_mean: float,
    cur_mean: float,
    base_stdev: Optional[float],
    cur_stdev: Optional[float],
    zero_floor: float,
) -> Optional[float]:
    """Hellinger distance between the two samples modeled as Gaussians — the first *joint*
    location-and-scale distributional signal, and Ogle's first step into the two-sample
    distribution-distance family (KS / PSI / Jensen–Shannon) the roadmap names next.

    Every other numeric signal splits the distribution into a piece. Cohen's d and the
    Welch z read the *mean* move rescaled by (but blind to changes in) spread; the stdev
    z reads the *spread* move ignoring location. Each can sit under its own threshold while
    the two distributions have, jointly, pulled well apart — a mean that crept half a sigma
    *and* a variance that doubled overlap far less than either move alone implies. Hellinger
    folds both moments into one number: how much probability mass the old and new
    distributions actually share.

    Modeling each side as a Gaussian from the mean+stdev already in the signature, the
    Bhattacharyya coefficient (their affinity) has a closed form, and Hellinger is its
    complement:

        BC = sqrt( 2*s_b*s_c / (s_b^2 + s_c^2) ) * exp( -0.25 * (m_c - m_b)^2 / (s_b^2 + s_c^2) )
        H  = sqrt( 1 - BC )

    H is a true metric bounded in [0, 1]: 0 when the distributions coincide, 1 when they are
    disjoint. Unsigned by design — it measures *separation*, not direction (the sign lives on
    d); a rise and a fall of equal joint magnitude read the same. Pure enrichment computed
    from the two moments alone — no sample size, no new signature data — so it never gates a
    finding; it only says how far the whole distribution moved, not just its center.

    Returns None when either side lacks a stdev (nothing to model) or the pooled spread is
    below `zero_floor` (both samples ~constant — the Gaussian degenerates to a spike and the
    constant-goes-variable case surfaces via stdev/range drift instead), mirroring the exact
    guard Cohen's d uses so the two annotations appear under the same conditions.
    """
    if base_stdev is None or cur_stdev is None:
        return None
    v_base = base_stdev ** 2
    v_cur = cur_stdev ** 2
    pooled = ((v_base + v_cur) / 2.0) ** 0.5
    if pooled < zero_floor:
        return None
    var_sum = v_base + v_cur
    coef = (2.0 * base_stdev * cur_stdev / var_sum) ** 0.5
    expo = math.exp(-0.25 * (cur_mean - base_mean) ** 2 / var_sum)
    bc = coef * expo
    # Clamp for float safety: BC is analytically in [0, 1], but rounding can nudge it a hair
    # past either end, which would make sqrt(1 - BC) NaN or push H above 1.
    bc = min(1.0, max(0.0, bc))
    return (1.0 - bc) ** 0.5


def _hellinger_band(h: float) -> str:
    """Classify a Hellinger distance into a plain-language separation band.

    A bare `H=0.42` is only actionable to someone who carries the metric's scale in their
    head. This maps H onto four bands so the narrative can say "moderate" next to the number
    and an operator can triage the joint move without the convention memorized:

        H < 0.1    negligible   (distributions all but coincide — the mean rule fired, but
                                 old and new still overlap almost entirely)
        0.1..0.3   small
        0.3..0.6   moderate
        H >= 0.6   large        (distributions largely disjoint — a genuine population move
                                 in location and/or scale, not a nudge)

    Bands are round cutoffs on the [0, 1] metric (there is no Cohen-style canon for
    Hellinger); a value on a boundary lands in the higher band. Pure labeling — it never
    gates the finding, it only makes the number legible.
    """
    if h < 0.1:
        return "negligible"
    if h < 0.3:
        return "small"
    if h < 0.6:
        return "moderate"
    return "large"


def _gaussian_psi(
    base_mean: float,
    cur_mean: float,
    base_stdev: Optional[float],
    cur_stdev: Optional[float],
    zero_floor: float,
) -> Optional[float]:
    """Population Stability Index between the two samples modeled as Gaussians — the *unbounded*
    twin of the Hellinger distance, and the metric the Devpost roadmap names first (PSI) in the
    two-sample distribution-distance family.

    PSI is the drift number ML practitioners actually deploy: bin both samples, then sum
    `(cur% - base%) * ln(cur% / base%)` over the bins. That sum is exactly the *symmetric* KL
    divergence (Jeffreys divergence) `KL(base‖cur) + KL(cur‖base)` of the two distributions —
    PSI is Jeffreys on a histogram. Modeling each side as the Gaussian its mean+stdev already
    imply, Jeffreys has a closed form with no binning, no bin-count knob, and no zero-bin
    epsilon fudge:

        J = 0.5 * (v_b/v_c + v_c/v_b - 2) + 0.5 * (m_c - m_b)^2 * (1/v_b + 1/v_c)

    where v = stdev^2. The first bracket is the pure *scale* penalty (0 iff the spreads match,
    growing as the variance ratio departs 1 in either direction); the second is the *location*
    penalty (the mean gap, weighted by both precisions). Both moments, one number — same joint
    reading as Hellinger, but where Hellinger saturates toward 1 once the Gaussians barely
    overlap, PSI keeps climbing, so it still *ranks* two already-far-apart moves. That is why an
    operator wants both: Hellinger for a calibrated [0,1] "how separated," PSI for the open-ended
    magnitude on the scale the industry's 0.1 / 0.25 thresholds are written against.

    Unsigned by design (it measures separation, not direction — that lives on Cohen's d). Pure
    enrichment computed from the two moments alone, so it never gates a finding.

    Returns None when either side lacks a stdev (nothing to model) or *either* stdev is below
    `zero_floor` — Jeffreys divides by each variance, so unlike Hellinger (which only needs the
    pooled spread non-zero) it needs both individual spreads non-degenerate; a variance collapse
    to ~0 is the constant-goes-variable case that surfaces via stdev/range drift instead.
    """
    if base_stdev is None or cur_stdev is None:
        return None
    if base_stdev < zero_floor or cur_stdev < zero_floor:
        return None
    v_base = base_stdev ** 2
    v_cur = cur_stdev ** 2
    scale_term = 0.5 * (v_base / v_cur + v_cur / v_base - 2.0)
    loc_term = 0.5 * (cur_mean - base_mean) ** 2 * (1.0 / v_base + 1.0 / v_cur)
    return scale_term + loc_term


def _psi_band(psi: float) -> str:
    """Classify a PSI value into the industry-canonical stability band.

    Unlike Hellinger, PSI carries a widely-used convention an operator likely already knows, so
    the bands are the standard cutoffs rather than invented ones:

        PSI < 0.1     stable        (no material population shift)
        0.1..0.25     moderate      (shift worth investigating)
        PSI >= 0.25   significant   (a real population move — retrain / action)

    A value on a boundary lands in the higher band. Pure labeling — it never gates the finding,
    it only puts the number on the scale those thresholds were written for.
    """
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def _gaussian_w2(
    base_mean: float,
    cur_mean: float,
    base_stdev: Optional[float],
    cur_stdev: Optional[float],
    zero_floor: float,
) -> Optional[float]:
    """2-Wasserstein (earth-mover) distance between the two samples modeled as Gaussians — the
    one distribution-distance in the family that reads in the feature's *own units*.

    Hellinger and PSI both answer "how far did the whole distribution move" as a *unitless*
    number: Hellinger a bounded [0,1] separation, PSI an unbounded divergence on the log scale.
    Powerful for ranking, but an operator still has to carry the metric's convention in their
    head — a bare `H=0.42` or `PSI=1.8` means nothing without it. The 2-Wasserstein distance
    answers the same joint-move question in the units the field is already printed in: it is the
    minimum "cost" (mass x ground distance) to reshape the old distribution into the new one, and
    for two Gaussians that optimal-transport cost collapses to a clean closed form:

        W2 = sqrt( (m_c - m_b)^2 + (s_c - s_b)^2 )

    The mean gap and the spread gap in quadrature — a plain Euclidean distance in the
    (location, scale) plane. When the spread is unchanged it is exactly the mean move, so it sits
    right next to the `100->140` already in the annotation and reads as "the distribution shifted
    by 40 <units>"; when the spread also moves it grows past the bare mean gap by the scale leg.
    A true metric (symmetric, zero iff the two Gaussians coincide), unsigned like its siblings —
    it measures how far, not which way (direction lives on Cohen's d).

    Where PSI keeps climbing without bound but on an abstract log scale, and Hellinger stays
    legible but saturates toward 1, W2 is the third face: unbounded *and* in real units, so it
    neither saturates nor needs a convention — it just says how far the mass had to travel.

    Pure enrichment from the two moments alone (no sample size, no new signature data), so it
    never gates a finding. Returns None under the *same* guard as Hellinger — both stdevs present
    and the pooled spread non-degenerate — so the two true-metric readings appear together (the
    constant-goes-variable case surfaces via stdev/range drift instead).
    """
    if base_stdev is None or cur_stdev is None:
        return None
    pooled = ((base_stdev ** 2 + cur_stdev ** 2) / 2.0) ** 0.5
    if pooled < zero_floor:
        return None
    return ((cur_mean - base_mean) ** 2 + (cur_stdev - base_stdev) ** 2) ** 0.5


def _w2_band(w2: float, base_stdev: float, cur_stdev: float) -> str:
    """Classify a 2-Wasserstein distance into a plain-language magnitude band.

    W2 is in the field's own units, so it has no universal cutoffs — "40" is huge for a rate in
    [0,1] and a rounding error for a dollar amount. To band it we standardize by the pooled
    spread (W2 / sqrt((s_b^2 + s_c^2)/2)): how many typical standard deviations of mass transport
    the move represents, a dimensionless magnitude that carries across features. On the pure
    mean-move axis that ratio reduces to |Cohen's d|, so the same Cohen-style cutoffs apply, but
    the scale leg lets it exceed d when the spread moved too:

        < 0.2    negligible
        0.2..0.5 small
        0.5..0.8 moderate
        >= 0.8   large

    A value on a boundary lands in the higher band. Pure labeling — it never gates the finding, it
    only makes the units-carrying number legible next to its unitless siblings.
    """
    pooled = ((base_stdev ** 2 + cur_stdev ** 2) / 2.0) ** 0.5
    ratio = w2 / pooled if pooled > 0 else 0.0
    if ratio < 0.2:
        return "negligible"
    if ratio < 0.5:
        return "small"
    if ratio < 0.8:
        return "moderate"
    return "large"


def _quantile_at(pairs: Sequence[Tuple[float, float]], p: float) -> float:
    """Value of the empirical quantile function at level `p`, piecewise-linear between knots.

    `pairs` is a sorted (p, v) quantile set (>= 2 points, guaranteed by `_clean_quantiles`).
    Between two known levels the quantile function is interpolated linearly; a `p` outside the
    known range is clamped to the nearest endpoint (flat extrapolation — never invent tail mass
    past the deepest quantile DataHub sampled). Pure and total.
    """
    if p <= pairs[0][0]:
        return pairs[0][1]
    if p >= pairs[-1][0]:
        return pairs[-1][1]
    for (p0, v0), (p1, v1) in zip(pairs, pairs[1:]):
        if p <= p1:
            span = p1 - p0
            if span <= 0:  # defensive; _clean_quantiles forbids it
                return v0
            return v0 + (v1 - v0) * (p - p0) / span
    return pairs[-1][1]


def _empirical_w1(
    base_q: Optional[Sequence[Tuple[float, float]]],
    cur_q: Optional[Sequence[Tuple[float, float]]],
) -> Optional[float]:
    """Empirical 1-Wasserstein (earth-mover) distance between two quantile functions.

    This is the *nonparametric* twin of the Gaussian W2: where `_gaussian_w2` models each side
    as a Gaussian from mean+stdev and folds the location+scale move into a closed form, this
    reads the raw quantiles DataHub sampled and measures the actual mass transport between the
    two empirical distributions — so it sees the skew and multimodal shape shifts a two-moment
    Gaussian summary is blind to (two distributions can share a mean and stdev yet have wildly
    different quantile functions).

    For one-dimensional distributions the optimal-transport cost has a clean quantile form: the
    1-Wasserstein distance is the integral of the gap between the two quantile functions,

        W1 = integral over p in [0,1] of |Q_cur(p) - Q_base(p)| dp

    In the field's own units, like W2 — "the average distance the mass had to travel". Computed
    by trapezoid over the union of the two sides' probability levels, each quantile function
    interpolated piecewise-linearly (`_quantile_at`). DataHub rarely samples the full [0,1]
    (typically 0.05..0.95), so the integral runs over the *shared* probability band [lo, hi]
    both sides cover and is normalized by its width — the mean quantile gap per unit probability,
    an honest in-units number that neither fabricates tail mass past the deepest sampled quantile
    nor rewards a wider-sampled side. Returns None when either side lacks a usable quantile set
    or the two share no probability band, so it rides *alongside* the Gaussian distances (which
    still fire from the moments) rather than replacing them.

    Pure enrichment — never gates a finding. Unsigned like its siblings (it measures how far,
    not which way; direction lives on Cohen's d).
    """
    if not base_q or not cur_q:
        return None
    lo = max(base_q[0][0], cur_q[0][0])
    hi = min(base_q[-1][0], cur_q[-1][0])
    if hi <= lo:
        return None  # no shared probability band — can't compare like-for-like
    grid = sorted(
        {p for p, _ in base_q if lo <= p <= hi}
        | {p for p, _ in cur_q if lo <= p <= hi}
        | {lo, hi}
    )
    gaps = [abs(_quantile_at(cur_q, p) - _quantile_at(base_q, p)) for p in grid]
    area = 0.0
    for i in range(len(grid) - 1):
        area += (grid[i + 1] - grid[i]) * (gaps[i] + gaps[i + 1]) / 2.0
    return area / (hi - lo)


def _cdf_at(pairs: Sequence[Tuple[float, float]], x: float) -> float:
    """Value of the empirical CDF at `x` — the probability-level inverse of `_quantile_at`.

    `pairs` is the same sorted (p, v) quantile set the quantile-function helper reads; here we
    invert it, reading the value axis to answer "what fraction of the distribution sits at or
    below `x`". Because the quantile function is monotone non-decreasing, its knots' values are
    non-decreasing too, so a piecewise-linear interpolation on the (v, p) segments gives the CDF.

    Outside the sampled value range the CDF is clamped to that side's nearest sampled level
    (`p_lo` below the smallest sampled value, `p_hi` above the largest) — the same flat
    extrapolation `_quantile_at` uses, and for the same reason: DataHub rarely samples the tails,
    so we never assert 0 or 1 mass past the deepest quantile it actually measured. A flat segment
    in the quantile function (v0 == v1, a vertical jump in the CDF) resolves to the upper level,
    keeping the CDF right-continuous. Pure and total.
    """
    if x <= pairs[0][1]:
        return pairs[0][0]
    if x >= pairs[-1][1]:
        return pairs[-1][0]
    for (p0, v0), (p1, v1) in zip(pairs, pairs[1:]):
        if x <= v1:
            span = v1 - v0
            if span <= 0:  # vertical CDF jump; report the upper (right-continuous) level
                return p1
            return p0 + (p1 - p0) * (x - v0) / span
    return pairs[-1][0]


def _empirical_ks(
    base_q: Optional[Sequence[Tuple[float, float]]],
    cur_q: Optional[Sequence[Tuple[float, float]]],
) -> Optional[float]:
    """Two-sample Kolmogorov-Smirnov statistic between two empirical quantile functions.

    The next member of the *empirical* distance family after W1 (`_empirical_w1`): where the
    Wasserstein twin reads how *far* the mass moved (an in-units integral of the quantile gap),
    KS reads how *separated* the two distributions are at their point of maximum divergence — the
    largest vertical gap between the two CDFs,

        D = sup over x of |F_cur(x) - F_base(x)|

    the classic nonparametric two-sample statistic. It is bounded [0, 1], unitless, and — like
    Hellinger — needs no stdev: it rides on the raw quantiles alone, so a field carrying sampled
    quantiles but a degenerate spread still gets a whole-distribution separation number. Because
    it is nonparametric it also sees the skew/multimodal shifts the Gaussian H/PSI/W2 idealize
    away; W1 answers "how far did it move", KS answers "how cleanly do the two populations pull
    apart" — a big location shift with tiny spread pins KS near its ceiling, while a broad
    overlapping smear keeps it low even at the same W1.

    Computed by inverting each quantile function to a CDF (`_cdf_at`) and taking the max gap over
    the union of both sides' knot values — for two piecewise-linear CDFs the supremum is attained
    at a breakpoint, so the finite knot set is exact, not a sample. Guarded exactly like W1: it
    needs a shared probability band both sides sampled (`lo < hi`), because two sides whose levels
    don't overlap tell us nothing comparable — reading a separation off non-overlapping tails
    would fabricate a difference DataHub never measured. Each side's CDF still clamps to its own
    sampled band, so the statistic is honestly capped by that band's width (a 0.05..0.95 sample
    can show at most 0.9) rather than pretending to see the untracked tails.

    Returns None when either side lacks a usable quantile set or the two share no probability
    band. Pure enrichment, never gates a finding; unsigned like its siblings (it measures how
    separated, not which way — direction lives on Cohen's d).
    """
    if not base_q or not cur_q:
        return None
    lo = max(base_q[0][0], cur_q[0][0])
    hi = min(base_q[-1][0], cur_q[-1][0])
    if hi <= lo:
        return None  # no shared probability band — can't compare like-for-like
    xs = sorted({v for _, v in base_q} | {v for _, v in cur_q})
    return max(abs(_cdf_at(cur_q, x) - _cdf_at(base_q, x)) for x in xs)


def _ks_band(d: float) -> str:
    """Classify a Kolmogorov-Smirnov statistic into a plain-language separation band.

    KS is a bounded [0, 1] separation with no Cohen-style canon, so — like `_hellinger_band` — we
    map it onto round cutoffs the narrative can say aloud next to the number:

        D < 0.1    negligible   (CDFs track each other everywhere — no clean population split)
        0.1..0.25  small
        0.25..0.5  moderate
        D >= 0.5   large        (the two populations pull at least half apart at some value — a
                                 decisive nonparametric separation, shape shifts included)

    A value on a boundary lands in the higher band. Pure labeling — it never gates the finding.
    """
    if d < 0.1:
        return "negligible"
    if d < 0.25:
        return "small"
    if d < 0.5:
        return "moderate"
    return "large"


def _mean_shift_z(
    base_mean: float,
    cur_mean: float,
    base_stdev: Optional[float],
    cur_stdev: Optional[float],
    n_base: Optional[int],
    n_cur: Optional[int],
    zero_floor: float,
) -> Optional[float]:
    """Welch two-sample z-statistic — the first mean-drift signal that weighs *sample size*.

    Cohen's d and its CLES readout answer "how big is the move relative to the spread?" but
    both are blind to how much data backs each side: they read identically whether each mean
    came from 5 rows or 5 million. That is exactly the question a page needs answered, though —
    a half-sigma shift measured on a handful of rows is sampling noise, while the same shift on
    a million rows is a certainty. This is the first signal that closes that gap: it scales the
    raw mean move by the *standard error of that difference*, which shrinks as the samples grow.

        SE = sqrt(s_base^2 / n_base + s_current^2 / n_current)      (Welch — unequal variances)
        z  = (mean_cur - mean_base) / SE

    Welch's form pools the two sampling variances without assuming the spreads are equal, so it
    stays honest when one side is noisier than the other. Signed like d: positive means the mean
    rose. Purely enrichment — it never gates the finding (that stays the relative-shift rule) —
    but where d only says the move is large, z says whether it is *distinguishable from noise*.

    Returns None when either stdev is missing (nothing to build a standard error from), when a
    sample size is missing or below 2 (a one-row sample has no sampling spread to speak of), or
    when the standard error is below `zero_floor` (both samples ~constant — z is undefined there,
    and the constant-goes-variable case already surfaces via stdev/range drift).
    """
    if base_stdev is None or cur_stdev is None:
        return None
    if n_base is None or n_cur is None or n_base < 2 or n_cur < 2:
        return None
    se = (base_stdev ** 2 / n_base + cur_stdev ** 2 / n_cur) ** 0.5
    if se < zero_floor:
        return None
    return (cur_mean - base_mean) / se


# Two-sided 95% normal critical value (z_{0.975}). Used to widen the mean-difference
# standard error into a confidence interval. Hardcoded to the conventional 95% level —
# the same "widely-cited default" spirit as Cohen's effect-size bands — so the CI needs
# no inverse-normal (the stdlib carries the forward erf/erfc but no ppf).
_Z_CRIT_95 = 1.959963984540054


def _mean_shift_ci(
    base_mean: float,
    cur_mean: float,
    base_stdev: Optional[float],
    cur_stdev: Optional[float],
    n_base: Optional[int],
    n_cur: Optional[int],
    zero_floor: float,
) -> Optional[Tuple[float, float]]:
    """95% confidence interval for the mean difference, in the field's *original units*.

    The z-statistic and its p-value answer "is the move distinguishable from noise?" and
    Cohen's d answers "how big is it in pooled sigma?" — but both discard the one quantity
    an operator triages a numeric drift with: how far the mean actually moved, in the field's
    own units, and how tightly that move is pinned down. A page that says "amount rose by 40,
    95% CI [31, 49]" is immediately actionable — the shift is real *and* its magnitude is
    bounded away from trivial; "[-2, 82]" says the same point estimate can't even be signed.

    Built on the same Welch standard error `_mean_shift_z` uses, so it is defined under exactly
    the same conditions and stays honest when one side is noisier than the other:

        SE   = sqrt(s_base^2 / n_base + s_current^2 / n_current)      (Welch — unequal variances)
        diff = mean_cur - mean_base
        CI   = diff +/- z_{0.975} * SE            (z_{0.975} = 1.959964, the textbook 5% cutoff)

    Signed and ordered (lo <= hi), centered on the raw move. A normal (z, not t) interval to
    match `_mean_shift_z`'s large-sample assumption — the ogle-scale row counts that back a
    field's mean put the t correction well inside rounding. The interval excludes 0 exactly
    when the two-sided p < 0.05, so it is the same significance verdict expressed as a *range*
    instead of a single number. Purely enrichment — it never gates the finding.

    Returns None under the same guards as the z-statistic: either stdev missing (no SE to build),
    a sample size missing or below 2 (no sampling spread), or SE below `zero_floor` (both samples
    ~constant — the interval collapses and the constant-goes-variable case surfaces elsewhere).
    """
    if base_stdev is None or cur_stdev is None:
        return None
    if n_base is None or n_cur is None or n_base < 2 or n_cur < 2:
        return None
    se = (base_stdev ** 2 / n_base + cur_stdev ** 2 / n_cur) ** 0.5
    if se < zero_floor:
        return None
    diff = cur_mean - base_mean
    half = _Z_CRIT_95 * se
    return (diff - half, diff + half)


def _spread_shift_z(
    base_stdev: float,
    cur_stdev: float,
    n_base: Optional[int],
    n_cur: Optional[int],
    zero_floor: float,
) -> Optional[float]:
    """Log-SD two-sample z-statistic — the spread rule's answer to "is this move real?".

    The stdev rule fires on a relative-spread shift the same way the mean rule fires on a
    relative-mean shift, and it was just as blind to sample size: a spread that "halved" on a
    handful of rows is sampling noise, while the same halving on a million rows is a certainty.
    This is the scale-side twin of `_mean_shift_z`. The right large-sample test for a *ratio* of
    standard deviations is not a difference-of-SDs z — the SD estimator's own spread grows with
    its magnitude, so a raw difference is heteroscedastic. Working in log space fixes that: for a
    sample of size n, ln(s) is approximately normal with variance 1 / (2(n-1)) regardless of the
    true scale (delta method on the chi-square of the sample variance). So the log-ratio has a
    clean standard error and

        SE = sqrt( 1 / (2(n_base - 1)) + 1 / (2(n_cur - 1)) )
        z  = ( ln(s_current) - ln(s_base) ) / SE

    Signed like the shift: positive means the spread grew, negative means it collapsed. Purely
    enrichment — it never gates the finding (that stays the relative-shift rule) — but where the
    relative shift only says the spread moved, z says whether the move is distinguishable from
    sampling noise, and feeds the same `_two_sided_p` / BH machinery the mean rule already uses.

    Returns None when either stdev is at or below `zero_floor` (log undefined at ~0, and the
    variance-collapse-to-constant case already surfaces via the range/stdev magnitude rules), or
    when a sample size is missing or below 2 (one row carries no dispersion to speak of).
    """
    if base_stdev <= zero_floor or cur_stdev <= zero_floor:
        return None
    if n_base is None or n_cur is None or n_base < 2 or n_cur < 2:
        return None
    se = (1.0 / (2 * (n_base - 1)) + 1.0 / (2 * (n_cur - 1))) ** 0.5
    if se < zero_floor:
        return None
    return (math.log(cur_stdev) - math.log(base_stdev)) / se


def _spread_shift_ci(
    base_stdev: float,
    cur_stdev: float,
    n_base: Optional[int],
    n_cur: Optional[int],
    zero_floor: float,
) -> Optional[Tuple[float, float]]:
    """95% confidence interval for the spread *ratio* (current stdev / baseline stdev).

    `_spread_shift_z` and its p-value answer "is the spread move distinguishable from noise?",
    but — exactly like the mean rule before it got `_mean_shift_ci` — they collapse the move to
    a single verdict and discard its magnitude. The operator triaging a spread drift wants the
    bound: did the spread grow by "1.4x-2.1x" (real and sized) or "0.8x-2.5x" (can't even tell
    if it grew or shrank)? This is the scale-side twin of the mean's confidence interval.

    A stdev is a *ratio* quantity, not a difference: the natural interval lives in log space,
    where `_spread_shift_z` already establishes ln(s) is approximately normal with the clean
    standard error SE = sqrt(1/(2(n_base-1)) + 1/(2(n_cur-1))). Build the symmetric interval on
    the log-ratio, then exponentiate back so the operator reads a *multiplicative* band on the
    ratio itself:

        logr = ln(s_current) - ln(s_base)
        CI   = ( exp(logr - z_{0.975} * SE),  exp(logr + z_{0.975} * SE) )

    The interval is strictly positive (a stdev ratio can never be negative — exp guarantees it),
    ordered (lo <= hi), asymmetric in original units (the multiplicative geometry a ratio wants),
    and — because it is built from the same SE as the z — it brackets 1.0 exactly when the
    two-sided p >= 0.05, so it is the same significance verdict expressed as a *range*. A ratio
    below 1 means the spread collapsed, above 1 means it grew. Purely enrichment — it never gates
    the finding (that stays the relative-shift rule).

    Returns None under the same guards as `_spread_shift_z`: either stdev at or below `zero_floor`
    (log undefined at ~0), a sample size missing or below 2 (no dispersion to bound), or SE below
    `zero_floor` (the interval collapses to a point).
    """
    if base_stdev <= zero_floor or cur_stdev <= zero_floor:
        return None
    if n_base is None or n_cur is None or n_base < 2 or n_cur < 2:
        return None
    se = (1.0 / (2 * (n_base - 1)) + 1.0 / (2 * (n_cur - 1))) ** 0.5
    if se < zero_floor:
        return None
    logr = math.log(cur_stdev) - math.log(base_stdev)
    half = _Z_CRIT_95 * se
    return (math.exp(logr - half), math.exp(logr + half))


def _null_shift_z(
    base_frac: float,
    cur_frac: float,
    n_base: Optional[int],
    n_cur: Optional[int],
) -> Optional[float]:
    """Two-proportion z-statistic — the null-rate rule's answer to "is this spike real?".

    The quality rule fires on an *absolute* jump in a field's null fraction the same way the
    mean rule fires on a relative mean shift, and it was just as blind to sample size: a null
    rate that "jumped" from 0% to 40% on 5 rows is two extra nulls — sampling noise — while the
    same jump on a million rows is a pipeline that broke. A null fraction is a *proportion*, so
    the right two-sample test is not a z on means but the classic two-proportion z-test, which
    pools the two samples under the null of equal rates and scores the observed gap against the
    standard error of that pooled proportion:

        p_pool = (x_base + x_cur) / (n_base + n_cur)      x = frac * n  (the null counts)
        SE     = sqrt( p_pool * (1 - p_pool) * (1/n_base + 1/n_cur) )
        z      = (cur_frac - base_frac) / SE

    Note the sample size here is the *full* row count on each side, not the effective-n the mean
    and spread rules use: every row either is or is not null, so all rows carry the proportion —
    there is no "net of nulls" to subtract. Signed like the shift (quality only flags increases,
    so z is >= 0). Purely enrichment — it never gates the finding (that stays the absolute-jump
    rule) — and feeds the same `_two_sided_p` / BH machinery the mean and spread rules already use.

    Returns None when a row count is missing or below 1 (no denominator for a proportion), or when
    the pooled variance is <= 0 (both sides all-null or all-populated — degenerate, and a jump that
    large is already carried by the absolute rule).
    """
    if n_base is None or n_cur is None or n_base < 1 or n_cur < 1:
        return None
    x_base = base_frac * n_base
    x_cur = cur_frac * n_cur
    p_pool = (x_base + x_cur) / (n_base + n_cur)
    var = p_pool * (1.0 - p_pool) * (1.0 / n_base + 1.0 / n_cur)
    if var <= 0:
        return None
    return (cur_frac - base_frac) / (var ** 0.5)


def _effective_n(sig: DatasetSignature, path: str) -> Optional[int]:
    """Rows that actually back a field's mean: total row count net of its null fraction.

    A field's sample size for a two-sample test is not the table's row count — it is the count
    of *non-null* values, since nulls carry no measurement. A 10k-row table where a field is 40%
    null backs that field's mean with only ~6k samples, and the standard error should reflect
    that. Returns None when the row count is unknown (no denominator to scale); a missing null
    fraction is treated as fully populated (0% null), matching how the signature omits it.
    """
    if sig.row_count is None:
        return None
    null_frac = sig.field_null_fractions.get(path, 0.0)
    return int(round(sig.row_count * (1.0 - null_frac)))


def _two_sided_p(z: float) -> float:
    """Two-sided normal p-value for a z-statistic: P(|Z| >= |z|) under the null of no shift.

    The z-score from `_mean_shift_z` grows without bound as samples pile up, so a bare "z=41"
    is as opaque as a bare d. This maps it onto the one number an operator reads cold: the
    probability of seeing a mean move at least this extreme *if nothing actually changed*. Small
    p (say < 0.01) means the shift is very unlikely to be sampling luck — a real move worth a
    page; a p near 1 means the relative rule fired but the data can't distinguish it from noise.

        p = P(|Z| >= |z|) = erfc(|z| / sqrt(2))

    Sign-independent (a rise and an equal-magnitude fall are equally surprising under the null),
    monot, and in [0, 1]: z=0 -> 1.0 (no evidence of a shift), |z|=1.96 -> ~0.05, big |z| -> ~0.
    Pure labeling computed from z alone — defined exactly when the z-statistic is.
    """
    return math.erfc(abs(z) / math.sqrt(2.0))


def _bh_qvalues(pvals: Dict[str, float]) -> Dict[str, float]:
    """Benjamini-Hochberg FDR-adjusted q-values across a family of simultaneous p-values.

    A two-sided p-value is honest for *one* field, but a mean finding tests every drifted
    numeric field at once, and testing many fields multiplies the chance that pure noise trips
    the threshold somewhere: 20 unchanged fields each tested at p<0.05 yield ~1 "significant"
    hit by luck alone. Reading each raw p in isolation therefore over-pages exactly when a wide
    table drifts. This applies the Benjamini-Hochberg step-up to the whole family, converting
    each raw p into a q-value — the false-discovery rate you accept if you call everything at or
    below it a real move.

    Procedure: sort the m p-values ascending (ranks 1..m); the rank-i value gets p_(i) * m / i,
    then enforce monotonicity from the largest rank downward (q_(i) = min(q_(i), q_(i+1))) and
    clamp to [0, 1] so a q never exceeds 1. With one test q == p (nothing to correct); with many,
    a lone small p among unchanged fields is pushed back toward 1 unless several fields move
    together, while a genuinely broad drift keeps its low q. Pure enrichment keyed by the same
    field paths — never gates a finding, only tells the operator which of several simultaneous
    "significant" moves survive multiplicity. Returned q-values are in [0, 1].
    """
    if not pvals:
        return {}
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(ordered)
    q: Dict[str, float] = {}
    # Walk from the largest p (rank m) down to the smallest (rank 1), carrying the running
    # minimum so the adjusted values stay monotone in the raw-p ordering (BH step-up).
    running_min = 1.0
    for rank in range(m, 0, -1):
        path, p = ordered[rank - 1]
        running_min = min(running_min, min(p * m / rank, 1.0))
        q[path] = running_min
    return q


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
            entry = {"baseline": base_frac, "current": cur_frac, "delta": delta}
            # Significance: does the null-rate jump survive the sample size, or is it a couple
            # of extra nulls on a tiny table? A null fraction is a proportion, so the sample
            # size is the full row count on each side (every row carries the proportion — no
            # net-of-nulls). Purely enrichment — a field with an unknown row count is still
            # flagged on the absolute-jump rule, it just carries no z.
            z = _null_shift_z(base_frac, cur_frac, baseline.row_count, current.row_count)
            if z is not None:
                entry["z_score"] = z
                entry["p_value"] = _two_sided_p(z)
            worsened[path] = entry

    if not worsened:
        return None

    # Multiple-comparison control, symmetric to the mean/stdev rules: a quality finding tests every
    # field's null rate at once, so a raw per-field p over-states significance on a wide table. When
    # two or more fields carry a p-value, adjust the family with Benjamini-Hochberg so each entry also
    # reports the false-discovery rate at which it would be called real. A single test needs no
    # correction (q == p), so it is left alone to keep the annotation clean.
    pvals = {p: e["p_value"] for p, e in worsened.items() if "p_value" in e}
    if len(pvals) >= 2:
        for p, qv in _bh_qvalues(pvals).items():
            worsened[p]["q_value"] = qv

    max_delta = max(v["delta"] for v in worsened.values())
    severity = _severity_from_ratio(max_delta, cfg.null_fraction_abs_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["delta"], reverse=True)

    def _annotate(p: str) -> str:
        base = f"{p} ({worsened[p]['baseline']:.0%}->{worsened[p]['current']:.0%}"
        pval = worsened[p].get("p_value")
        if pval is not None:
            base += f", p={pval:.1g}"
            qval = worsened[p].get("q_value")
            if qval is not None:
                base += f", q={qval:.1g}"
        return base + ")"

    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.QUALITY,
        severity=severity,
        message="null rate spiked on " + ", ".join(_annotate(p) for p in fields),
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
            # Joint location+scale signal: how far the *whole* distribution moved, not just its
            # center. Cohen's d rescales the mean move by spread but is blind to a spread change
            # itself; Hellinger folds both moments into one bounded [0,1] separation. Same guard
            # as d (needs both stdevs, non-degenerate spread), so it rides alongside it.
            dist = _gaussian_hellinger(
                base_mean,
                cur_mean,
                baseline.field_stdevs.get(path),
                current.field_stdevs.get(path),
                cfg.stdev_zero_floor,
            )
            if dist is not None:
                entry["dist_shift"] = dist
                entry["dist_magnitude"] = _hellinger_band(dist)
            # The unbounded twin of Hellinger: Gaussian PSI (symmetric KL / Jeffreys). Hellinger
            # saturates near 1 once the two Gaussians barely overlap, so it stops ranking moves
            # that are all "far"; PSI keeps climbing and reads on the scale the industry's
            # 0.1/0.25 thresholds are written against. Stricter guard than Hellinger (needs each
            # stdev non-degenerate, not just the pooled spread), so a field can carry H but not
            # PSI when one side is ~constant.
            psi = _gaussian_psi(
                base_mean,
                cur_mean,
                baseline.field_stdevs.get(path),
                current.field_stdevs.get(path),
                cfg.stdev_zero_floor,
            )
            if psi is not None:
                entry["psi"] = psi
                entry["psi_band"] = _psi_band(psi)
            # The in-units face of the same joint move: the 2-Wasserstein (earth-mover) distance,
            # sqrt(dmean^2 + dstdev^2). Hellinger and PSI both read unitless (bounded separation /
            # unbounded divergence); W2 reports how far the whole distribution shifted in the
            # field's own units, so it sits directly comparable to the mean values in the line.
            # Same guard as Hellinger, so the two true-metric readings ride together.
            w2 = _gaussian_w2(
                base_mean,
                cur_mean,
                baseline.field_stdevs.get(path),
                current.field_stdevs.get(path),
                cfg.stdev_zero_floor,
            )
            if w2 is not None:
                entry["w2"] = w2
                entry["w2_band"] = _w2_band(
                    w2,
                    baseline.field_stdevs[path],
                    current.field_stdevs[path],
                )
            # The EMPIRICAL twin of that W2: the 1-Wasserstein between the two sides' raw quantile
            # functions. W2/H/PSI all model each side as a Gaussian from mean+stdev, so they are
            # blind to skew and multimodal shape shifts that leave both moments unchanged; W1 reads
            # DataHub's sampled quantiles directly and sees the actual mass transport. In the
            # field's own units like W2, so it sits directly beside it. Fires only when BOTH sides
            # carry a usable quantile set (independent of the stdev guard above), and is banded by
            # the pooled spread exactly like W2 when both stdevs are present.
            w1 = _empirical_w1(
                baseline.field_quantiles.get(path),
                current.field_quantiles.get(path),
            )
            if w1 is not None:
                entry["w1_emp"] = w1
                bs = baseline.field_stdevs.get(path)
                cs = current.field_stdevs.get(path)
                if bs is not None and cs is not None:
                    entry["w1_emp_band"] = _w2_band(w1, bs, cs)
            # The other empirical face: the two-sample Kolmogorov-Smirnov separation between the
            # two sides' raw quantile CDFs. W1 (above) reads how far the mass moved in the field's
            # units; KS reads how cleanly the two populations pull apart at their point of maximum
            # divergence — a bounded [0,1] separation that, being nonparametric, sees the skew and
            # multimodal shifts the Gaussian H/PSI/W2 idealize away. Rides on the quantiles alone
            # (no stdev guard) and is already unitless, so unlike W1 it always carries its band.
            ks = _empirical_ks(
                baseline.field_quantiles.get(path),
                current.field_quantiles.get(path),
            )
            if ks is not None:
                entry["ks"] = ks
                entry["ks_band"] = _ks_band(ks)
            # Significance: does the move survive the sample size, or is it noise? Effective
            # per-field n = rows that actually carry a value (row_count net of the null
            # fraction) — a field 40% null on 10k rows only backs the mean with 6k samples.
            z = _mean_shift_z(
                base_mean,
                cur_mean,
                baseline.field_stdevs.get(path),
                current.field_stdevs.get(path),
                _effective_n(baseline, path),
                _effective_n(current, path),
                cfg.stdev_zero_floor,
            )
            if z is not None:
                entry["z_score"] = z
                entry["p_value"] = _two_sided_p(z)
            # Bound the move: the 95% CI for the mean difference in the field's own units, so
            # the page carries not just "significant" but *how far* it plausibly moved. Same
            # Welch SE as the z-test, so it appears under exactly the same conditions.
            ci = _mean_shift_ci(
                base_mean,
                cur_mean,
                baseline.field_stdevs.get(path),
                current.field_stdevs.get(path),
                _effective_n(baseline, path),
                _effective_n(current, path),
                cfg.stdev_zero_floor,
            )
            if ci is not None:
                entry["ci_low"], entry["ci_high"] = ci
            worsened[path] = entry

    if not worsened:
        return None

    # Multiple-comparison control: a mean finding tests every drifted numeric field at once, so
    # a raw per-field p over-states significance on a wide table. When two or more fields carry
    # a p-value, adjust the family with Benjamini-Hochberg so each entry also reports the
    # false-discovery rate at which it would be called real. A single test needs no correction
    # (q == p), so it is left alone to keep the annotation clean.
    pvals = {p: e["p_value"] for p, e in worsened.items() if "p_value" in e}
    if len(pvals) >= 2:
        for p, qv in _bh_qvalues(pvals).items():
            worsened[p]["q_value"] = qv

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
        dist = worsened[p].get("dist_shift")
        if dist is not None:
            base += f", H={dist:.2f} {worsened[p]['dist_magnitude']}"
        psi = worsened[p].get("psi")
        if psi is not None:
            base += f", PSI={psi:.2f} {worsened[p]['psi_band']}"
        w2 = worsened[p].get("w2")
        if w2 is not None:
            base += f", W2={w2:g} {worsened[p]['w2_band']}"
        w1 = worsened[p].get("w1_emp")
        if w1 is not None:
            base += f", W1emp={w1:g}"
            w1band = worsened[p].get("w1_emp_band")
            if w1band is not None:
                base += f" {w1band}"
        ks = worsened[p].get("ks")
        if ks is not None:
            base += f", KS={ks:.2f} {worsened[p]['ks_band']}"
        pval = worsened[p].get("p_value")
        if pval is not None:
            base += f", p={pval:.1g}"
            qval = worsened[p].get("q_value")
            if qval is not None:
                base += f", q={qval:.1g}"
        lo = worsened[p].get("ci_low")
        if lo is not None:
            base += f", 95% CI [{lo:+g}, {worsened[p]['ci_high']:+g}]"
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
            entry = {
                "baseline": base_stdev,
                "current": cur_stdev,
                "rel_shift": rel_shift,
            }
            # Significance: does the spread move survive the sample size, or is it noise? Same
            # effective per-field n as the mean rule (rows net of the null fraction), scored in
            # log space so a ratio of stdevs gets an honest standard error. Purely enrichment — a
            # field without a usable n or with a ~0 stdev is still flagged on the relative rule,
            # it just carries no z.
            z = _spread_shift_z(
                base_stdev,
                cur_stdev,
                _effective_n(baseline, path),
                _effective_n(current, path),
                cfg.stdev_zero_floor,
            )
            if z is not None:
                entry["z_score"] = z
                entry["p_value"] = _two_sided_p(z)
            # Bound the move: the 95% CI for the spread ratio (current/baseline stdev), so the
            # page carries not just "significant" but *how many times* the spread plausibly
            # moved. Same log-space SE as the z-test, so it appears under exactly the same
            # conditions — the scale-side twin of the mean rule's confidence interval.
            ci = _spread_shift_ci(
                base_stdev,
                cur_stdev,
                _effective_n(baseline, path),
                _effective_n(current, path),
                cfg.stdev_zero_floor,
            )
            if ci is not None:
                entry["ci_low"], entry["ci_high"] = ci
            # Whole-distribution separation, symmetric to the mean rule's empirical family: a spread
            # shift pulls the two quantile functions apart even when the mean holds still — and a
            # symmetric variance change is invisible to `score_mean`, which never fires on this
            # field, so the only distribution-distance readings Ogle computes would otherwise be
            # dropped on the exact path they best describe. W1 reads how far the mass had to travel
            # between the two sides' sampled quantile functions (the field's own units, banded by the
            # pooled spread like W2); KS reads how cleanly the two populations pull apart at their
            # point of maximum divergence (bounded [0,1], nonparametric — so it catches a variance
            # move that is really a bimodal split rather than a uniform widening). Both ride on the
            # raw quantiles alone, independent of the stdev magnitude the finding already fired on;
            # pure enrichment, never gates. base_stdev/cur_stdev are guaranteed present and
            # non-degenerate here, so W1 always carries its band when the quantiles exist.
            w1 = _empirical_w1(
                baseline.field_quantiles.get(path),
                current.field_quantiles.get(path),
            )
            if w1 is not None:
                entry["w1_emp"] = w1
                entry["w1_emp_band"] = _w2_band(w1, base_stdev, cur_stdev)
            ks = _empirical_ks(
                baseline.field_quantiles.get(path),
                current.field_quantiles.get(path),
            )
            if ks is not None:
                entry["ks"] = ks
                entry["ks_band"] = _ks_band(ks)
            worsened[path] = entry

    if not worsened:
        return None

    # Multiple-comparison control, symmetric to the mean rule: a stdev finding tests every drifted
    # numeric field at once, so a raw per-field p over-states significance on a wide table. When two
    # or more fields carry a p-value, adjust the family with Benjamini-Hochberg so each entry also
    # reports the false-discovery rate at which it would be called real. A single test needs no
    # correction (q == p), so it is left alone to keep the annotation clean.
    pvals = {p: e["p_value"] for p, e in worsened.items() if "p_value" in e}
    if len(pvals) >= 2:
        for p, qv in _bh_qvalues(pvals).items():
            worsened[p]["q_value"] = qv

    max_shift = max(v["rel_shift"] for v in worsened.values())
    severity = _severity_from_ratio(max_shift, cfg.stdev_rel_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["rel_shift"], reverse=True)

    def _annotate(p: str) -> str:
        base = (
            f"{p} (stdev {worsened[p]['baseline']:g}->{worsened[p]['current']:g}, "
            f"{(worsened[p]['current'] - worsened[p]['baseline']) / abs(worsened[p]['baseline']):+.0%}"
        )
        pval = worsened[p].get("p_value")
        if pval is not None:
            base += f", p={pval:.1g}"
            qval = worsened[p].get("q_value")
            if qval is not None:
                base += f", q={qval:.1g}"
        w1 = worsened[p].get("w1_emp")
        if w1 is not None:
            base += f", W1emp={w1:g} {worsened[p]['w1_emp_band']}"
        ks = worsened[p].get("ks")
        if ks is not None:
            base += f", KS={ks:.2f} {worsened[p]['ks_band']}"
        lo = worsened[p].get("ci_low")
        if lo is not None:
            base += f", 95% CI [{lo:.3g}x, {worsened[p]['ci_high']:.3g}x]"
        return base + ")"

    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.STDEV,
        severity=severity,
        message="numeric spread shifted on " + ", ".join(_annotate(p) for p in fields),
        details={"fields": worsened},
    )


def _breach_sigma(
    base_min: float,
    base_max: float,
    cur_min: float,
    cur_max: float,
    base_stdev: Optional[float],
    zero_floor: float,
) -> Optional[float]:
    """How far the worst envelope excursion sits beyond the historical bound, in baseline sigma.

    The range rule scores a breach as a fraction of the baseline *span* (max - min), which
    self-scales to the field's range but says nothing about how *surprising* the new extreme is
    given the field's spread. A max that pokes 40% past a wide, flat envelope is ordinary; the
    same 40% on a tight, low-variance field is a screaming integrity fault. This rescales the
    worst single-side excursion by the baseline standard deviation — the extreme-value analogue
    of the mean rule's Cohen's d — so an operator reads "3.0σ past" (a genuine new regime) apart
    from "0.2σ past" (rounding at the edge).

    Uses the larger of the two one-sided excursions (a max that shot up or a min that fell), each
    measured beyond its own historical bound, so a two-sided breach reports its worst edge rather
    than a blended number. Unsigned — direction is already legible in the [min, max] annotation.
    Pure enrichment: returns None when the baseline carries no stdev (nothing to scale by) or its
    stdev is below `zero_floor` (a ~constant field, where a sigma scale is undefined and the
    constant-goes-variable case already surfaces via stdev drift).
    """
    if base_stdev is None or base_stdev < zero_floor:
        return None
    breach_above = max(0.0, cur_max - base_max)
    breach_below = max(0.0, base_min - cur_min)
    excursion = max(breach_above, breach_below)
    return excursion / base_stdev


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
            entry = {
                "baseline_min": base_min,
                "baseline_max": base_max,
                "current_min": cur_min,
                "current_max": cur_max,
                "breach": breach,
            }
            # Enrich: how surprising is the worst excursion given the field's own spread? Express
            # the larger one-sided breach in baseline-sigma — the extreme-value analogue of the
            # mean rule's Cohen's d. Never gates; a field without a usable baseline stdev is still
            # flagged on the span-fraction rule, it just carries no sigma.
            sigma = _breach_sigma(
                base_min, base_max, cur_min, cur_max,
                baseline.field_stdevs.get(path), cfg.stdev_zero_floor,
            )
            if sigma is not None:
                entry["breach_sigma"] = sigma
            worsened[path] = entry

    if not worsened:
        return None

    max_breach = max(v["breach"] for v in worsened.values())
    severity = _severity_from_ratio(max_breach, cfg.range_rel_threshold)
    fields = sorted(worsened, key=lambda p: worsened[p]["breach"], reverse=True)

    def _annotate(p: str) -> str:
        e = worsened[p]
        base = (
            f"{p} ([{e['baseline_min']:g}, {e['baseline_max']:g}]"
            f"->[{e['current_min']:g}, {e['current_max']:g}], "
            f"+{e['breach']:.0%} of span"
        )
        sigma = e.get("breach_sigma")
        if sigma is not None:
            base += f", {sigma:.1f}σ past"
        return base + ")"

    return DriftFinding(
        urn=current.urn,
        kind=DriftKind.RANGE,
        severity=severity,
        message="numeric range breached on " + ", ".join(_annotate(p) for p in fields),
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
