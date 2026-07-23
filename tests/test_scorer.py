"""Unit tests for ogle.scorer — drift findings from a baseline/current signature pair.

Fixtures mirror the Task #2 ML layer: `customer_purchase_features` reads the real
showcase `customers` dataset, and `churn_predictor` (deployed via
`churn_predictor_endpoint`, IN_SERVICE) sits downstream — so a serving-path escalation
is the realistic case, not a contrived one.
"""

import pytest

from ogle.scorer import (
    DriftKind,
    ScoreConfig,
    Severity,
    _bh_qvalues,
    _effect_magnitude,
    _mean_shift_z,
    _prob_superiority,
    _spread_shift_z,
    _two_sided_p,
    build_score_config,
    score_dataset,
)
from ogle.signature import build_signature

CUSTOMERS_URN = (
    "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
)


def _sig(**kw):
    kw.setdefault("urn", CUSTOMERS_URN)
    kw.setdefault("schema_fields", [("id", "int"), ("email", "string"), ("region", "string")])
    kw.setdefault("row_count", 10_000)
    return build_signature(**kw)


# ---- no drift ----

def test_identical_signatures_produce_no_findings():
    assert score_dataset(_sig(), _sig()) == []


def test_added_column_is_low_severity_only():
    base = _sig()
    cur = _sig(schema_fields=[("id", "int"), ("email", "string"), ("region", "string"), ("tier", "string")])
    findings = score_dataset(base, cur)
    assert len(findings) == 1
    assert findings[0].kind == DriftKind.SCHEMA
    assert findings[0].severity == Severity.LOW


# ---- schema drift ----

def test_removed_column_is_high_severity():
    base = _sig()
    cur = _sig(schema_fields=[("id", "int"), ("email", "string")])  # dropped region
    (finding,) = score_dataset(base, cur)
    assert finding.kind == DriftKind.SCHEMA
    assert finding.severity == Severity.HIGH
    assert finding.details["removed"] == ["region"]


def test_retyped_column_is_high_severity():
    base = _sig()
    cur = _sig(schema_fields=[("id", "bigint"), ("email", "string"), ("region", "string")])
    (finding,) = score_dataset(base, cur)
    assert finding.kind == DriftKind.SCHEMA
    assert finding.severity == Severity.HIGH
    assert finding.details["retyped"] == ["id"]


# ---- volume drift ----

def test_row_count_collapse_is_high():
    base = _sig(row_count=10_000)
    cur = _sig(row_count=0)
    (finding,) = score_dataset(base, cur)
    assert finding.kind == DriftKind.VOLUME
    assert finding.severity == Severity.HIGH
    assert finding.details["current"] == 0


def test_small_volume_change_under_threshold_ignored():
    base = _sig(row_count=10_000)
    cur = _sig(row_count=11_000)  # +10%, below default 30%
    assert score_dataset(base, cur) == []


def test_large_volume_growth_flagged_and_bands_by_magnitude():
    base = _sig(row_count=10_000)
    # +100% is over 3x the 30% threshold -> HIGH
    (finding,) = score_dataset(base, cur := _sig(row_count=20_000))
    assert finding.kind == DriftKind.VOLUME
    assert finding.severity == Severity.HIGH
    assert finding.details["rel_change"] == pytest.approx(1.0)


def test_missing_profile_skips_volume_scoring():
    base = _sig(row_count=None)
    cur = _sig(row_count=None)
    assert score_dataset(base, cur) == []


# ---- quality drift ----

def test_null_spike_flagged():
    base = _sig(field_null_fractions={"email": 0.01})
    cur = _sig(field_null_fractions={"email": 0.40})
    (finding,) = score_dataset(base, cur)
    assert finding.kind == DriftKind.QUALITY
    assert "email" in finding.message
    assert finding.details["fields"]["email"]["delta"] == pytest.approx(0.39)


def test_null_improvement_not_flagged():
    base = _sig(field_null_fractions={"email": 0.40})
    cur = _sig(field_null_fractions={"email": 0.01})
    assert score_dataset(base, cur) == []


def test_new_field_null_fraction_without_baseline_skipped():
    base = _sig(field_null_fractions={})
    cur = _sig(field_null_fractions={"email": 0.9})
    assert score_dataset(base, cur) == []


# ---- serving escalation ----

def test_serving_path_escalates_severity():
    base = _sig(row_count=10_000)
    cur = _sig(row_count=15_000)  # +50% -> MEDIUM normally
    normal = score_dataset(base, cur, serving=False)[0]
    escalated = score_dataset(base, cur, serving=True)[0]
    assert normal.severity == Severity.MEDIUM
    assert escalated.severity == Severity.HIGH
    assert escalated.details["serving"] is True
    assert escalated.message.endswith("[serving]")


def test_serving_escalation_can_be_disabled_by_config():
    cfg = ScoreConfig(escalate_when_serving=False)
    base = _sig(row_count=10_000)
    cur = _sig(row_count=15_000)
    finding = score_dataset(base, cur, cfg=cfg, serving=True)[0]
    assert finding.severity == Severity.MEDIUM


# ---- combined + ordering ----

def test_multiple_drifts_sorted_most_severe_first():
    base = _sig(row_count=10_000, field_null_fractions={"email": 0.01})
    cur = _sig(
        schema_fields=[("id", "int"), ("email", "string"), ("region", "string"), ("tier", "string")],  # add -> LOW schema
        row_count=10_500,  # +5% -> no volume finding
        field_null_fractions={"email": 0.50},  # -> quality finding
    )
    findings = score_dataset(base, cur)
    kinds = [f.kind for f in findings]
    assert DriftKind.SCHEMA in kinds and DriftKind.QUALITY in kinds
    ranks = [f.severity.rank for f in findings]
    assert ranks == sorted(ranks, reverse=True)


def test_mismatched_urns_raise():
    a = build_signature("urn:a")
    b = build_signature("urn:b")
    with pytest.raises(ValueError):
        score_dataset(a, b)


def test_config_threshold_is_respected():
    cfg = ScoreConfig(volume_rel_threshold=0.05)
    base = _sig(row_count=10_000)
    cur = _sig(row_count=10_800)  # +8%, over the tightened 5% threshold
    (finding,) = score_dataset(base, cur, cfg=cfg)
    assert finding.kind == DriftKind.VOLUME


# ---- build_score_config: the validated CLI/config tuning seam ----------------------
def test_build_config_defaults_match_dataclass():
    cfg = build_score_config()
    assert cfg == ScoreConfig()  # all-None keeps every default


def test_build_config_none_keeps_each_default():
    # A partial override must not disturb the untouched fields.
    cfg = build_score_config(volume_threshold=0.5)
    assert cfg.volume_rel_threshold == 0.5
    assert cfg.null_fraction_abs_threshold == ScoreConfig.null_fraction_abs_threshold
    assert cfg.escalate_when_serving is ScoreConfig.escalate_when_serving


def test_build_config_overrides_all_fields():
    cfg = build_score_config(
        volume_threshold=0.1, null_threshold=0.05, escalate_when_serving=False
    )
    assert cfg.volume_rel_threshold == 0.1
    assert cfg.null_fraction_abs_threshold == 0.05
    assert cfg.escalate_when_serving is False


def test_build_config_loose_threshold_actually_suppresses():
    """A looser volume band must let a change through that the default would flag."""
    base = _sig(row_count=10_000)
    cur = _sig(row_count=13_000)  # +30% exactly at the default edge
    assert score_dataset(base, cur, cfg=build_score_config())  # default flags it
    loose = build_score_config(volume_threshold=0.5)  # ±50% band
    assert score_dataset(base, cur, cfg=loose) == []  # now quiet


@pytest.mark.parametrize("bad", [0, -0.1, -5])
def test_build_config_rejects_nonpositive_volume(bad):
    with pytest.raises(ValueError, match="volume threshold must be > 0"):
        build_score_config(volume_threshold=bad)


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, 2])
def test_build_config_rejects_out_of_range_null(bad):
    with pytest.raises(ValueError, match=r"null threshold must be in \(0, 1\]"):
        build_score_config(null_threshold=bad)


def test_build_config_null_at_boundary_one_is_allowed():
    assert build_score_config(null_threshold=1).null_fraction_abs_threshold == 1.0


def test_build_config_escalation_toggle_changes_serving_severity():
    """Turning escalation off must lower a serving-path finding by one band."""
    base = _sig(row_count=10_000)
    cur = _sig(row_count=5_000)  # -50%: MEDIUM by magnitude, escalates to HIGH on serving
    on = score_dataset(base, cur, cfg=build_score_config(), serving=True)
    off = score_dataset(
        base, cur, cfg=build_score_config(escalate_when_serving=False), serving=True
    )
    assert on[0].severity.rank > off[0].severity.rank


# ---- distribution drift (distinct-value / cardinality collapse) --------------------

def test_cardinality_collapse_flagged():
    """A categorical field stuck on ~one value (unique fraction collapses) is drift."""
    base = _sig(field_unique_fractions={"region": 0.80})
    cur = _sig(field_unique_fractions={"region": 0.05})
    findings = score_dataset(base, cur)
    dist = [f for f in findings if f.kind is DriftKind.DISTRIBUTION]
    assert len(dist) == 1
    assert "region" in dist[0].message
    assert dist[0].details["fields"]["region"]["drop"] == pytest.approx(0.75)


def test_key_uniqueness_loss_flagged():
    """An id column losing uniqueness (1.0 -> 0.5, a fan-out join) is distribution drift."""
    base = _sig(field_unique_fractions={"id": 1.0})
    cur = _sig(field_unique_fractions={"id": 0.5})
    dist = [f for f in score_dataset(base, cur) if f.kind is DriftKind.DISTRIBUTION]
    assert len(dist) == 1


def test_cardinality_rise_not_flagged():
    """More variety is benign — only a *drop* pages."""
    base = _sig(field_unique_fractions={"region": 0.10})
    cur = _sig(field_unique_fractions={"region": 0.90})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.DISTRIBUTION] == []


def test_small_cardinality_drop_below_threshold_quiet():
    base = _sig(field_unique_fractions={"region": 0.50})
    cur = _sig(field_unique_fractions={"region": 0.30})  # drop 0.20 < 0.30 default
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.DISTRIBUTION] == []


def test_new_unique_fraction_without_baseline_skipped():
    base = _sig(field_unique_fractions={})
    cur = _sig(field_unique_fractions={"region": 0.01})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.DISTRIBUTION] == []


def test_distribution_severity_scales_with_drop():
    """A bigger distinct-value collapse earns a higher severity band."""
    base = _sig(field_unique_fractions={"region": 0.95})
    mild = _sig(field_unique_fractions={"region": 0.55})   # drop 0.40 -> ~1.3x -> LOW
    severe = _sig(field_unique_fractions={"region": 0.00})  # drop 0.95 -> >3x -> HIGH
    low = [f for f in score_dataset(base, mild) if f.kind is DriftKind.DISTRIBUTION][0]
    high = [f for f in score_dataset(base, severe) if f.kind is DriftKind.DISTRIBUTION][0]
    assert high.severity.rank > low.severity.rank


def test_distribution_escalates_on_serving_path():
    base = _sig(field_unique_fractions={"region": 0.90})
    cur = _sig(field_unique_fractions={"region": 0.40})  # drop 0.50 -> MEDIUM
    off = [f for f in score_dataset(base, cur) if f.kind is DriftKind.DISTRIBUTION][0]
    on = [
        f for f in score_dataset(base, cur, serving=True)
        if f.kind is DriftKind.DISTRIBUTION
    ][0]
    assert on.severity.rank > off.severity.rank
    assert on.details.get("serving") is True


def test_distribution_reports_worst_field_first():
    base = _sig(field_unique_fractions={"region": 0.90, "tier": 0.90})
    cur = _sig(field_unique_fractions={"region": 0.50, "tier": 0.10})  # tier drops more
    dist = [f for f in score_dataset(base, cur) if f.kind is DriftKind.DISTRIBUTION][0]
    assert dist.message.index("tier") < dist.message.index("region")


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, 2])
def test_build_config_rejects_out_of_range_unique_drop(bad):
    with pytest.raises(ValueError, match=r"unique-drop threshold must be in \(0, 1\]"):
        build_score_config(unique_drop_threshold=bad)


def test_build_config_unique_drop_tuning_suppresses():
    base = _sig(field_unique_fractions={"region": 0.90})
    cur = _sig(field_unique_fractions={"region": 0.50})  # drop 0.40
    assert [f for f in score_dataset(base, cur, cfg=build_score_config())
            if f.kind is DriftKind.DISTRIBUTION]  # default (0.30) flags it
    loose = build_score_config(unique_drop_threshold=0.5)
    assert [f for f in score_dataset(base, cur, cfg=loose)
            if f.kind is DriftKind.DISTRIBUTION] == []  # now quiet


# ---- mean drift (numeric covariate shift) ------------------------------------------

def test_mean_shift_flagged():
    """A numeric feature whose mean moved past the band is covariate drift."""
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 140.0})  # +40% > 25% default
    means = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN]
    assert len(means) == 1
    assert "amount" in means[0].message
    assert means[0].details["fields"]["amount"]["rel_shift"] == pytest.approx(0.40)


def test_mean_shift_flags_both_directions():
    """Unlike distribution's drop-only rule, a mean that *fell* pages too."""
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 50.0})  # -50%
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN]


def test_small_mean_shift_below_threshold_quiet():
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 110.0})  # +10% < 25% default
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN] == []


def test_mean_near_zero_baseline_skipped():
    """A relative shift against a ~0 baseline is undefined — skipped, never guessed."""
    base = _sig(field_means={"delta": 0.0})
    cur = _sig(field_means={"delta": 5.0})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN] == []


def test_new_mean_without_baseline_skipped():
    base = _sig(field_means={})
    cur = _sig(field_means={"amount": 42.0})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN] == []


def test_mean_severity_scales_with_shift():
    base = _sig(field_means={"amount": 100.0})
    mild = _sig(field_means={"amount": 140.0})   # +40% -> ~1.6x band -> MEDIUM-ish
    severe = _sig(field_means={"amount": 500.0})  # +400% -> >3x band -> HIGH
    low = [f for f in score_dataset(base, mild) if f.kind is DriftKind.MEAN][0]
    high = [f for f in score_dataset(base, severe) if f.kind is DriftKind.MEAN][0]
    assert high.severity.rank > low.severity.rank


def test_mean_escalates_on_serving_path():
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 160.0})  # +60%
    off = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    on = [f for f in score_dataset(base, cur, serving=True) if f.kind is DriftKind.MEAN][0]
    assert on.severity.rank > off.severity.rank
    assert on.details.get("serving") is True


def test_mean_reports_worst_field_first():
    base = _sig(field_means={"amount": 100.0, "score": 100.0})
    cur = _sig(field_means={"amount": 130.0, "score": 400.0})  # score moved more
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert m.message.index("score") < m.message.index("amount")


def test_negative_mean_shift_uses_magnitude():
    """A mean crossing sign (e.g. -100 -> 100) is a 200% move, flagged HIGH."""
    base = _sig(field_means={"pnl": -100.0})
    cur = _sig(field_means={"pnl": 100.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert m.details["fields"]["pnl"]["rel_shift"] == pytest.approx(2.0)


# ---- mean drift: Cohen's d effect-size enrichment (first two-sample signal) ---------

def test_mean_finding_carries_effect_size_when_stdevs_present():
    """A flagged mean move is annotated with pooled sigma (Cohen's d) when spread is known."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0})
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0})  # +40 raw
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    # pooled sd = sqrt((100+100)/2) = 10; d = (140-100)/10 = +4.0
    assert m.details["fields"]["amount"]["effect_size"] == pytest.approx(4.0)
    assert "d=+4.0" in m.message


def test_effect_size_is_signed_for_a_falling_mean():
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 20.0})
    cur = _sig(field_means={"amount": 60.0}, field_stdevs={"amount": 20.0})  # -40 raw
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    # pooled sd = 20; d = (60-100)/20 = -2.0
    assert m.details["fields"]["amount"]["effect_size"] == pytest.approx(-2.0)


def test_effect_size_pools_unequal_stdevs():
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 30.0})
    cur = _sig(field_means={"amount": 150.0}, field_stdevs={"amount": 40.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    # pooled sd = sqrt((900+1600)/2) = sqrt(1250) ~= 35.355; d = 50/35.355 ~= 1.414
    assert m.details["fields"]["amount"]["effect_size"] == pytest.approx(50.0 / (1250.0 ** 0.5))


def test_mean_finding_omits_effect_size_without_stdev():
    """No stdev on a side -> nothing to pool -> the move is still flagged, just no d."""
    base = _sig(field_means={"amount": 100.0})  # no stdevs
    cur = _sig(field_means={"amount": 140.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "effect_size" not in m.details["fields"]["amount"]
    assert "d=" not in m.message


def test_effect_size_skipped_when_pooled_spread_is_zero():
    """Both samples ~constant -> a standardized move is undefined -> no d, still flagged."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 0.0})
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 0.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "effect_size" not in m.details["fields"]["amount"]


@pytest.mark.parametrize(
    "d, band",
    [
        (0.0, "negligible"),
        (0.19, "negligible"),
        (0.2, "small"),  # boundary lands in the higher band
        (0.49, "small"),
        (0.5, "medium"),
        (0.79, "medium"),
        (0.8, "large"),
        (4.0, "large"),
    ],
)
def test_effect_magnitude_bands(d, band):
    """Cohen's (1988) cutoffs, boundary-inclusive on the upper band."""
    assert _effect_magnitude(d) == band


def test_effect_magnitude_is_sign_independent():
    """A rise and a fall of equal standardized size read the same magnitude."""
    assert _effect_magnitude(-1.2) == _effect_magnitude(1.2) == "large"
    assert _effect_magnitude(-0.3) == _effect_magnitude(0.3) == "small"


def test_mean_finding_carries_effect_magnitude_band():
    """A large standardized move is labeled 'large' in details and narrative alike."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0})
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0})  # d=+4.0
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert m.details["fields"]["amount"]["effect_magnitude"] == "large"
    assert "d=+4.0 large" in m.message


def test_negligible_effect_labeled_on_a_wide_field():
    """The relative rule can fire while the move is tiny vs the field's own spread."""
    # +30% relative move (past the 0.25 default) but pooled sd ~= 300 -> d ~= 0.1 negligible.
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 300.0})
    cur = _sig(field_means={"amount": 130.0}, field_stdevs={"amount": 300.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert m.details["fields"]["amount"]["effect_magnitude"] == "negligible"
    assert "negligible" in m.message


def test_effect_magnitude_absent_without_effect_size():
    """No stdev -> no d -> no magnitude band either (nothing to label)."""
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 140.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "effect_magnitude" not in m.details["fields"]["amount"]


# ---- mean drift: probability-of-superiority (common-language effect size) ------------

@pytest.mark.parametrize(
    "d, prob",
    [
        (0.0, 0.5),          # coincident distributions -> a coin flip
        (0.8, 0.7141),       # Cohen's "large" -> ~71% (the textbook CLES value)
        (-0.8, 0.2859),      # symmetric: a fall of equal size is 1 - the rise
        (4.0, 0.9977),       # a big positive d saturates toward 1.0
        (-4.0, 0.0023),      # ...and a big negative toward 0.0
    ],
)
def test_prob_superiority_from_cohens_d(d, prob):
    """P(new>old) = 0.5*(1+erf(d/2)) under the normal, equal-spread approximation."""
    assert _prob_superiority(d) == pytest.approx(prob, abs=1e-4)


def test_prob_superiority_is_symmetric_about_a_half():
    """A rise and an equal-magnitude fall are mirror images around 0.5."""
    assert _prob_superiority(1.3) + _prob_superiority(-1.3) == pytest.approx(1.0)


def test_mean_finding_carries_prob_superiority_when_effect_size_present():
    """A flagged move with known spread reports the chance a new row outranks an old one."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0})
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0})  # d=+4.0
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert m.details["fields"]["amount"]["prob_superiority"] == pytest.approx(0.9977, abs=1e-4)
    assert "P(new>old)=100%" in m.message  # 0.9977 rounds to 100% at :.0%


def test_prob_superiority_absent_without_effect_size():
    """No stdev -> no d -> no probability either (defined exactly when d is)."""
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 140.0})
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "prob_superiority" not in m.details["fields"]["amount"]
    assert "P(new>old)" not in m.message


# ---- mean drift: Welch two-sample z-test (significance vs sampling noise) -----------

def test_mean_shift_z_uses_sample_size():
    """The z-statistic scales the raw move by the standard error of the difference."""
    # base/cur mean 100->140, sd 10 both sides, n = 10k rows each.
    # SE = sqrt(100/10000 + 100/10000) = sqrt(0.02); z = 40 / sqrt(0.02).
    z = _mean_shift_z(100.0, 140.0, 10.0, 10.0, 10_000, 10_000, 1e-9)
    assert z == pytest.approx(40.0 / (0.02 ** 0.5))


def test_mean_shift_z_is_signed_for_a_fall():
    """A falling mean yields a negative z, mirroring the direction of the move."""
    z = _mean_shift_z(100.0, 60.0, 10.0, 10.0, 10_000, 10_000, 1e-9)
    assert z < 0


def test_mean_shift_z_shrinks_with_smaller_samples():
    """Same means and spread, fewer rows -> a smaller (less significant) z."""
    big = _mean_shift_z(100.0, 140.0, 10.0, 10.0, 10_000, 10_000, 1e-9)
    small = _mean_shift_z(100.0, 140.0, 10.0, 10.0, 4, 4, 1e-9)
    assert abs(small) < abs(big)


def test_mean_shift_z_none_without_stdev():
    """No spread on a side -> no standard error to build -> no z."""
    assert _mean_shift_z(100.0, 140.0, None, 10.0, 10_000, 10_000, 1e-9) is None


def test_mean_shift_z_none_without_sample_size():
    """No row count -> no denominator -> no z (defined only when n is known)."""
    assert _mean_shift_z(100.0, 140.0, 10.0, 10.0, None, 10_000, 1e-9) is None


def test_mean_shift_z_none_for_degenerate_sample():
    """A one-row sample has no sampling spread -> z is undefined."""
    assert _mean_shift_z(100.0, 140.0, 10.0, 10.0, 1, 10_000, 1e-9) is None


def test_mean_shift_z_none_when_standard_error_vanishes():
    """Both samples ~constant -> SE below the floor -> z undefined (not a divide-by-zero)."""
    assert _mean_shift_z(100.0, 140.0, 0.0, 0.0, 10_000, 10_000, 1e-9) is None


@pytest.mark.parametrize(
    "z, p",
    [
        (0.0, 1.0),        # no shift -> no evidence against the null
        (1.959964, 0.05),  # the textbook two-sided 5% critical value
        (-1.959964, 0.05), # sign-independent: a fall is as surprising as a rise
        (5.656854, 1.5417e-08),
    ],
)
def test_two_sided_p_from_z(z, p):
    """p = erfc(|z|/sqrt(2)): the chance of a move this extreme if nothing changed."""
    assert _two_sided_p(z) == pytest.approx(p, rel=1e-3, abs=1e-12)


def test_two_sided_p_is_sign_independent():
    """A rise and an equal-magnitude fall carry the same p-value."""
    assert _two_sided_p(2.3) == pytest.approx(_two_sided_p(-2.3))


def test_mean_finding_carries_significance_when_samples_known():
    """A flagged move with spread + row counts reports z and a p-value, and annotates p."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0}, row_count=10_000)
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0}, row_count=10_000)
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    entry = m.details["fields"]["amount"]
    assert entry["z_score"] == pytest.approx(40.0 / (0.02 ** 0.5))
    assert entry["p_value"] == pytest.approx(0.0, abs=1e-6)  # ~283 sigma -> vanishing p
    assert "p=" in m.message


def test_significance_reflects_non_null_sample_size():
    """A heavily-null field backs its mean with fewer rows -> a smaller z than a full one."""
    full = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0}, row_count=10_000)
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0}, row_count=10_000)
    sparse = _sig(
        field_means={"amount": 140.0},
        field_stdevs={"amount": 10.0},
        row_count=10_000,
        field_null_fractions={"amount": 0.99},  # only ~100 rows carry a value
    )
    base_sparse = _sig(
        field_means={"amount": 100.0},
        field_stdevs={"amount": 10.0},
        row_count=10_000,
        field_null_fractions={"amount": 0.99},
    )
    z_full = [f for f in score_dataset(base, full) if f.kind is DriftKind.MEAN][0].details["fields"]["amount"]["z_score"]
    z_sparse = [f for f in score_dataset(base_sparse, sparse) if f.kind is DriftKind.MEAN][0].details["fields"]["amount"]["z_score"]
    assert abs(z_sparse) < abs(z_full)


def test_significance_absent_without_stdev():
    """No spread -> no z or p (defined only when a standard error exists)."""
    base = _sig(field_means={"amount": 100.0}, row_count=10_000)
    cur = _sig(field_means={"amount": 140.0}, row_count=10_000)
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "z_score" not in m.details["fields"]["amount"]
    assert "p_value" not in m.details["fields"]["amount"]
    assert "p=" not in m.message


def test_significance_absent_without_row_count():
    """No row count on a side -> no sample size -> no z or p, but the move still flags."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0}, row_count=None)
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0}, row_count=None)
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "z_score" not in m.details["fields"]["amount"]


# ---- Benjamini-Hochberg FDR correction across a mean finding's fields ----

def test_bh_qvalues_leaves_a_single_test_unchanged():
    """With one test there is nothing to correct: q == p (BH factor m/i = 1/1)."""
    assert _bh_qvalues({"a": 0.02}) == pytest.approx({"a": 0.02})


def test_bh_qvalues_empty_input():
    """No p-values -> no q-values (defined only over a non-empty family)."""
    assert _bh_qvalues({}) == {}


def test_bh_qvalues_match_hand_computed_step_up():
    """Three ordered p-values map to the textbook BH-adjusted values, monotone in p."""
    # p sorted: a=0.01 (rank1), b=0.02 (rank2), c=0.03 (rank3), m=3
    # raw: a=0.01*3/1=0.03, b=0.02*3/2=0.03, c=0.03*3/3=0.03 -> all pinned to 0.03 by step-up
    q = _bh_qvalues({"a": 0.01, "c": 0.03, "b": 0.02})
    assert q["a"] == pytest.approx(0.03)
    assert q["b"] == pytest.approx(0.03)
    assert q["c"] == pytest.approx(0.03)


def test_bh_qvalues_push_a_lone_small_p_up_among_noise():
    """One small p among many null fields is inflated toward 1 — the false-discovery guard."""
    pvals = {"real": 0.01, **{f"noise{i}": 0.9 for i in range(19)}}  # 20 tests
    q = _bh_qvalues(pvals)
    assert q["real"] == pytest.approx(0.01 * 20 / 1)  # 0.2 — no longer "significant" at 0.05
    assert all(q[f"noise{i}"] <= 1.0 for i in range(19))


def test_bh_qvalues_are_monotone_and_bounded():
    """q is monotone in the raw p ordering and never exceeds 1."""
    q = _bh_qvalues({"a": 0.001, "b": 0.4, "c": 0.6, "d": 0.95})
    assert q["a"] <= q["b"] <= q["c"] <= q["d"]
    assert all(0.0 <= v <= 1.0 for v in q.values())


def test_mean_finding_carries_qvalue_when_multiple_fields_significant():
    """Two drifted numeric fields with p-values get FDR q-values and a q= annotation."""
    base = _sig(
        field_means={"amount": 100.0, "score": 50.0},
        field_stdevs={"amount": 10.0, "score": 5.0},
        row_count=10_000,
    )
    cur = _sig(
        field_means={"amount": 140.0, "score": 70.0},
        field_stdevs={"amount": 10.0, "score": 5.0},
        row_count=10_000,
    )
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "q_value" in m.details["fields"]["amount"]
    assert "q_value" in m.details["fields"]["score"]
    assert "q=" in m.message


def test_mean_finding_omits_qvalue_for_a_single_field():
    """A lone significant field carries a p but no q — with one test correction is a no-op."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0}, row_count=10_000)
    cur = _sig(field_means={"amount": 140.0}, field_stdevs={"amount": 10.0}, row_count=10_000)
    m = [f for f in score_dataset(base, cur) if f.kind is DriftKind.MEAN][0]
    assert "p_value" in m.details["fields"]["amount"]
    assert "q_value" not in m.details["fields"]["amount"]
    assert "q=" not in m.message


@pytest.mark.parametrize("bad", [0, -0.1, -1])
def test_build_config_rejects_nonpositive_mean_threshold(bad):
    with pytest.raises(ValueError, match=r"mean threshold must be > 0"):
        build_score_config(mean_threshold=bad)


def test_build_config_mean_tuning_suppresses():
    base = _sig(field_means={"amount": 100.0})
    cur = _sig(field_means={"amount": 140.0})  # +40%
    assert [f for f in score_dataset(base, cur, cfg=build_score_config())
            if f.kind is DriftKind.MEAN]  # default (0.25) flags it
    loose = build_score_config(mean_threshold=0.5)
    assert [f for f in score_dataset(base, cur, cfg=loose)
            if f.kind is DriftKind.MEAN] == []  # now quiet


# ---- stdev drift (numeric spread / scale shift) ------------------------------------

def test_stdev_shift_flagged():
    """A numeric feature whose spread moved past the band is scale drift."""
    base = _sig(field_stdevs={"amount": 10.0})
    cur = _sig(field_stdevs={"amount": 14.0})  # +40% > 25% default
    stds = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV]
    assert len(stds) == 1
    assert "amount" in stds[0].message
    assert stds[0].details["fields"]["amount"]["rel_shift"] == pytest.approx(0.40)


def test_stdev_collapse_flagged():
    """A spread collapsing toward 0 (sensor stuck) is scale drift, both directions page."""
    base = _sig(field_stdevs={"reading": 8.0})
    cur = _sig(field_stdevs={"reading": 1.0})  # -87.5%
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV]


def test_small_stdev_shift_below_threshold_quiet():
    base = _sig(field_stdevs={"amount": 10.0})
    cur = _sig(field_stdevs={"amount": 11.0})  # +10% < 25% default
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV] == []


def test_stdev_near_zero_baseline_skipped():
    """A relative shift against a ~0 baseline stdev is undefined — skipped, never guessed."""
    base = _sig(field_stdevs={"const": 0.0})
    cur = _sig(field_stdevs={"const": 5.0})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV] == []


def test_new_stdev_without_baseline_skipped():
    base = _sig(field_stdevs={})
    cur = _sig(field_stdevs={"amount": 4.0})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV] == []


def test_stdev_severity_scales_with_shift():
    base = _sig(field_stdevs={"amount": 10.0})
    mild = _sig(field_stdevs={"amount": 14.0})    # +40% -> ~1.6x band
    severe = _sig(field_stdevs={"amount": 50.0})  # +400% -> >3x band -> HIGH
    low = [f for f in score_dataset(base, mild) if f.kind is DriftKind.STDEV][0]
    high = [f for f in score_dataset(base, severe) if f.kind is DriftKind.STDEV][0]
    assert high.severity.rank > low.severity.rank


def test_stdev_escalates_on_serving_path():
    base = _sig(field_stdevs={"amount": 10.0})
    cur = _sig(field_stdevs={"amount": 16.0})  # +60%
    off = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV][0]
    on = [f for f in score_dataset(base, cur, serving=True) if f.kind is DriftKind.STDEV][0]
    assert on.severity.rank > off.severity.rank
    assert on.details.get("serving") is True


def test_stdev_reports_worst_field_first():
    base = _sig(field_stdevs={"amount": 10.0, "score": 10.0})
    cur = _sig(field_stdevs={"amount": 13.0, "score": 40.0})  # score moved more
    s = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV][0]
    assert s.message.index("score") < s.message.index("amount")


def test_stdev_is_independent_of_mean():
    """A spread change with the mean held constant is caught by stdev alone, not mean."""
    base = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 10.0})
    cur = _sig(field_means={"amount": 100.0}, field_stdevs={"amount": 20.0})  # mean same, spread 2x
    kinds = {f.kind for f in score_dataset(base, cur)}
    assert DriftKind.STDEV in kinds
    assert DriftKind.MEAN not in kinds


# ---- stdev significance: log-SD two-sample z-test ----

def test_spread_shift_z_sign_follows_direction():
    """Spread growing -> positive z; spread collapsing -> negative z (log-ratio sign)."""
    grew = _spread_shift_z(10.0, 20.0, 10_000, 10_000, 1e-9)
    shrank = _spread_shift_z(20.0, 10.0, 10_000, 10_000, 1e-9)
    assert grew is not None and grew > 0
    assert shrank is not None and shrank < 0
    assert grew == pytest.approx(-shrank)  # exact ratio symmetry in log space


def test_spread_shift_z_matches_closed_form():
    """z = ln(s_cur/s_base) / sqrt(1/(2(n_b-1)) + 1/(2(n_c-1)))."""
    import math as _m
    z = _spread_shift_z(10.0, 15.0, 5_000, 8_000, 1e-9)
    se = (1.0 / (2 * 4_999) + 1.0 / (2 * 7_999)) ** 0.5
    assert z == pytest.approx(_m.log(15.0 / 10.0) / se)


def test_spread_shift_z_grows_with_sample_size():
    """The same spread ratio on more rows is more distinguishable from noise -> larger |z|."""
    small = _spread_shift_z(10.0, 15.0, 100, 100, 1e-9)
    big = _spread_shift_z(10.0, 15.0, 1_000_000, 1_000_000, 1e-9)
    assert abs(big) > abs(small)


def test_spread_shift_z_none_for_zero_stdev():
    """A stdev at ~0 -> log undefined -> no z (the collapse-to-constant case is range/stdev's)."""
    assert _spread_shift_z(0.0, 10.0, 10_000, 10_000, 1e-9) is None
    assert _spread_shift_z(10.0, 0.0, 10_000, 10_000, 1e-9) is None


def test_spread_shift_z_none_for_degenerate_or_missing_sample():
    """One-row (or unknown) samples carry no dispersion -> z undefined."""
    assert _spread_shift_z(10.0, 15.0, 1, 10_000, 1e-9) is None
    assert _spread_shift_z(10.0, 15.0, None, 10_000, 1e-9) is None


def test_stdev_finding_carries_significance_when_samples_known():
    """A flagged spread move with row counts reports z + p and annotates p in the message."""
    base = _sig(field_stdevs={"amount": 10.0}, row_count=10_000)
    cur = _sig(field_stdevs={"amount": 20.0}, row_count=10_000)  # +100% > 25%
    s = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV][0]
    entry = s.details["fields"]["amount"]
    assert entry["z_score"] == pytest.approx(_spread_shift_z(10.0, 20.0, 10_000, 10_000, 1e-9))
    assert entry["p_value"] == pytest.approx(0.0, abs=1e-6)  # huge n -> vanishing p
    assert "p=" in s.message


def test_stdev_significance_reflects_non_null_sample_size():
    """A heavily-null field backs its spread with fewer rows -> a smaller |z| than a full one."""
    base_full = _sig(field_stdevs={"amount": 10.0}, row_count=10_000)
    full = _sig(field_stdevs={"amount": 20.0}, row_count=10_000)
    base_sparse = _sig(
        field_stdevs={"amount": 10.0}, row_count=10_000,
        field_null_fractions={"amount": 0.99},  # ~100 rows carry a value
    )
    sparse = _sig(
        field_stdevs={"amount": 20.0}, row_count=10_000,
        field_null_fractions={"amount": 0.99},
    )
    z_full = [f for f in score_dataset(base_full, full) if f.kind is DriftKind.STDEV][0].details["fields"]["amount"]["z_score"]
    z_sparse = [f for f in score_dataset(base_sparse, sparse) if f.kind is DriftKind.STDEV][0].details["fields"]["amount"]["z_score"]
    assert abs(z_sparse) < abs(z_full)


def test_stdev_significance_absent_without_row_count():
    """No row count -> no sample size -> no z or p, but the spread move still flags."""
    base = _sig(field_stdevs={"amount": 10.0}, row_count=None)
    cur = _sig(field_stdevs={"amount": 20.0}, row_count=None)
    s = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV][0]
    assert "z_score" not in s.details["fields"]["amount"]
    assert "p=" not in s.message


def test_stdev_finding_carries_bh_qvalues_across_fields():
    """Two+ drifted fields with p-values get BH q-values, symmetric to the mean rule."""
    base = _sig(field_stdevs={"amount": 10.0, "score": 10.0}, row_count=10_000)
    cur = _sig(field_stdevs={"amount": 20.0, "score": 18.0}, row_count=10_000)
    s = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV][0]
    assert "q_value" in s.details["fields"]["amount"]
    assert "q_value" in s.details["fields"]["score"]
    assert "q=" in s.message


def test_stdev_single_field_has_no_qvalue():
    """One drifted field needs no multiplicity correction — q is left off (q == p)."""
    base = _sig(field_stdevs={"amount": 10.0}, row_count=10_000)
    cur = _sig(field_stdevs={"amount": 20.0}, row_count=10_000)
    s = [f for f in score_dataset(base, cur) if f.kind is DriftKind.STDEV][0]
    assert "q_value" not in s.details["fields"]["amount"]
    assert "q=" not in s.message


@pytest.mark.parametrize("bad", [0, -0.1, -1])
def test_build_config_rejects_nonpositive_stdev_threshold(bad):
    with pytest.raises(ValueError, match=r"stdev threshold must be > 0"):
        build_score_config(stdev_threshold=bad)


def test_build_config_stdev_tuning_suppresses():
    base = _sig(field_stdevs={"amount": 10.0})
    cur = _sig(field_stdevs={"amount": 14.0})  # +40%
    assert [f for f in score_dataset(base, cur, cfg=build_score_config())
            if f.kind is DriftKind.STDEV]  # default (0.25) flags it
    loose = build_score_config(stdev_threshold=0.5)
    assert [f for f in score_dataset(base, cur, cfg=loose)
            if f.kind is DriftKind.STDEV] == []  # now quiet


# ---- range drift (numeric min/max bounds breach) -----------------------------------

def test_range_breach_above_flagged():
    """A max escaping the historical envelope past the band is bounds drift."""
    base = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 100.0})
    cur = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 140.0})  # +40 over a 100 span
    rng = [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE]
    assert len(rng) == 1
    assert "amount" in rng[0].message
    assert rng[0].details["fields"]["amount"]["breach"] == pytest.approx(0.40)


def test_range_breach_below_flagged():
    """A min dropping below the historical floor is bounds drift, both directions page."""
    base = _sig(field_mins={"temp": 10.0}, field_maxes={"temp": 30.0})  # span 20
    cur = _sig(field_mins={"temp": 0.0}, field_maxes={"temp": 30.0})    # -10 below -> 50%
    rng = [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE]
    assert len(rng) == 1
    assert rng[0].details["fields"]["temp"]["breach"] == pytest.approx(0.50)


def test_range_breach_both_ends_sums():
    """Breaches at both ends add — the envelope escaped on two sides."""
    base = _sig(field_mins={"x": 0.0}, field_maxes={"x": 10.0})  # span 10
    cur = _sig(field_mins={"x": -2.0}, field_maxes={"x": 13.0})  # +3 above, +2 below -> 50%
    rng = [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE]
    assert rng[0].details["fields"]["x"]["breach"] == pytest.approx(0.50)


def test_small_range_breach_below_threshold_quiet():
    base = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 100.0})
    cur = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 110.0})  # +10% < 25% default
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE] == []


def test_range_inside_envelope_quiet():
    """Values that stayed within the historical [min, max] are not drift."""
    base = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 100.0})
    cur = _sig(field_mins={"amount": 5.0}, field_maxes={"amount": 95.0})  # tighter, no breach
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE] == []


def test_range_zero_span_baseline_skipped():
    """A relative breach against a ~0 baseline span (constant column) is undefined — skipped."""
    base = _sig(field_mins={"const": 7.0}, field_maxes={"const": 7.0})  # span 0
    cur = _sig(field_mins={"const": 7.0}, field_maxes={"const": 99.0})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE] == []


def test_range_partial_envelope_skipped():
    """A field missing a min or max on either side is not scored (never guessed)."""
    base = _sig(field_maxes={"amount": 100.0})  # no min
    cur = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 200.0})
    assert [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE] == []


def test_range_severity_scales_with_breach():
    base = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 100.0})
    mild = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 140.0})   # +40% of span
    severe = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 300.0}) # +200% -> HIGH
    low = [f for f in score_dataset(base, mild) if f.kind is DriftKind.RANGE][0]
    high = [f for f in score_dataset(base, severe) if f.kind is DriftKind.RANGE][0]
    assert high.severity.rank > low.severity.rank


def test_range_escalates_on_serving_path():
    base = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 100.0})
    cur = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 160.0})  # +60%
    off = [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE][0]
    on = [f for f in score_dataset(base, cur, serving=True) if f.kind is DriftKind.RANGE][0]
    assert on.severity.rank > off.severity.rank
    assert on.details.get("serving") is True


def test_range_reports_worst_field_first():
    # Field tokens f1/f2 don't collide with the message header words or the numeric values.
    base = _sig(field_mins={"f1": 0.0, "f2": 0.0}, field_maxes={"f1": 100.0, "f2": 100.0})
    cur = _sig(field_mins={"f1": 0.0, "f2": 0.0}, field_maxes={"f1": 130.0, "f2": 400.0})  # f2 worse
    r = [f for f in score_dataset(base, cur) if f.kind is DriftKind.RANGE][0]
    assert r.message.index("f2") < r.message.index("f1")


def test_range_is_independent_of_mean_and_stdev():
    """A few outliers breach the envelope while mean and stdev stay put — range alone fires."""
    base = _sig(
        field_means={"amount": 50.0}, field_stdevs={"amount": 10.0},
        field_mins={"amount": 0.0}, field_maxes={"amount": 100.0},
    )
    cur = _sig(
        field_means={"amount": 50.0}, field_stdevs={"amount": 10.0},  # moments held
        field_mins={"amount": 0.0}, field_maxes={"amount": 150.0},    # envelope breached
    )
    kinds = {f.kind for f in score_dataset(base, cur)}
    assert DriftKind.RANGE in kinds
    assert DriftKind.MEAN not in kinds
    assert DriftKind.STDEV not in kinds


@pytest.mark.parametrize("bad", [0, -0.1, -1])
def test_build_config_rejects_nonpositive_range_threshold(bad):
    with pytest.raises(ValueError, match=r"range threshold must be > 0"):
        build_score_config(range_threshold=bad)


def test_build_config_range_tuning_suppresses():
    base = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 100.0})
    cur = _sig(field_mins={"amount": 0.0}, field_maxes={"amount": 140.0})  # +40% of span
    assert [f for f in score_dataset(base, cur, cfg=build_score_config())
            if f.kind is DriftKind.RANGE]  # default (0.25) flags it
    loose = build_score_config(range_threshold=0.5)
    assert [f for f in score_dataset(base, cur, cfg=loose)
            if f.kind is DriftKind.RANGE] == []  # now quiet


# ---- freshness drift (data-staleness SLA) ------------------------------------------

DAY = 86_400.0
# A fixed reference "now" so freshness tests are deterministic (no wall clock).
NOW = 1_800_000_000.0  # some fixed epoch


def _fresh_cfg(max_age_seconds=DAY):
    return build_score_config(freshness_max_age_seconds=max_age_seconds)


def _at(seconds_before_now):
    """An ISO-8601 `computed_at` stamp `seconds_before_now` earlier than NOW."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(NOW - seconds_before_now, tz=timezone.utc).isoformat()


def test_freshness_off_by_default_even_when_ancient():
    """No SLA configured -> freshness never scored, however old the stamp is."""
    base = _sig(computed_at=_at(30 * DAY))
    cur = _sig(computed_at=_at(30 * DAY))
    assert [f for f in score_dataset(base, cur, now=NOW) if f.kind is DriftKind.FRESHNESS] == []


def test_stale_data_flagged_when_sla_set():
    base = _sig(computed_at=_at(3 * DAY))
    cur = _sig(computed_at=_at(3 * DAY))  # 3 days old, SLA 1 day
    fresh = [
        f for f in score_dataset(base, cur, cfg=_fresh_cfg(), now=NOW)
        if f.kind is DriftKind.FRESHNESS
    ]
    assert len(fresh) == 1
    assert fresh[0].details["age_seconds"] == pytest.approx(3 * DAY)
    assert "stale" in fresh[0].message


def test_fresh_data_within_sla_is_quiet():
    cur = _sig(computed_at=_at(6 * 3600))  # 6h old, SLA 24h
    assert [
        f for f in score_dataset(_sig(computed_at=_at(6 * 3600)), cur, cfg=_fresh_cfg(), now=NOW)
        if f.kind is DriftKind.FRESHNESS
    ] == []


def test_freshness_not_scored_without_now():
    cur = _sig(computed_at=_at(10 * DAY))
    assert [
        f for f in score_dataset(_sig(computed_at=_at(10 * DAY)), cur, cfg=_fresh_cfg(), now=None)
        if f.kind is DriftKind.FRESHNESS
    ] == []


def test_freshness_unparseable_stamp_skipped_never_guessed():
    cur = _sig(computed_at="not-a-timestamp")
    assert [
        f for f in score_dataset(_sig(computed_at="not-a-timestamp"), cur, cfg=_fresh_cfg(), now=NOW)
        if f.kind is DriftKind.FRESHNESS
    ] == []


def test_freshness_absent_stamp_skipped():
    cur = _sig(computed_at=None)
    assert [
        f for f in score_dataset(_sig(), cur, cfg=_fresh_cfg(), now=NOW)
        if f.kind is DriftKind.FRESHNESS
    ] == []


def test_freshness_severity_scales_with_age():
    cfg = _fresh_cfg()  # 1-day SLA
    mild = _sig(computed_at=_at(int(1.2 * DAY)))   # ~1.2x -> LOW
    severe = _sig(computed_at=_at(10 * DAY))       # 10x -> HIGH
    low = [f for f in score_dataset(mild, mild, cfg=cfg, now=NOW) if f.kind is DriftKind.FRESHNESS][0]
    high = [f for f in score_dataset(severe, severe, cfg=cfg, now=NOW) if f.kind is DriftKind.FRESHNESS][0]
    assert high.severity.rank > low.severity.rank


def test_freshness_escalates_on_serving_path():
    cur = _sig(computed_at=_at(2 * DAY))  # 2x SLA -> MEDIUM by magnitude
    off = [f for f in score_dataset(cur, cur, cfg=_fresh_cfg(), now=NOW) if f.kind is DriftKind.FRESHNESS][0]
    on = [
        f for f in score_dataset(cur, cur, cfg=_fresh_cfg(), serving=True, now=NOW)
        if f.kind is DriftKind.FRESHNESS
    ][0]
    assert on.severity.rank > off.severity.rank
    assert on.details.get("serving") is True


def test_future_stamp_clamps_to_zero_age_not_flagged():
    """Clock skew (stamp ahead of now) reads age 0, not a negative that trips the SLA."""
    cur = _sig(computed_at=_at(-3600))  # one hour in the future
    assert [
        f for f in score_dataset(cur, cur, cfg=_fresh_cfg(), now=NOW)
        if f.kind is DriftKind.FRESHNESS
    ] == []


def test_stale_but_otherwise_clean_still_pages():
    """The whole point: schema/volume/quality/distribution identical, only the clock moved."""
    base = _sig(computed_at=_at(5 * DAY))
    cur = _sig(computed_at=_at(5 * DAY))  # every other dimension identical to base
    kinds = {f.kind for f in score_dataset(base, cur, cfg=_fresh_cfg(), now=NOW)}
    assert kinds == {DriftKind.FRESHNESS}


@pytest.mark.parametrize("bad", [0, -1, -86_400])
def test_build_config_rejects_nonpositive_freshness(bad):
    with pytest.raises(ValueError, match="freshness max-age must be > 0"):
        build_score_config(freshness_max_age_seconds=bad)


def test_build_config_freshness_none_keeps_off():
    assert build_score_config().freshness_max_age_seconds is None
    assert build_score_config(volume_threshold=0.5).freshness_max_age_seconds is None
