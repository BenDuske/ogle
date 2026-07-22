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
