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
