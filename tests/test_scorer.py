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
