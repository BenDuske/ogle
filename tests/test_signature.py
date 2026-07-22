"""Unit tests for ogle.signature — the dataset fingerprint."""

import pytest

from ogle.signature import DatasetSignature, SchemaField, build_signature


def test_schema_hash_is_order_independent():
    a = build_signature("urn:x", [("id", "int"), ("name", "string")])
    b = build_signature("urn:x", [("name", "string"), ("id", "int")])
    assert a.schema_hash == b.schema_hash


def test_schema_hash_changes_on_retype():
    a = build_signature("urn:x", [("id", "int")])
    b = build_signature("urn:x", [("id", "bigint")])
    assert a.schema_hash != b.schema_hash


def test_schema_hash_changes_on_add_and_remove():
    base = build_signature("urn:x", [("id", "int")])
    added = build_signature("urn:x", [("id", "int"), ("extra", "string")])
    removed = build_signature("urn:x", [])
    assert base.schema_hash != added.schema_hash
    assert base.schema_hash != removed.schema_hash


def test_empty_schema_has_stable_hash():
    assert build_signature("urn:x").schema_hash == build_signature("urn:y").schema_hash


def test_duplicate_field_paths_collapse_to_last():
    sig = build_signature("urn:x", [("id", "int"), ("id", "bigint")])
    assert len(sig.schema_fields) == 1
    assert sig.schema_fields[0] == SchemaField("id", "bigint")


def test_field_paths_property():
    sig = build_signature("urn:x", [("a", "int"), ("b", "string")])
    assert sig.field_paths == {"a", "b"}


def test_round_trip_to_from_dict():
    sig = build_signature(
        "urn:li:dataset:x",
        [("id", "int"), ("email", "string")],
        row_count=1000,
        field_null_fractions={"email": 0.05},
        computed_at="2026-07-16T00:00:00Z",
    )
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.schema_hash == sig.schema_hash


def test_to_dict_includes_denormalized_hash():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.to_dict()["schema_hash"] == sig.schema_hash


def test_negative_row_count_rejected():
    with pytest.raises(ValueError):
        build_signature("urn:x", row_count=-1)


@pytest.mark.parametrize("bad", [-0.01, 1.5, 2.0])
def test_out_of_range_null_fraction_rejected(bad):
    with pytest.raises(ValueError):
        build_signature("urn:x", field_null_fractions={"f": bad})


def test_row_count_and_nulls_optional():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.row_count is None
    assert sig.field_null_fractions == {}


# ---- field_unique_fractions (distinct-value fraction, for distribution drift) ------

def test_unique_fractions_round_trip():
    sig = build_signature(
        "urn:li:dataset:x",
        [("id", "int"), ("region", "string")],
        row_count=1000,
        field_null_fractions={"region": 0.01},
        field_unique_fractions={"id": 1.0, "region": 0.4},
        computed_at="2026-07-22T00:00:00Z",
    )
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.field_unique_fractions == {"id": 1.0, "region": 0.4}


def test_unique_fractions_default_empty():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.field_unique_fractions == {}


@pytest.mark.parametrize("bad", [-0.01, 1.5, 2.0])
def test_out_of_range_unique_fraction_rejected(bad):
    with pytest.raises(ValueError, match="unique fraction"):
        build_signature("urn:x", field_unique_fractions={"f": bad})


def test_from_dict_without_unique_fractions_is_backward_compatible():
    """A baseline persisted before this field existed must still load (empty map)."""
    legacy = {
        "urn": "urn:x",
        "schema_fields": [["id", "int"]],
        "row_count": 5,
        "field_null_fractions": {"id": 0.0},
    }
    restored = DatasetSignature.from_dict(legacy)
    assert restored.field_unique_fractions == {}


# ---- field_means (numeric mean, for covariate/mean drift) --------------------------

def test_means_round_trip():
    sig = build_signature(
        "urn:li:dataset:x",
        [("id", "int"), ("amount", "double")],
        row_count=1000,
        field_means={"amount": 42.5, "id": -3.0},
        computed_at="2026-07-22T00:00:00Z",
    )
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.field_means == {"amount": 42.5, "id": -3.0}


def test_means_default_empty():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.field_means == {}


def test_means_allow_negative_and_large():
    """A mean is unbounded — negatives and big magnitudes are valid, unlike fractions."""
    sig = build_signature("urn:x", field_means={"pnl": -1_000_000.0, "rate": 12345.6})
    assert sig.field_means == {"pnl": -1_000_000.0, "rate": 12345.6}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_mean_rejected(bad):
    with pytest.raises(ValueError, match="mean.*finite"):
        build_signature("urn:x", field_means={"f": bad})


def test_from_dict_without_means_is_backward_compatible():
    """A baseline persisted before mean drift existed must still load (empty map)."""
    legacy = {
        "urn": "urn:x",
        "schema_fields": [["id", "int"]],
        "row_count": 5,
        "field_null_fractions": {"id": 0.0},
        "field_unique_fractions": {"id": 1.0},
    }
    restored = DatasetSignature.from_dict(legacy)
    assert restored.field_means == {}


# ---- field_stdevs (numeric standard deviation, for spread/scale drift) -------------

def test_stdevs_round_trip():
    sig = build_signature(
        "urn:li:dataset:x",
        [("id", "int"), ("amount", "double")],
        row_count=1000,
        field_stdevs={"amount": 12.5, "id": 0.0},
        computed_at="2026-07-22T00:00:00Z",
    )
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.field_stdevs == {"amount": 12.5, "id": 0.0}


def test_stdevs_default_empty():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.field_stdevs == {}


def test_stdevs_allow_zero_and_large():
    """A stdev is non-negative and unbounded above — 0 (a constant column) and big are valid."""
    sig = build_signature("urn:x", field_stdevs={"const": 0.0, "spread": 987654.3})
    assert sig.field_stdevs == {"const": 0.0, "spread": 987654.3}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_stdev_rejected(bad):
    with pytest.raises(ValueError, match="stdev.*finite"):
        build_signature("urn:x", field_stdevs={"f": bad})


def test_negative_stdev_rejected():
    """Unlike a mean, a standard deviation below zero is nonsense and is rejected."""
    with pytest.raises(ValueError, match="stdev.*>= 0"):
        build_signature("urn:x", field_stdevs={"f": -1.0})


def test_from_dict_without_stdevs_is_backward_compatible():
    """A baseline persisted before spread drift existed must still load (empty map)."""
    legacy = {
        "urn": "urn:x",
        "schema_fields": [["id", "int"]],
        "row_count": 5,
        "field_null_fractions": {"id": 0.0},
        "field_unique_fractions": {"id": 1.0},
        "field_means": {"id": 3.0},
    }
    restored = DatasetSignature.from_dict(legacy)
    assert restored.field_stdevs == {}


# ---- field_mins / field_maxes (numeric bounds, for range/envelope drift) ------------

def test_mins_maxes_round_trip():
    sig = build_signature(
        "urn:li:dataset:x",
        [("id", "int"), ("amount", "double")],
        row_count=1000,
        field_mins={"amount": -5.0, "id": 1.0},
        field_maxes={"amount": 999.5, "id": 1000.0},
        computed_at="2026-07-22T00:00:00Z",
    )
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.field_mins == {"amount": -5.0, "id": 1.0}
    assert restored.field_maxes == {"amount": 999.5, "id": 1000.0}


def test_mins_maxes_default_empty():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.field_mins == {}
    assert sig.field_maxes == {}


def test_mins_maxes_allow_signed_and_large():
    """A min/max is a signed, unbounded real — negatives and huge values are valid."""
    sig = build_signature(
        "urn:x", field_mins={"a": -1e9}, field_maxes={"a": 1e9}
    )
    assert sig.field_mins == {"a": -1e9}
    assert sig.field_maxes == {"a": 1e9}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_min_rejected(bad):
    with pytest.raises(ValueError, match="min.*finite"):
        build_signature("urn:x", field_mins={"f": bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_max_rejected(bad):
    with pytest.raises(ValueError, match="max.*finite"):
        build_signature("urn:x", field_maxes={"f": bad})


def test_inverted_envelope_rejected():
    """A field whose min exceeds its max is a nonsense envelope and is rejected."""
    with pytest.raises(ValueError, match=r"min for 'f'.*<= max"):
        build_signature("urn:x", field_mins={"f": 10.0}, field_maxes={"f": 5.0})


def test_equal_min_max_allowed():
    """A constant column (min == max) is a valid, degenerate envelope."""
    sig = build_signature("urn:x", field_mins={"f": 7.0}, field_maxes={"f": 7.0})
    assert sig.field_mins == {"f": 7.0}
    assert sig.field_maxes == {"f": 7.0}


def test_from_dict_without_mins_maxes_is_backward_compatible():
    """A baseline persisted before range drift existed must still load (empty maps)."""
    legacy = {
        "urn": "urn:x",
        "schema_fields": [["id", "int"]],
        "row_count": 5,
        "field_means": {"id": 3.0},
        "field_stdevs": {"id": 1.0},
    }
    restored = DatasetSignature.from_dict(legacy)
    assert restored.field_mins == {}
    assert restored.field_maxes == {}
