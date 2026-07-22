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
