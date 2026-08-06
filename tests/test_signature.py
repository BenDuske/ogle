"""Unit tests for ogle.signature — the dataset fingerprint."""

import json

import pytest

from ogle.signature import (
    DatasetSignature,
    SchemaField,
    build_signature,
    parse_iso_epoch,
)


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


def test_quantiles_round_trip():
    """Quantile sets round-trip through to_dict/from_dict as sorted (p, v) tuples."""
    sig = build_signature(
        "urn:li:dataset:x",
        [("amount", "double")],
        row_count=1000,
        # deliberately unsorted on input — build_signature must sort by level
        field_quantiles={"amount": [(0.75, 30.0), (0.25, 10.0), (0.5, 20.0)]},
        computed_at="2026-07-24T00:00:00Z",
    )
    assert sig.field_quantiles == {"amount": ((0.25, 10.0), (0.5, 20.0), (0.75, 30.0))}
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig
    assert restored.field_quantiles == {"amount": ((0.25, 10.0), (0.5, 20.0), (0.75, 30.0))}


def test_quantiles_do_not_affect_schema_hash():
    """The schema hash is over (path, type) only — quantiles must never perturb it."""
    plain = build_signature("urn:x", [("a", "int")])
    withq = build_signature(
        "urn:x", [("a", "int")], field_quantiles={"a": [(0.5, 1.0), (0.9, 2.0)]}
    )
    assert plain.schema_hash == withq.schema_hash


def test_quantiles_default_empty():
    sig = build_signature("urn:x", [("id", "int")])
    assert sig.field_quantiles == {}


def test_quantiles_thin_set_dropped():
    """A field with fewer than two points can't describe a distribution — it is dropped."""
    sig = build_signature("urn:x", field_quantiles={"a": [(0.5, 1.0)]})
    assert sig.field_quantiles == {}


def test_quantiles_level_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"quantile level for 'f'.*\[0,1\]"):
        build_signature("urn:x", field_quantiles={"f": [(0.5, 1.0), (1.5, 2.0)]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_quantiles_non_finite_value_rejected(bad):
    with pytest.raises(ValueError, match="quantile value.*finite"):
        build_signature("urn:x", field_quantiles={"f": [(0.25, 1.0), (0.75, bad)]})


@pytest.mark.parametrize("bad", [True, False])
def test_quantiles_bool_level_rejected(bad):
    """A bool quantile level must be rejected, not laundered via float(True/False).

    `float(True) == 1.0` / `float(False) == 0.0` would forge a valid-looking probability
    level the empirical scorer trusts — the exact silent-misread class every sibling numeric
    guard (mean/stdev/min/max/fractions/row_count) already rejects. Quantiles was the last
    carrier still coercing a bare `float(...)`; it must reject a bool like the moments do.
    """
    with pytest.raises(ValueError, match="quantile level.*finite"):
        build_signature("urn:x", field_quantiles={"f": [(bad, 1.0), (0.75, 2.0)]})


@pytest.mark.parametrize("bad", [True, False])
def test_quantiles_bool_value_rejected(bad):
    """A bool quantile value must be rejected, not laundered to 1.0/0.0 via float()."""
    with pytest.raises(ValueError, match="quantile value.*finite"):
        build_signature("urn:x", field_quantiles={"f": [(0.25, 1.0), (0.75, bad)]})


@pytest.mark.parametrize("bad", [(0.25, 1.0, 9.0), (0.25,), "ab", 0.5, None])
def test_quantiles_malformed_pair_rejected_as_valueerror(bad):
    """A wrong-arity / non-array quantile pair must raise ValueError, not IndexError/TypeError.

    A truncated `[0.25]` or a scalar would raise IndexError/TypeError from `pair[1]`, and
    IndexError falls OUTSIDE `BaselineStore.load`'s (ValueError, KeyError, TypeError) recovery
    net — a scheduled `ogle check` would crash-loop on exactly the corrupt file the net exists
    to quarantine. Reject up front with the ValueError contract so load() quarantines instead.
    """
    with pytest.raises(ValueError, match="quantile pair for 'f'.*\\[level, value\\]"):
        build_signature("urn:x", field_quantiles={"f": [(0.25, 5.0), bad]})


def test_quantiles_malformed_pair_survives_store_load(tmp_path):
    """End-to-end: a baseline with a truncated quantile pair QUARANTINES, never crash-loops.

    Proves the ValueError from the arity guard is caught by `BaselineStore.load`'s recovery
    net (which an IndexError would have escaped), so an operator's scheduled check degrades
    loudly (re-baseline) instead of going blind.
    """
    from ogle.store import BaselineStore, STORE_VERSION

    p = tmp_path / "baselines.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {
                    "urn:x": {
                        "urn": "urn:x",
                        "schema_fields": [["amount", "double"]],
                        "field_quantiles": {"amount": [[0.25, 5.0], [0.75]]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.recovered_from_corruption is True
    assert store.get_baseline("urn:x") is None


def test_quantiles_duplicate_level_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_signature("urn:x", field_quantiles={"f": [(0.5, 1.0), (0.5, 2.0)]})


def test_quantiles_backwards_value_rejected():
    """A quantile function cannot run backwards: Q(0.75) < Q(0.25) is nonsense."""
    with pytest.raises(ValueError, match="non-decreasing"):
        build_signature("urn:x", field_quantiles={"f": [(0.25, 5.0), (0.75, 1.0)]})


def test_from_dict_without_quantiles_is_backward_compatible():
    """A baseline persisted before empirical quantiles existed must still load (empty map)."""
    legacy = {
        "urn": "urn:x",
        "schema_fields": [["id", "int"]],
        "row_count": 5,
        "field_means": {"id": 3.0},
    }
    restored = DatasetSignature.from_dict(legacy)
    assert restored.field_quantiles == {}


def test_from_dict_resorts_unordered_persisted_quantiles():
    """A hand-edited/corrupt baseline with out-of-order levels loads sorted, not as-written.

    The empirical scorers trust every quantile set to be sorted by level (`_quantile_at` and the
    Wasserstein helpers that read it). `from_dict` must uphold that invariant on the load path just
    as `build_signature` does on the build path, or a reordered persisted set would silently poison
    the earth-mover integral.
    """
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["amount", "double"]],
        "field_quantiles": {"amount": [[0.75, 30.0], [0.25, 10.0], [0.5, 20.0]]},
    }
    restored = DatasetSignature.from_dict(persisted)
    assert restored.field_quantiles == {"amount": ((0.25, 10.0), (0.5, 20.0), (0.75, 30.0))}


def test_from_dict_rejects_backwards_persisted_quantiles():
    """A persisted quantile function that runs backwards is nonsense — load must raise, matching
    `build_signature`, so `BaselineStore.load` can quarantine the file instead of scoring garbage."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["f", "double"]],
        "field_quantiles": {"f": [[0.25, 5.0], [0.75, 1.0]]},
    }
    with pytest.raises(ValueError, match="non-decreasing"):
        DatasetSignature.from_dict(persisted)


def test_from_dict_drops_thin_persisted_quantile_set():
    """A persisted single-point set can't describe a distribution — dropped on load, as on build."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["f", "double"]],
        "field_quantiles": {"f": [[0.5, 1.0]]},
    }
    restored = DatasetSignature.from_dict(persisted)
    assert restored.field_quantiles == {}


# ---- Load-path enforcement of the SCALAR invariants (not just quantiles) ---------------------
# `from_dict` delegates to `build_signature`, so every per-field invariant the scorers trust is
# re-checked on load — a poisoned baseline degrades LOUDLY (ValueError -> BaselineStore.load
# quarantines + re-baselines) instead of feeding a NaN/negative/inverted value into a scorer,
# where it would compare false against every threshold and SILENTLY miss real drift.


def test_from_dict_full_scalar_signature_round_trips():
    """A clean signature carrying every scalar family survives to_dict/from_dict unchanged — the
    strengthened load path must not reject legitimately-built data (validators idempotent on it)."""
    sig = build_signature(
        "urn:x",
        [("amount", "double")],
        row_count=1000,
        field_null_fractions={"amount": 0.1},
        field_unique_fractions={"amount": 0.9},
        field_means={"amount": 42.5},
        field_stdevs={"amount": 3.0},
        field_mins={"amount": -5.0},
        field_maxes={"amount": 88.0},
    )
    restored = DatasetSignature.from_dict(sig.to_dict())
    assert restored == sig


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_from_dict_rejects_non_finite_persisted_mean(bad):
    """A NaN/inf mean in a hand-edited baseline would make the mean-drift delta NaN, which
    compares false against the threshold — real covariate shift silently missed. Load must raise."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["amount", "double"]],
        "field_means": {"amount": bad},
    }
    with pytest.raises(ValueError, match="mean"):
        DatasetSignature.from_dict(persisted)


def test_from_dict_rejects_negative_persisted_stdev():
    """A standard deviation below zero is nonsense a dispersion can't take — load rejects it just
    as the build path does, rather than poison the relative-spread math in the scorer."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["amount", "double"]],
        "field_stdevs": {"amount": -1.0},
    }
    with pytest.raises(ValueError, match="stdev"):
        DatasetSignature.from_dict(persisted)


def test_from_dict_rejects_inverted_persisted_envelope():
    """A persisted min > max is an inverted envelope: the baseline span goes negative and the
    range-breach math scores garbage. Load must reject it, matching `build_signature`."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["amount", "double"]],
        "field_mins": {"amount": 10.0},
        "field_maxes": {"amount": 1.0},
    }
    with pytest.raises(ValueError, match="must be <= max"):
        DatasetSignature.from_dict(persisted)


@pytest.mark.parametrize("frac", [-0.1, 1.5])
def test_from_dict_rejects_out_of_range_persisted_null_fraction(frac):
    """A null fraction outside [0,1] is impossible for a proportion — load rejects it on the way
    in rather than let a >1 or negative fraction distort the quality-drift score."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["amount", "double"]],
        "field_null_fractions": {"amount": frac},
    }
    with pytest.raises(ValueError, match="null fraction"):
        DatasetSignature.from_dict(persisted)


def test_from_dict_rejects_negative_persisted_row_count():
    """A negative row count can't describe a table — load raises, matching the build path, so a
    corrupt volume baseline is quarantined instead of inverting the volume-drift ratio."""
    persisted = {
        "urn": "urn:x",
        "schema_fields": [["amount", "double"]],
        "row_count": -5,
    }
    with pytest.raises(ValueError, match="row_count"):
        DatasetSignature.from_dict(persisted)


@pytest.mark.parametrize("bad_elem", ["ab", {"foo": 1, "bar": 2}])
def test_from_dict_rejects_char_splittable_schema_field(bad_elem):
    """A length-2 string / 2-key dict schema_fields element must be REJECTED, not char-split.

    `tuple("ab")` -> `('a','b')` and `tuple({"foo":1,"bar":2})` -> `('foo','bar')` both slip
    through the builder's `(path, native_type)` unpack silently and forge a bogus SchemaField,
    poisoning the schema hash so real schema drift is quietly mis-scored. (A 3+-char string or
    other-arity dict already raises an unpack error caught by load — only the length-2 case
    misreads silently.) Load must raise so BaselineStore.load quarantines the foreign file."""
    persisted = {"urn": "urn:x", "schema_fields": [bad_elem]}
    with pytest.raises(ValueError, match="schema_fields entries must be"):
        DatasetSignature.from_dict(persisted)


def test_from_dict_accepts_both_list_and_tuple_schema_field_pairs():
    """The guard rejects only non-array elements — genuine list AND tuple pairs still load."""
    persisted = {"urn": "urn:x", "schema_fields": [["a", "string"], ("n", "int")]}
    restored = DatasetSignature.from_dict(persisted)
    assert [(f.path, f.native_type) for f in restored.schema_fields] == [
        ("a", "string"),
        ("n", "int"),
    ]


# ---- non-dict baseline VALUE: the seen_incidents-record guard's twin, on the baselines side ------
# BaselineStore.from_dict proves the `baselines` SECTION is a dict, but a per-URN VALUE can still be
# a valid-JSON-but-non-object (a list/scalar from a truncated/hand-mangled file that still parses AND
# carries the right version): {"baselines": {"urn:x": [1,2]}}. That value reaches `data.get(...)` in
# DatasetSignature.from_dict and raises AttributeError — OUTSIDE BaselineStore.load's (ValueError,
# KeyError, TypeError) net → crash-loop on exactly the file the net exists to quarantine.
# _IncidentRecord.from_dict already guards this on the seen_incidents side; the baselines side did not.

@pytest.mark.parametrize("bad", [[1, 2], "urn", 5, 0.5, None, True])
def test_from_dict_rejects_non_dict_payload(bad):
    """A baseline persisted as a JSON array/scalar/null must raise ValueError (caught by load),
    not AttributeError (which escapes load's recovery net and crash-loops the scheduled check)."""
    with pytest.raises(ValueError, match="dataset signature must be a JSON object"):
        DatasetSignature.from_dict(bad)


def test_from_dict_non_dict_baseline_survives_store_load(tmp_path):
    """End-to-end: a non-dict baseline VALUE QUARANTINES, never crash-loops.

    Regression guard for the AttributeError hole on the baselines side — a list/scalar value
    reaching `data.get(...)` escaped BaselineStore.load's (ValueError, KeyError, TypeError) net.
    The from_dict guard converts it to a caught ValueError so the scheduled check re-baselines
    loudly instead of going blind to drift on exactly the corrupt file the net exists for."""
    from ogle.store import BaselineStore, STORE_VERSION

    p = tmp_path / "baselines.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {"urn:x": [1, 2]},  # value is a list, not a signature object
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.recovered_from_corruption is True
    assert store.get_baseline("urn:x") is None


# ---- non-dict per-field maps: a JSON array where a {path: value} object belongs ----------------
# A hand-edited/foreign store can carry a LIST where a per-field map should be an object, and the
# two map families fail differently — both wrong. The dict(...)-fed maps SILENTLY coerce a list of
# [k, v] pairs into a mapping (dict([["a",0.5]]) == {"a":0.5}), forging readings the scorer trusts;
# field_quantiles is worse — `_clean_quantiles` calls `.items()`, and a list has none, raising
# AttributeError OUTSIDE BaselineStore.load's (ValueError, KeyError, TypeError) net → crash-loop on
# exactly the file the net exists to quarantine. `from_dict` must reject a present non-dict map.

@pytest.mark.parametrize(
    "map_key",
    [
        "field_null_fractions",
        "field_unique_fractions",
        "field_means",
        "field_stdevs",
        "field_mins",
        "field_maxes",
        "field_quantiles",
    ],
)
@pytest.mark.parametrize("bad_val", [[["a", 0.5]], [], "abc", 0.5])
def test_from_dict_rejects_non_dict_per_field_map(map_key, bad_val):
    """A per-field map persisted as a JSON array/scalar must raise ValueError at load, so
    BaselineStore.load quarantines the foreign file instead of silently coercing it (or, for
    field_quantiles, crash-looping on an AttributeError the recovery net never catches)."""
    persisted = {"urn": "urn:x", "schema_fields": [["a", "int"]], map_key: bad_val}
    with pytest.raises(ValueError, match=f"{map_key} must be a"):
        DatasetSignature.from_dict(persisted)


def test_from_dict_non_dict_quantiles_survives_store_load(tmp_path):
    """End-to-end: field_quantiles as a JSON array QUARANTINES, never crash-loops.

    Regression guard for the AttributeError hole — `_clean_quantiles` calling `.items()` on a
    list escaped BaselineStore.load's (ValueError, KeyError, TypeError) net. The from_dict guard
    converts it to a caught ValueError so the operator's scheduled check re-baselines loudly."""
    from ogle.store import BaselineStore, STORE_VERSION

    p = tmp_path / "baselines.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {
                    "urn:x": {
                        "urn": "urn:x",
                        "schema_fields": [["amount", "double"]],
                        "field_quantiles": [["amount", [[0.25, 5.0], [0.75, 9.0]]]],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.recovered_from_corruption is True
    assert store.get_baseline("urn:x") is None


# ---- non-string schema-field path/type: the moment guards' schema-hash twin --------------------
# A [path, type] pair whose halves are the right ARITY but the wrong TYPE (a bare number/null/nested
# array in a hand-edited or foreign store) is invisible to both the char-split guard above (it IS a
# 2-array) and the numeric guards below. It corrupts the schema hash two ways: a UNIFORM non-string
# type builds silently but hashes DIFFERENTLY from the real string type (schema drift mis-scores —
# the cardinal sin), and a MIXED set (int path beside str path) DEFERS a `TypeError` to
# `schema_hash`'s `sorted()` — a crash in the scorer, not a quarantine at load. Reject at build/load.

@pytest.mark.parametrize(
    "schema_fields",
    [
        [("col", None)],          # uniform non-string TYPE -> silent mis-hash
        [("col", 123)],           # uniform non-string TYPE (number)
        [(123, "double")],        # non-string PATH
        [(True, "double")],       # bool path (isinstance(True, str) is False)
        [("col", True)],          # bool type
        [(1, "t"), ("col", "t")],  # MIXED path types -> deferred sorted() TypeError
        [(["nested"], "t")],      # non-string (unhashable) path
    ],
)
def test_non_string_schema_field_rejected(schema_fields):
    """A non-str path or native_type must raise at build so a corrupt store quarantines LOUDLY
    instead of silently mis-hashing the schema or deferring a TypeError into the scorer."""
    with pytest.raises(ValueError, match="schema field must be"):
        build_signature("urn:x", schema_fields=schema_fields)


@pytest.mark.parametrize(
    "bad_pair",
    [["col", None], ["col", 123], [123, "double"], [True, "double"], ["col", True]],
)
def test_from_dict_rejects_non_string_schema_field(bad_pair):
    """The persisted-store path: a non-string half raises on load so BaselineStore quarantines."""
    persisted = {"urn": "urn:x", "schema_fields": [bad_pair]}
    with pytest.raises(ValueError, match="schema field must be"):
        DatasetSignature.from_dict(persisted)


def test_string_schema_field_still_builds_and_hashes():
    """The guard must not disturb a legitimately-typed schema — it still builds AND its
    hash is reachable (the deferred-crash path the mixed-type case used to take)."""
    sig = build_signature("urn:x", schema_fields=[("amount", "double"), ("id", "bigint")])
    assert [(f.path, f.native_type) for f in sig.schema_fields] == [
        ("amount", "double"),
        ("id", "bigint"),
    ]
    assert isinstance(sig.schema_hash, str) and len(sig.schema_hash) == 64


# ---- bool / non-number laundering of persisted numeric scalars -------------------------------
# `float(True) == 1.0`, so a bare JSON `true` in a hand-edited/foreign store slips past the
# NaN/inf finiteness checks as a real 1.0 (or 0.0 for `false`) and SILENTLY poisons its scorer —
# the exact class the store's timestamp path closed with `_finite_epoch` (9df2132). A str/None in
# a finite-only *moment* map (mean/stdev/min/max) is worse: it trips no NaN/inf check either, so it
# passes build and only crashes later in the scorer's arithmetic — a *deferred* failure on a future
# `ogle check`. Both must be rejected at build/load so a corrupt store quarantines LOUDLY instead.

@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("mean", {"field_means": {"f": True}}),
        ("stdev", {"field_stdevs": {"f": True}}),
        ("min", {"field_mins": {"f": True}}),
        ("max", {"field_maxes": {"f": True}}),
    ],
)
def test_bool_moment_rejected(kind, kwargs):
    """A bool in a moment map launders as 1.0 past the finiteness check — reject the type."""
    with pytest.raises(ValueError, match=f"{kind}.*finite"):
        build_signature("urn:x", **kwargs)


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("mean", {"field_means": {"f": "1.5"}}),
        ("stdev", {"field_stdevs": {"f": "1.5"}}),
        ("min", {"field_mins": {"f": "1.5"}}),
        ("max", {"field_maxes": {"f": "1.5"}}),
    ],
)
def test_non_number_moment_rejected(kind, kwargs):
    """A str moment trips no NaN/inf check, so it used to pass build and defer a crash to the
    scorer. The type guard rejects it at build so the file quarantines at load instead."""
    with pytest.raises(ValueError, match=f"{kind}.*finite"):
        build_signature("urn:x", **kwargs)


@pytest.mark.parametrize(
    "match,kwargs",
    [
        ("null fraction", {"field_null_fractions": {"f": True}}),
        ("unique fraction", {"field_unique_fractions": {"f": True}}),
    ],
)
def test_bool_fraction_rejected(match, kwargs):
    """`True` is 1.0 and 1.0 IS in [0,1], so a bool fraction passes the range check — a bogus
    "100% null/unique" reading. Reject the bool type before it reaches the scorer."""
    with pytest.raises(ValueError, match=match):
        build_signature("urn:x", **kwargs)


def test_bool_row_count_rejected():
    """`True < 0` is False, so a bool row_count slips through as a real 1-row count. Reject it."""
    with pytest.raises(ValueError, match="row_count"):
        build_signature("urn:x", row_count=True)


@pytest.mark.parametrize(
    "match,field,payload",
    [
        ("mean.*finite", "field_means", {"amount": True}),
        ("stdev.*finite", "field_stdevs", {"amount": True}),
        ("min.*finite", "field_mins", {"amount": True}),
        ("max.*finite", "field_maxes", {"amount": True}),
        ("null fraction", "field_null_fractions", {"amount": True}),
        ("unique fraction", "field_unique_fractions", {"amount": True}),
    ],
)
def test_from_dict_rejects_bool_persisted_scalar(match, field, payload):
    """A `true` persisted in any numeric map must raise on load so BaselineStore quarantines the
    file and re-baselines, rather than score a laundered 1.0 as if it were real data."""
    persisted = {"urn": "urn:x", "schema_fields": [["amount", "double"]], field: payload}
    with pytest.raises(ValueError, match=match):
        DatasetSignature.from_dict(persisted)


def test_from_dict_rejects_bool_persisted_row_count():
    persisted = {"urn": "urn:x", "schema_fields": [["amount", "double"]], "row_count": True}
    with pytest.raises(ValueError, match="row_count"):
        DatasetSignature.from_dict(persisted)


def test_int_moment_and_row_count_still_load():
    """The bool guard must not reject a genuine int — a JSON integer mean/count is legitimate
    (bool is an int SUBTYPE, so the guard keys on `isinstance(bool)`, not int-ness)."""
    sig = build_signature(
        "urn:x", row_count=5, field_means={"amount": 42}, field_mins={"amount": 0}
    )
    assert sig.row_count == 5
    assert sig.field_means == {"amount": 42}
    assert sig.field_mins == {"amount": 0}


# ---- parse_iso_epoch: the single clock-free reader behind staleness + the freshness dimension ----

# Oracle epochs computed the tz-explicit way, so these assertions hold on any host timezone.
from datetime import datetime, timezone

_UTC_MIDNIGHT_2026_07_16 = datetime(2026, 7, 16, 0, 0, 0, tzinfo=timezone.utc).timestamp()


def test_parse_iso_epoch_reads_trailing_z_as_utc():
    assert parse_iso_epoch("2026-07-16T00:00:00Z") == _UTC_MIDNIGHT_2026_07_16


def test_parse_iso_epoch_reads_lowercase_z_as_utc():
    """DataHub stamps are usually upper-case Z, but the normalization is case-insensitive."""
    assert parse_iso_epoch("2026-07-16T00:00:00z") == _UTC_MIDNIGHT_2026_07_16


def test_parse_iso_epoch_explicit_utc_offset_matches_z():
    assert parse_iso_epoch("2026-07-16T00:00:00+00:00") == _UTC_MIDNIGHT_2026_07_16


def test_parse_iso_epoch_naive_stamp_is_assumed_utc():
    """A stamp with no offset must be read as UTC, not the host's local time — otherwise the
    freshness SLA and the CLI's capture-age would drift by the host's UTC offset. Fault-injection:
    dropping the tzinfo=UTC coercion makes `.timestamp()` read the host's local zone, so this
    equality fails on any box that isn't already UTC."""
    assert parse_iso_epoch("2026-07-16T00:00:00") == _UTC_MIDNIGHT_2026_07_16
    # …and it agrees with the explicit-offset form to the second.
    assert parse_iso_epoch("2026-07-16T00:00:00") == parse_iso_epoch("2026-07-16T00:00:00+00:00")


def test_parse_iso_epoch_honors_a_non_utc_offset():
    """A real offset is respected: -05:00 midnight is five hours *after* UTC midnight."""
    assert parse_iso_epoch("2026-07-16T00:00:00-05:00") == _UTC_MIDNIGHT_2026_07_16 + 5 * 3600


def test_parse_iso_epoch_tolerates_surrounding_whitespace():
    assert parse_iso_epoch("  2026-07-16T00:00:00Z  ") == _UTC_MIDNIGHT_2026_07_16


def test_parse_iso_epoch_degrades_to_none_on_unparseable_input():
    """Free-form provenance: anything that isn't an ISO instant returns None so the caller
    treats the age as *unknown* rather than guessing."""
    for junk in ("not-a-date", "", "   ", "2026-13-99", "yesterday"):
        assert parse_iso_epoch(junk) is None


def test_parse_iso_epoch_none_input_is_none():
    assert parse_iso_epoch(None) is None
