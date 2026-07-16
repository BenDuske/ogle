"""Unit tests for ogle.store — baselines + incident dedup memory.

Pure: a JSON file on a tmp_path is the only I/O; no DataHub, no clock. Fixtures reuse the
Task #2 shape (`customers` feeds the deployed `churn_predictor`).
"""

import json

import pytest

from ogle.signature import DatasetSignature, build_signature
from ogle.store import STORE_VERSION, BaselineStore

CUSTOMERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
ORDERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"


def _sig(urn=CUSTOMERS_URN, **kw):
    kw.setdefault("schema_fields", [("id", "int"), ("email", "string")])
    kw.setdefault("row_count", 1000)
    return build_signature(urn, **kw)


# ---- baselines --------------------------------------------------------------------
def test_empty_store_has_no_baseline():
    store = BaselineStore()
    assert store.get_baseline(CUSTOMERS_URN) is None
    assert len(store) == 0
    assert store.urns() == []


def test_put_then_get_roundtrips_signature():
    store = BaselineStore()
    sig = _sig()
    store.put_baseline(sig)
    got = store.get_baseline(CUSTOMERS_URN)
    assert got is sig
    assert CUSTOMERS_URN in store
    assert len(store) == 1


def test_put_baseline_upserts_same_urn():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.put_baseline(_sig(row_count=2000))
    assert len(store) == 1
    assert store.get_baseline(CUSTOMERS_URN).row_count == 2000


def test_urns_sorted_and_stable():
    store = BaselineStore()
    store.put_baseline(_sig(urn=ORDERS_URN))
    store.put_baseline(_sig(urn=CUSTOMERS_URN))
    assert store.urns() == sorted([ORDERS_URN, CUSTOMERS_URN])


def test_put_many():
    store = BaselineStore()
    store.put_many([_sig(urn=CUSTOMERS_URN), _sig(urn=ORDERS_URN)])
    assert len(store) == 2


# ---- incident dedup ---------------------------------------------------------------
def test_unseen_incident_is_not_seen():
    store = BaselineStore()
    assert store.has_seen("abc123") is False


def test_record_incident_counts_and_marks_seen():
    store = BaselineStore()
    assert store.record_incident("fp") == 1
    assert store.has_seen("fp") is True
    assert store.record_incident("fp") == 2
    assert store.record_incident("fp") == 3


def test_forget_incident():
    store = BaselineStore()
    store.record_incident("fp")
    store.forget_incident("fp")
    assert store.has_seen("fp") is False
    # forgetting an unknown fingerprint is a no-op, not an error
    store.forget_incident("never")


# ---- persistence ------------------------------------------------------------------
def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "store.json"
    store = BaselineStore(path=p)
    store.put_baseline(_sig(field_null_fractions={"email": 0.1}))
    store.record_incident("fp1")
    store.record_incident("fp1")
    store.save()

    loaded = BaselineStore.load(p)
    got = loaded.get_baseline(CUSTOMERS_URN)
    assert got is not None
    assert got.urn == CUSTOMERS_URN
    assert got.row_count == 1000
    assert got.field_null_fractions == {"email": 0.1}
    assert got.schema_hash == _sig(field_null_fractions={"email": 0.1}).schema_hash
    assert loaded.has_seen("fp1") is True
    assert loaded.seen_incidents["fp1"].count == 2


def test_load_missing_file_is_fresh_store(tmp_path):
    p = tmp_path / "does-not-exist.json"
    store = BaselineStore.load(p)
    assert len(store) == 0
    assert store.path == p


def test_save_is_atomic_no_tmp_left_behind(tmp_path):
    p = tmp_path / "store.json"
    BaselineStore(path=p).save()
    leftovers = list(tmp_path.glob(".ogle-store-*.tmp"))
    assert leftovers == []
    assert p.exists()


def test_save_creates_parent_dirs(tmp_path):
    p = tmp_path / "nested" / "deep" / "store.json"
    store = BaselineStore()
    store.save(p)
    assert p.exists()
    assert store.path == p


def test_save_without_path_raises():
    with pytest.raises(ValueError, match="no path"):
        BaselineStore().save()


def test_save_then_pass_explicit_path(tmp_path):
    p = tmp_path / "explicit.json"
    store = BaselineStore()
    store.put_baseline(_sig())
    returned = store.save(p)
    assert returned == p
    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN) is not None


def test_on_disk_shape_has_version(tmp_path):
    p = tmp_path / "store.json"
    BaselineStore(path=p).save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == STORE_VERSION
    assert "baselines" in data
    assert "seen_incidents" in data


def test_load_rejects_wrong_version(tmp_path):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"version": 999, "baselines": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        BaselineStore.load(p)


def test_save_overwrites_prior_good_baseline(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.put_baseline(_sig(row_count=1000))
    s1.save()

    s2 = BaselineStore.load(p)
    s2.put_baseline(_sig(row_count=5000))
    s2.save()

    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN).row_count == 5000
