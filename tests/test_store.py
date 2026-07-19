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


def test_forget_baseline_removes_only_that_dataset():
    store = BaselineStore()
    store.put_baseline(_sig(urn=CUSTOMERS_URN))
    store.put_baseline(_sig(urn=ORDERS_URN))
    assert store.forget_baseline(ORDERS_URN) is True
    assert ORDERS_URN not in store and CUSTOMERS_URN in store
    assert len(store) == 1


def test_forget_baseline_unknown_urn_is_false_noop():
    store = BaselineStore()
    store.put_baseline(_sig(urn=CUSTOMERS_URN))
    assert store.forget_baseline(ORDERS_URN) is False  # never watched
    assert len(store) == 1


def test_forget_baseline_also_clears_mute_and_snooze():
    # A mute/snooze pointing at a forgotten dataset is an orphan — forget clears both forms.
    store = BaselineStore()
    store.put_baseline(_sig(urn=CUSTOMERS_URN))
    store.put_baseline(_sig(urn=ORDERS_URN))
    store.mute(CUSTOMERS_URN)               # permanent
    store.mute(ORDERS_URN, until=1e18)      # snooze
    assert store.forget_baseline(CUSTOMERS_URN) is True
    assert store.forget_baseline(ORDERS_URN) is True
    assert store.muted() == []             # both mute forms gone with their datasets


def test_forget_baseline_leaves_incidents_untouched():
    # Incidents are keyed by fingerprint (a drift event), not URN — forget must not drop them.
    store = BaselineStore()
    store.put_baseline(_sig(urn=CUSTOMERS_URN))
    store.record_incident("fp_for_customers", severity="high")
    store.forget_baseline(CUSTOMERS_URN)
    assert store.has_seen("fp_for_customers")  # the drift memory outlives the dataset row


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


# ---- incident memory (provenance for `ogle incidents`) ----------------------------
def test_incidents_empty_store():
    assert BaselineStore().incidents() == []


def test_record_incident_stores_provenance():
    store = BaselineStore()
    store.record_incident(
        "fp", severity="high", title="HIGH drift across 2 datasets", datasets=2, serving=True
    )
    (rec,) = store.incidents()
    assert rec["fingerprint"] == "fp"
    assert rec["count"] == 1
    assert rec["severity"] == "high"
    assert rec["title"] == "HIGH drift across 2 datasets"
    assert rec["datasets"] == 2
    assert rec["serving"] is True


def test_record_incident_refreshes_provenance_to_latest_sighting():
    store = BaselineStore()
    store.record_incident("fp", severity="low", title="LOW drift", datasets=1)
    store.record_incident("fp", severity="high", title="HIGH drift", datasets=3, serving=True)
    (rec,) = store.incidents()
    assert rec["count"] == 2          # recurrence still accrues
    assert rec["severity"] == "high"  # latest sighting wins
    assert rec["datasets"] == 3
    assert rec["serving"] is True


def test_bare_record_incident_does_not_blank_prior_provenance():
    # A metadata-less dedup ping must not erase the human context an earlier rich call set.
    store = BaselineStore()
    store.record_incident("fp", severity="medium", title="MED drift", datasets=1)
    store.record_incident("fp")  # bare ping (e.g. a caller that only dedups)
    (rec,) = store.incidents()
    assert rec["count"] == 2
    assert rec["severity"] == "medium"
    assert rec["title"] == "MED drift"


def test_incident_provenance_roundtrips_through_disk(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="HIGH drift", datasets=2, serving=True)
    s1.save()
    (rec,) = BaselineStore.load(p).incidents()
    assert rec == {
        "fingerprint": "fp",
        "count": 1,
        "severity": "high",
        "title": "HIGH drift",
        "datasets": 2,
        "serving": True,
    }


def test_bare_incident_record_serializes_minimally(tmp_path):
    # An incident recorded without provenance keeps the old on-disk shape (count only),
    # so old and new Ogle round-trip the same bytes for a bare record.
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp")
    s1.save()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["seen_incidents"]["fp"] == {"count": 1}


def test_legacy_bare_count_record_loads(tmp_path):
    # A store file written by an older Ogle (count only, no provenance keys) must load and
    # surface as an incident with empty/defaulted provenance — never crash.
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps({"version": STORE_VERSION, "seen_incidents": {"old": {"count": 5}}}),
        encoding="utf-8",
    )
    (rec,) = BaselineStore.load(p).incidents()
    assert rec["fingerprint"] == "old"
    assert rec["count"] == 5
    # provenance absent on a bare legacy record -> keys simply not present (to_dict omits
    # None/zero/False), which the CLI reads with .get() defaults.
    assert rec.get("severity") is None
    assert rec.get("datasets", 0) == 0
    assert rec.get("serving", False) is False


# ---- muting (known false positives) -----------------------------------------------
def test_empty_store_mutes_nothing():
    store = BaselineStore()
    assert store.is_muted(CUSTOMERS_URN) is False
    assert store.muted() == []


def test_mute_marks_and_reports_newly():
    store = BaselineStore()
    assert store.mute(CUSTOMERS_URN) is True
    assert store.is_muted(CUSTOMERS_URN) is True
    # muting again is a no-op that reports "not newly muted"
    assert store.mute(CUSTOMERS_URN) is False


def test_unmute_reverses_and_reports():
    store = BaselineStore()
    store.mute(CUSTOMERS_URN)
    assert store.unmute(CUSTOMERS_URN) is True
    assert store.is_muted(CUSTOMERS_URN) is False
    # un-muting something never muted reports False, not an error
    assert store.unmute(ORDERS_URN) is False


def test_muted_list_sorted_and_stable():
    store = BaselineStore()
    store.mute(ORDERS_URN)
    store.mute(CUSTOMERS_URN)
    assert store.muted() == sorted([ORDERS_URN, CUSTOMERS_URN])


def test_muted_urns_survive_save_load(tmp_path):
    p = tmp_path / "store.json"
    store = BaselineStore(path=p)
    store.mute(CUSTOMERS_URN)
    store.save()
    assert BaselineStore.load(p).is_muted(CUSTOMERS_URN) is True


def test_muted_urns_in_on_disk_shape(tmp_path):
    p = tmp_path / "store.json"
    s = BaselineStore(path=p)
    s.mute(CUSTOMERS_URN)
    s.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["muted_urns"] == [CUSTOMERS_URN]


def test_load_old_store_without_muted_key_is_empty(tmp_path):
    # A file written by an older Ogle (no muted_urns key) must still load — additive field.
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps({"version": STORE_VERSION, "baselines": {}, "seen_incidents": {}}),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.muted() == []


# ---- timed mutes / snooze ---------------------------------------------------------
def test_snooze_active_before_expiry_lapses_after():
    store = BaselineStore()
    assert store.mute(CUSTOMERS_URN, until=100.0) is True
    # Active while now < expiry, gone once now >= expiry.
    assert store.is_muted(CUSTOMERS_URN, now=50.0) is True
    assert store.is_muted(CUSTOMERS_URN, now=100.0) is False
    assert store.is_muted(CUSTOMERS_URN, now=150.0) is False


def test_snooze_without_now_reads_as_configured_muted():
    # No clock supplied -> "is it in the mute list at all" -> True even for a snooze.
    store = BaselineStore()
    store.mute(CUSTOMERS_URN, until=100.0)
    assert store.is_muted(CUSTOMERS_URN) is True
    assert store.mute_expiry(CUSTOMERS_URN) == 100.0


def test_permanent_mute_supersedes_snooze():
    store = BaselineStore()
    store.mute(CUSTOMERS_URN, until=100.0)
    assert store.mute(CUSTOMERS_URN) is True  # promote to permanent
    assert store.mute_expiry(CUSTOMERS_URN) is None
    assert store.is_muted(CUSTOMERS_URN, now=10_000.0) is True  # never expires now
    # And a snooze can't downgrade an existing permanent mute.
    assert store.mute(CUSTOMERS_URN, until=200.0) is False
    assert store.mute_expiry(CUSTOMERS_URN) is None


def test_muted_list_excludes_expired_snooze_with_now():
    store = BaselineStore()
    store.mute(CUSTOMERS_URN)                 # permanent
    store.mute(ORDERS_URN, until=100.0)       # snooze
    assert store.muted(now=50.0) == sorted([CUSTOMERS_URN, ORDERS_URN])
    assert store.muted(now=150.0) == [CUSTOMERS_URN]  # snooze lapsed
    # Without a clock, both count as configured.
    assert store.muted() == sorted([CUSTOMERS_URN, ORDERS_URN])


def test_unmute_clears_a_snooze():
    store = BaselineStore()
    store.mute(CUSTOMERS_URN, until=100.0)
    assert store.unmute(CUSTOMERS_URN) is True
    assert store.mute_expiry(CUSTOMERS_URN) is None
    assert store.is_muted(CUSTOMERS_URN, now=50.0) is False


def test_purge_expired_mutes_drops_only_lapsed():
    store = BaselineStore()
    store.mute(CUSTOMERS_URN)                 # permanent — never purged
    store.mute(ORDERS_URN, until=100.0)       # expired by now=150
    freed = store.purge_expired_mutes(now=150.0)
    assert freed == [ORDERS_URN]
    assert store.mute_expiry(ORDERS_URN) is None
    assert store.is_muted(CUSTOMERS_URN) is True


def test_snooze_survives_save_load_and_on_disk_shape(tmp_path):
    p = tmp_path / "store.json"
    s = BaselineStore(path=p)
    s.mute(ORDERS_URN, until=12345.0)
    s.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["muted_until"] == {ORDERS_URN: 12345.0}
    reloaded = BaselineStore.load(p)
    assert reloaded.mute_expiry(ORDERS_URN) == 12345.0
    assert reloaded.is_muted(ORDERS_URN, now=1.0) is True


def test_load_old_store_without_muted_until_key_is_empty(tmp_path):
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {},
                "seen_incidents": {},
                "muted_urns": [CUSTOMERS_URN],
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.muted() == [CUSTOMERS_URN]
    assert store.mute_expiry(CUSTOMERS_URN) is None


def test_load_coerces_permanent_over_conflicting_snooze(tmp_path):
    # A hand-edited file listing a URN as BOTH permanent and snoozed: permanent wins,
    # the snooze entry is dropped so state stays coherent.
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {},
                "seen_incidents": {},
                "muted_urns": [CUSTOMERS_URN],
                "muted_until": {CUSTOMERS_URN: 100.0},
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.mute_expiry(CUSTOMERS_URN) is None
    assert store.is_muted(CUSTOMERS_URN, now=10_000.0) is True


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


def test_load_rejects_wrong_version_in_strict_mode(tmp_path):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"version": 999, "baselines": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        BaselineStore.load(p, recover_corrupt=False)


# ---- corruption resilience -------------------------------------------------------
def test_load_recovers_from_corrupt_json(tmp_path):
    # A scheduled `ogle check` must not crash-loop (and go blind) on a mangled store file.
    p = tmp_path / "store.json"
    p.write_text("{ this is not: valid json ]", encoding="utf-8")
    store = BaselineStore.load(p)
    # Fresh empty store, flagged so a caller can warn.
    assert len(store) == 0
    assert store.recovered_from_corruption is True
    assert store.path == p
    # Bad file was preserved for forensics, not deleted...
    backup = p.with_name(p.name + ".corrupt")
    assert store.corrupt_backup_path == backup
    assert backup.read_text(encoding="utf-8") == "{ this is not: valid json ]"
    # ...and moved aside so the canonical path is free for a clean re-baseline.
    assert not p.exists()
    store.put_baseline(_sig())
    store.save()
    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN) is not None


def test_load_recovers_from_wrong_version_by_default(tmp_path):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"version": 999, "baselines": {}}), encoding="utf-8")
    store = BaselineStore.load(p)
    assert store.recovered_from_corruption is True
    assert store.corrupt_backup_path.exists()


def test_corrupt_backup_never_clobbers_prior_forensic_copy(tmp_path):
    p = tmp_path / "store.json"
    # First recovery claims <name>.corrupt
    p.write_text("garbage-1", encoding="utf-8")
    s1 = BaselineStore.load(p)
    assert s1.corrupt_backup_path == p.with_name(p.name + ".corrupt")
    # A second corrupt file must land at .corrupt.1, leaving the first copy intact.
    p.write_text("garbage-2", encoding="utf-8")
    s2 = BaselineStore.load(p)
    assert s2.corrupt_backup_path == p.with_name(p.name + ".corrupt.1")
    assert p.with_name(p.name + ".corrupt").read_text(encoding="utf-8") == "garbage-1"
    assert p.with_name(p.name + ".corrupt.1").read_text(encoding="utf-8") == "garbage-2"


def test_load_of_good_store_is_not_flagged_recovered(tmp_path):
    p = tmp_path / "store.json"
    s = BaselineStore(path=p)
    s.put_baseline(_sig())
    s.save()
    loaded = BaselineStore.load(p)
    assert loaded.recovered_from_corruption is False
    assert loaded.corrupt_backup_path is None
    # No spurious quarantine file created for a healthy load.
    assert not p.with_name(p.name + ".corrupt").exists()


def test_recovery_flags_excluded_from_equality_and_persistence(tmp_path):
    # The runtime-only recovery flags must not leak into the on-disk shape or break eq.
    p = tmp_path / "store.json"
    bad = BaselineStore(path=p)
    bad.recovered_from_corruption = True
    bad.corrupt_backup_path = p.with_name("x.corrupt")
    good = BaselineStore(path=p)
    assert bad == good  # compare=False fields ignored
    bad.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "recovered_from_corruption" not in data
    assert "corrupt_backup_path" not in data


def test_save_overwrites_prior_good_baseline(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.put_baseline(_sig(row_count=1000))
    s1.save()

    s2 = BaselineStore.load(p)
    s2.put_baseline(_sig(row_count=5000))
    s2.save()

    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN).row_count == 5000


# ---- incident last_seen: the temporal axis behind `ogle incidents` age/--stale ----------


def test_record_incident_stamps_last_seen_when_now_given():
    store = BaselineStore()
    store.record_incident("fp", severity="high", now=1000.0)
    (rec,) = store.incidents()
    assert rec["last_seen"] == 1000.0


def test_record_incident_refreshes_last_seen_to_latest_sighting():
    store = BaselineStore()
    store.record_incident("fp", now=1000.0)
    store.record_incident("fp", now=2500.0)
    (rec,) = store.incidents()
    assert rec["count"] == 2            # recurrence still accrues
    assert rec["last_seen"] == 2500.0   # last_seen = most recent sighting


def test_record_incident_without_now_omits_last_seen():
    # A sighting recorded without a clock (now=None) leaves the record untimed rather than
    # inventing a timestamp — to_dict drops the key so old bare-count records round-trip.
    store = BaselineStore()
    store.record_incident("fp", severity="low")
    (rec,) = store.incidents()
    assert "last_seen" not in rec


def test_record_incident_none_now_does_not_clear_existing_last_seen():
    # An untimed dedup ping must not erase an age an earlier timed call captured (mirrors
    # the provenance-refresh rule).
    store = BaselineStore()
    store.record_incident("fp", now=1000.0)
    store.record_incident("fp")  # no now
    (rec,) = store.incidents()
    assert rec["last_seen"] == 1000.0


def test_incident_last_seen_round_trips_through_disk(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", now=1234.5)
    s1.save()
    (rec,) = BaselineStore.load(p).incidents()
    assert rec["last_seen"] == 1234.5


def test_incident_record_without_last_seen_loads_as_none(tmp_path):
    # A store written by an older Ogle (no last_seen key) loads with last_seen absent —
    # additive schema, no STORE_VERSION bump.
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {},
                "seen_incidents": {"fp": {"count": 2, "severity": "high"}},
            }
        ),
        encoding="utf-8",
    )
    (rec,) = BaselineStore.load(p).incidents()
    assert "last_seen" not in rec
    assert rec["count"] == 2
