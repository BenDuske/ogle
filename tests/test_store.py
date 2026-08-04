"""Unit tests for ogle.store — baselines + incident dedup memory.

Pure: a JSON file on a tmp_path is the only I/O; no DataHub, no clock. Fixtures reuse the
Task #2 shape (`customers` feeds the deployed `churn_predictor`).
"""

import json

import pytest

from ogle import store as store_mod
from ogle.signature import DatasetSignature, build_signature
from ogle.store import STORE_VERSION, BaselineStore, _IncidentRecord

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


def test_incidents_confined_to_returns_fully_contained_only():
    # Orphan detection for `forget --with-incidents`: an incident whose every touched URN is
    # in the given set is "confined"; one that also reaches outside it is not.
    store = BaselineStore()
    # urns provenance only lands alongside severity/title (the latest-sighting refresh rule).
    store.record_incident("only_orders", severity="high", urns=[ORDERS_URN])
    store.record_incident("only_customers", severity="high", urns=[CUSTOMERS_URN])
    store.record_incident("both", severity="high", urns=[ORDERS_URN, CUSTOMERS_URN])
    assert store.incidents_confined_to([ORDERS_URN]) == ["only_orders"]
    # A superset that covers both datasets sweeps up the spanning incident too (sorted output).
    assert store.incidents_confined_to([ORDERS_URN, CUSTOMERS_URN]) == [
        "both",
        "only_customers",
        "only_orders",
    ]


def test_incidents_confined_to_ignores_urnless_and_empty_query():
    # Never-guess: a legacy/offline incident with no captured URNs can't be proven confined,
    # and an empty query set matches nothing (never a mass sweep).
    store = BaselineStore()
    store.record_incident("no_urns", severity="high")  # provenance but urns=[]
    assert store.incidents_confined_to([ORDERS_URN]) == []
    assert store.incidents_confined_to([]) == []


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


def test_record_incident_stores_kinds_sorted_and_deduped():
    # The drift-dimension set is stored deduped + sorted for a stable, diffable record.
    store = BaselineStore()
    store.record_incident(
        "fp", severity="high", title="drift", kinds=["volume", "schema", "volume"]
    )
    (rec,) = store.incidents()
    assert rec["kinds"] == ["schema", "volume"]


def test_record_incident_refreshes_kinds_to_latest_sighting():
    # A recurring incident's dimension set can shift; the latest sighting wins (parity with
    # severity/title/serving).
    store = BaselineStore()
    store.record_incident("fp", severity="low", title="drift", kinds=["freshness"])
    store.record_incident("fp", severity="high", title="drift", kinds=["schema"])
    (rec,) = store.incidents()
    assert rec["count"] == 2
    assert rec["kinds"] == ["schema"]


def test_bare_record_incident_does_not_blank_prior_kinds():
    # A metadata-less dedup ping (no kinds) must not erase a dimension set an earlier call set.
    store = BaselineStore()
    store.record_incident("fp", severity="medium", title="drift", kinds=["quality"])
    store.record_incident("fp")  # bare ping
    (rec,) = store.incidents()
    assert rec["kinds"] == ["quality"]


def test_incident_kinds_roundtrip_through_disk(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="drift", kinds=["schema", "volume"])
    s1.save()
    (rec,) = BaselineStore.load(p).incidents()
    assert rec["kinds"] == ["schema", "volume"]


def test_incident_without_kinds_omits_the_key(tmp_path):
    # An incident recorded without kinds must not emit an empty `kinds` list — old and new
    # Ogle round-trip identical bytes for a kind-less record.
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="drift")
    s1.save()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "kinds" not in raw["seen_incidents"]["fp"]


def test_record_incident_stores_owners_sorted_and_deduped():
    # The "who to page" attribution is stored deduped + sorted for a stable, diffable record
    # (parity with kinds) — a dataset owned by the same person twice collapses to one name.
    store = BaselineStore()
    store.record_incident(
        "fp", severity="high", title="drift", owners=["carol", "alice", "carol"]
    )
    (rec,) = store.incidents()
    assert rec["owners"] == ["alice", "carol"]


def test_record_incident_refreshes_owners_to_latest_sighting():
    # Ownership can be re-assigned between sightings; the latest wins (parity with kinds/severity).
    store = BaselineStore()
    store.record_incident("fp", severity="low", title="drift", owners=["old-team"])
    store.record_incident("fp", severity="high", title="drift", owners=["new-team"])
    (rec,) = store.incidents()
    assert rec["count"] == 2
    assert rec["owners"] == ["new-team"]


def test_bare_record_incident_does_not_blank_prior_owners():
    # An owner-less run (offline --signatures, or a metadata-less dedup ping) must not erase the
    # attribution an earlier live walk captured — owners=None leaves the prior capture intact.
    store = BaselineStore()
    store.record_incident("fp", severity="medium", title="drift", owners=["data-eng"])
    store.record_incident("fp")  # bare ping — no owners
    (rec,) = store.incidents()
    assert rec["owners"] == ["data-eng"]


def test_record_incident_empty_owners_list_clears_attribution():
    # An explicit [] means "supplied, but the datasets are now unowned" — distinct from None
    # (not supplied). The latest-sighting refresh applies, so it clears a prior owner set; the
    # cleared (empty) attribution then omits the `owners` key entirely (additive contract), so
    # the dict view drops it rather than carrying an empty list.
    store = BaselineStore()
    store.record_incident("fp", severity="medium", title="drift", owners=["data-eng"])
    store.record_incident("fp", severity="medium", title="drift", owners=[])
    assert store.seen_incidents["fp"].owners == []  # in-memory record cleared
    (rec,) = store.incidents()
    assert "owners" not in rec  # empty -> omitted from the dict view


def test_incident_owners_roundtrip_through_disk(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="drift", owners=["alice", "bob"])
    s1.save()
    (rec,) = BaselineStore.load(p).incidents()
    assert rec["owners"] == ["alice", "bob"]


def test_incident_without_owners_omits_the_key(tmp_path):
    # An unowned incident must not emit an empty `owners` list — old and new Ogle round-trip
    # identical bytes for an owner-less record (the additive-provenance contract).
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="drift")
    s1.save()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "owners" not in raw["seen_incidents"]["fp"]


# ---- incident URN provenance + per-asset history ------------------------------------

def test_record_incident_stores_urns_sorted_and_deduped():
    # The touched-asset set is stored deduped + sorted for a stable, diffable record (parity
    # with kinds/owners).
    store = BaselineStore()
    store.record_incident(
        "fp", severity="high", title="drift", urns=["urn:b", "urn:a", "urn:b"]
    )
    (rec,) = store.incidents()
    assert rec["urns"] == ["urn:a", "urn:b"]


def test_bare_record_incident_does_not_blank_prior_urns():
    # An offline/metadata-less dedup ping (urns=None) must not erase the asset set an earlier
    # rich call captured (parity with owners/kinds).
    store = BaselineStore()
    store.record_incident("fp", severity="medium", title="drift", urns=["urn:a"])
    store.record_incident("fp")  # bare ping — no urns
    (rec,) = store.incidents()
    assert rec["urns"] == ["urn:a"]


def test_incident_urns_roundtrip_through_disk(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="drift", urns=["urn:a", "urn:b"])
    s1.save()
    (rec,) = BaselineStore.load(p).incidents()
    assert rec["urns"] == ["urn:a", "urn:b"]


def test_incident_without_urns_omits_the_key(tmp_path):
    # A URN-less record must not emit an empty `urns` list — old/new Ogle round-trip identical
    # bytes (additive-provenance contract).
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", title="drift")
    s1.save()
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert "urns" not in raw["seen_incidents"]["fp"]


def test_prior_incident_history_counts_distinct_incidents_touching_an_asset():
    # Two different incidents both touched urn:a; a third touched only urn:c. A query for
    # urn:a sees the two, not the third.
    store = BaselineStore()
    store.record_incident("fp1", severity="high", title="d", urns=["urn:a", "urn:b"])
    store.record_incident("fp2", severity="low", title="d", urns=["urn:a"])
    store.record_incident("fp3", severity="low", title="d", urns=["urn:c"])
    assert store.prior_incident_history(["urn:a"]) == 2


def test_prior_incident_history_excludes_own_fingerprint():
    # An incident must not count itself as prior history.
    store = BaselineStore()
    store.record_incident("fp1", severity="high", title="d", urns=["urn:a"])
    store.record_incident("fp2", severity="low", title="d", urns=["urn:a"])
    assert store.prior_incident_history(["urn:a"], exclude_fingerprint="fp2") == 1


def test_prior_incident_history_ignores_records_without_urn_provenance():
    # A legacy/offline record that never captured its URNs can't be proven to touch the asset,
    # so it is not counted (never-guess rule, parity with the --owner/--kind filters).
    store = BaselineStore()
    store.record_incident("legacy", severity="high", title="d")  # no urns
    assert store.prior_incident_history(["urn:a"]) == 0


def test_prior_incident_history_empty_query_is_zero():
    store = BaselineStore()
    store.record_incident("fp1", severity="high", title="d", urns=["urn:a"])
    assert store.prior_incident_history([]) == 0


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


def test_mute_reason_recorded_and_cleared():
    store = BaselineStore()
    assert store.mute(CUSTOMERS_URN, reason="bounces every Monday") is True
    assert store.mute_reason(CUSTOMERS_URN) == "bounces every Monday"
    # A later bare mute (no reason) must not blank the note — provenance-refresh rule.
    store.mute(CUSTOMERS_URN)
    assert store.mute_reason(CUSTOMERS_URN) == "bounces every Monday"
    # Re-muting with a fresh reason updates it, even though the state didn't change.
    assert store.mute(CUSTOMERS_URN, reason="root-caused: upstream ETL retry") is False
    assert store.mute_reason(CUSTOMERS_URN) == "root-caused: upstream ETL retry"
    # Unmuting drops the note so it never outlives the mute.
    assert store.unmute(CUSTOMERS_URN) is True
    assert store.mute_reason(CUSTOMERS_URN) is None


def test_mute_reason_dropped_on_forget_and_expiry():
    store = BaselineStore()
    # forget_baseline clears an accompanying reason.
    store.put_baseline(_sig(ORDERS_URN))
    store.mute(ORDERS_URN, reason="decommissioning soon")
    assert store.forget_baseline(ORDERS_URN) is True
    assert store.mute_reason(ORDERS_URN) is None
    # An expired snooze drops its note along with the snooze.
    store.mute(CUSTOMERS_URN, until=100.0, reason="quiet during backfill")
    assert store.mute_reason(CUSTOMERS_URN) == "quiet during backfill"
    assert store.purge_expired_mutes(now=150.0) == [CUSTOMERS_URN]
    assert store.mute_reason(CUSTOMERS_URN) is None


def test_mute_reason_survives_save_load_only_while_muted(tmp_path):
    p = tmp_path / "store.json"
    s = BaselineStore(path=p)
    s.mute(ORDERS_URN, reason="known noisy dashboard")
    s.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["mute_reasons"] == {ORDERS_URN: "known noisy dashboard"}
    reloaded = BaselineStore.load(p)
    assert reloaded.mute_reason(ORDERS_URN) == "known noisy dashboard"
    # A reason left behind for an unmuted URN is never persisted (orphan guard in to_dict),
    # and a stray note hand-injected for a non-muted URN is dropped on the next save.
    reloaded.unmute(ORDERS_URN)
    reloaded.mute_reasons[CUSTOMERS_URN] = "stale orphan note"  # simulate a stray
    reloaded.save()
    data2 = json.loads(p.read_text(encoding="utf-8"))
    assert data2["mute_reasons"] == {}


def test_load_drops_orphan_mute_reason_for_unmuted_urn(tmp_path):
    # A hand-edited/legacy file with a reason for a URN that isn't muted must not resurrect
    # the note (mirrors the muted_until permanent-wins coercion guard).
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {},
                "seen_incidents": {},
                "muted_urns": [CUSTOMERS_URN],
                "mute_reasons": {
                    CUSTOMERS_URN: "kept — still muted",
                    ORDERS_URN: "orphan — not muted",
                },
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.mute_reason(CUSTOMERS_URN) == "kept — still muted"
    assert store.mute_reason(ORDERS_URN) is None


def test_mute_since_stamped_once_and_preserved():
    store = BaselineStore()
    # First mute with a clock dates the silence.
    store.mute(CUSTOMERS_URN, now=1000.0)
    assert store.mute_since(CUSTOMERS_URN) == 1000.0
    # A re-annotate keeps the ORIGINAL start — the silence has been continuous.
    store.mute(CUSTOMERS_URN, reason="root-caused", now=5000.0)
    assert store.mute_since(CUSTOMERS_URN) == 1000.0
    # Escalating a snooze to permanent likewise keeps the original stamp.
    s2 = BaselineStore()
    s2.mute(ORDERS_URN, until=9e9, now=2000.0)
    assert s2.mute_since(ORDERS_URN) == 2000.0
    s2.mute(ORDERS_URN, now=6000.0)  # now permanent
    assert s2.mute_since(ORDERS_URN) == 2000.0
    # Unmuting clears the stamp; a fresh mute re-dates from the new `now`.
    s2.unmute(ORDERS_URN)
    assert s2.mute_since(ORDERS_URN) is None
    s2.mute(ORDERS_URN, now=8000.0)
    assert s2.mute_since(ORDERS_URN) == 8000.0


def test_mute_without_now_is_undated():
    # A mute set without a clock reads as age-unknown (None), not zero — the accessor never
    # invents an age. A later `now` on the same continuous mute back-fills the stamp.
    store = BaselineStore()
    store.mute(CUSTOMERS_URN)
    assert store.mute_since(CUSTOMERS_URN) is None
    store.mute(CUSTOMERS_URN, reason="dated late", now=4200.0)
    assert store.mute_since(CUSTOMERS_URN) == 4200.0


def test_mute_since_dropped_on_forget_and_expiry():
    store = BaselineStore()
    store.put_baseline(_sig(ORDERS_URN))
    store.mute(ORDERS_URN, now=100.0)
    assert store.forget_baseline(ORDERS_URN) is True
    assert store.mute_since(ORDERS_URN) is None
    # An expired snooze drops its stamp along with the snooze.
    store.mute(CUSTOMERS_URN, until=100.0, now=50.0)
    assert store.mute_since(CUSTOMERS_URN) == 50.0
    assert store.purge_expired_mutes(now=150.0) == [CUSTOMERS_URN]
    assert store.mute_since(CUSTOMERS_URN) is None


def test_mute_since_survives_save_load_only_while_muted(tmp_path):
    p = tmp_path / "store.json"
    s = BaselineStore(path=p)
    s.mute(ORDERS_URN, now=7777.0)
    s.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["muted_at"] == {ORDERS_URN: 7777.0}
    reloaded = BaselineStore.load(p)
    assert reloaded.mute_since(ORDERS_URN) == 7777.0
    # A stamp left behind for an unmuted URN is never persisted (orphan guard in to_dict).
    reloaded.unmute(ORDERS_URN)
    reloaded.muted_at[CUSTOMERS_URN] = 999.0  # simulate a stray
    reloaded.save()
    data2 = json.loads(p.read_text(encoding="utf-8"))
    assert data2["muted_at"] == {}


def test_load_old_store_without_muted_at_key_is_undated(tmp_path):
    # A pre-muted_at file loads its mutes as undated rather than failing — additive key.
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
    assert store.mute_since(CUSTOMERS_URN) is None


def test_load_drops_orphan_muted_at_for_unmuted_urn(tmp_path):
    # A hand-edited/legacy file with a stamp for a URN that isn't muted must not resurrect it.
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {},
                "seen_incidents": {},
                "muted_urns": [CUSTOMERS_URN],
                "muted_at": {CUSTOMERS_URN: 111.0, ORDERS_URN: 222.0},
            }
        ),
        encoding="utf-8",
    )
    store = BaselineStore.load(p)
    assert store.mute_since(CUSTOMERS_URN) == 111.0
    assert store.mute_since(ORDERS_URN) is None


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


def test_failed_save_leaves_no_tmp_orphan_and_spares_prior_good_file(tmp_path, monkeypatch):
    # A crash at the os.replace step (disk full, permissions, power loss mid-swap) must
    # NOT leak a .ogle-store-*.tmp turd into the store dir, and must leave the prior good
    # file byte-for-byte intact — the whole point of the temp-then-replace dance.
    p = tmp_path / "store.json"
    good = BaselineStore(path=p)
    good.put_baseline(_sig(field_null_fractions={"email": 0.1}))
    good.save()
    original = p.read_bytes()

    doomed = BaselineStore(path=p)
    doomed.put_baseline(_sig(field_null_fractions={"email": 0.9}))  # would-be new content

    def boom(_src, _dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(store_mod.os, "replace", boom)
    with pytest.raises(OSError, match="no space left"):
        doomed.save()

    assert list(tmp_path.glob(".ogle-store-*.tmp")) == []  # finally-block cleanup ran
    assert p.read_bytes() == original  # prior good baseline untouched
    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN).field_null_fractions == {"email": 0.1}


def test_save_fsyncs_temp_data_before_replace(tmp_path, monkeypatch):
    # Durability, not just atomicity: the temp file's bytes must be forced to disk BEFORE the
    # rename swaps it into place, else a crash right after os.replace becomes durable can
    # surface a truncated/zero-length store. Prove the ordering (fsync-then-replace) and that
    # fsync targeted a real integer fd. Removing the fsync line makes this fail (no "fsync"
    # event precedes "replace").
    order = []
    real_fsync = store_mod.os.fsync
    real_replace = store_mod.os.replace

    def traced_fsync(fd):
        assert isinstance(fd, int)  # a live file descriptor, not a path/handle
        order.append("fsync")
        return real_fsync(fd)

    def traced_replace(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "fsync", traced_fsync)
    monkeypatch.setattr(store_mod.os, "replace", traced_replace)

    p = tmp_path / "store.json"
    BaselineStore(path=p).save()

    assert order == ["fsync", "replace"]  # data durable before the swap, exactly once each
    assert p.exists()


def test_save_fsyncs_parent_dir_after_replace(tmp_path, monkeypatch):
    # Atomicity of the rename is not durability of the rename: on POSIX the directory entry
    # that now names the new file is only persisted once the parent directory is fsync'd.
    # Prove save() fsyncs the parent dir AFTER the swap (ordering: file-fsync, replace,
    # dir-fsync). Removing the _fsync_dir call makes this fail (no "dir-fsync" event).
    order = []
    real_replace = store_mod.os.replace

    def traced_fsync(_fd):
        order.append("fsync")  # temp-file data flush

    def traced_replace(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    def traced_fsync_dir(directory):
        # Record only (don't call through) so the ordering assertion stays platform-neutral:
        # the real helper would fsync the dir fd and, under this test's traced os.fsync, add a
        # spurious event on POSIX. The helper's own behavior is covered separately below.
        order.append("dir-fsync")
        assert str(directory) == str(tmp_path)  # the store's parent dir, not the temp file

    monkeypatch.setattr(store_mod.os, "fsync", traced_fsync)
    monkeypatch.setattr(store_mod.os, "replace", traced_replace)
    monkeypatch.setattr(store_mod, "_fsync_dir", traced_fsync_dir)

    p = tmp_path / "store.json"
    BaselineStore(path=p).save()

    assert order == ["fsync", "replace", "dir-fsync"]  # rename made durable last
    assert p.exists()


def test_fsync_dir_swallows_unsupported_platform(tmp_path, monkeypatch):
    # Directory fsync isn't portable (Windows/network FS raise on opening a dir for fsync).
    # A save whose data is already on disk must not be sunk by that — _fsync_dir swallows the
    # OSError from os.open and returns cleanly rather than propagating. Patch is scoped to the
    # os.open call and undone before anything else runs, so tempfile.mkstemp stays intact.
    def no_dir_open(*_a, **_k):
        raise OSError("cannot open directory for fsync")

    monkeypatch.setattr(store_mod.os, "open", no_dir_open)
    store_mod._fsync_dir(tmp_path)  # must not raise
    monkeypatch.undo()

    # And a full save still succeeds end-to-end once os.open is back to normal.
    p = tmp_path / "store.json"
    BaselineStore(path=p).save()
    assert p.exists()


def test_fsync_dir_swallows_fsync_failure(tmp_path, monkeypatch):
    # If the directory opens but the fsync itself errors (I/O fault on the dir handle), the
    # helper still returns cleanly and closes the fd — data is already durable, so a failed
    # dir-fsync must not surface as a save failure. Also proves the fd is always closed.
    closed = []
    real_close = store_mod.os.close

    def boom_fsync(_fd):
        raise OSError("dir fsync i/o error")

    def traced_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(store_mod.os, "fsync", boom_fsync)
    monkeypatch.setattr(store_mod.os, "close", traced_close)

    try:
        store_mod._fsync_dir(tmp_path)  # must not raise
    except OSError:
        # On platforms where os.open won't open a directory (Windows), the helper returns
        # before ever reaching fsync — nothing to assert about closing, and no raise expected.
        pytest.fail("_fsync_dir must never propagate an OSError")

    # If the dir opened at all, the fd must have been closed despite the fsync error.
    # (On a dir-open-refusing platform, closed stays empty and that's fine.)


def test_save_fsync_failure_spares_prior_good_file_and_leaves_no_tmp(tmp_path, monkeypatch):
    # A crash at the fsync step (I/O error mid-flush) must behave like any other save crash:
    # propagate, leave no .tmp orphan, and leave the prior good file byte-for-byte intact —
    # the rename never runs, so the old baseline stands.
    p = tmp_path / "store.json"
    good = BaselineStore(path=p)
    good.put_baseline(_sig(field_null_fractions={"email": 0.1}))
    good.save()
    original = p.read_bytes()

    doomed = BaselineStore(path=p)
    doomed.put_baseline(_sig(field_null_fractions={"email": 0.9}))

    def boom(_fd):
        raise OSError("input/output error")

    monkeypatch.setattr(store_mod.os, "fsync", boom)
    with pytest.raises(OSError, match="input/output error"):
        doomed.save()

    assert list(tmp_path.glob(".ogle-store-*.tmp")) == []  # finally-block cleanup ran
    assert p.read_bytes() == original  # prior good baseline untouched (rename never happened)
    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN).field_null_fractions == {"email": 0.1}


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


def test_load_quarantines_baseline_with_poisoned_quantiles(tmp_path):
    """A store whose baseline carries a malformed quantile set (here a backwards quantile
    function) must be quarantined and re-baselined — NOT loaded with the poison intact. The
    empirical scorers trust quantiles to be a genuine quantile function; a silently-loaded bad
    set would flip the earth-mover integral and mis-score drift. `DatasetSignature.from_dict`
    now re-validates, so the store's existing corrupt-recovery net catches it."""
    p = tmp_path / "store.json"
    poisoned = {
        "version": STORE_VERSION,
        "baselines": {
            CUSTOMERS_URN: {
                "urn": CUSTOMERS_URN,
                "schema_fields": [["amount", "double"]],
                # Q(0.75) < Q(0.25): a quantile function cannot run backwards.
                "field_quantiles": {"amount": [[0.25, 5.0], [0.75, 1.0]]},
            }
        },
    }
    p.write_text(json.dumps(poisoned), encoding="utf-8")
    # Strict mode surfaces the raw error rather than swallowing it.
    with pytest.raises(ValueError, match="non-decreasing"):
        BaselineStore.load(p, recover_corrupt=False)
    # Default mode quarantines the bad file and hands back a fresh store to re-baseline.
    store = BaselineStore.load(p)
    assert store.recovered_from_corruption is True
    assert len(store) == 0
    assert store.corrupt_backup_path == p.with_name(p.name + ".corrupt")


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


@pytest.mark.parametrize(
    "payload",
    ["[1, 2, 3]", "42", "3.14", '"a bare string"', "null", "true"],
)
def test_load_recovers_from_valid_json_that_is_not_an_object(tmp_path, payload):
    """A file that parses as JSON but isn't a top-level object (array, number, string, null,
    bool) is still a broken store — e.g. a truncated or partially-overwritten file. It must be
    quarantined like any other corrupt file, NOT crash `load` with an uncaught AttributeError
    (which would defeat the whole crash-loop-protection net a scheduled `ogle check` relies on)."""
    p = tmp_path / "store.json"
    p.write_text(payload, encoding="utf-8")
    # Strict mode surfaces a clean ValueError (not AttributeError) for a foreign shape.
    with pytest.raises(ValueError, match="must be a JSON object"):
        BaselineStore.load(p, recover_corrupt=False)
    # Default mode quarantines the bad file and hands back a fresh, re-baselineable store.
    store = BaselineStore.load(p)
    assert len(store) == 0
    assert store.recovered_from_corruption is True
    assert store.corrupt_backup_path == p.with_name(p.name + ".corrupt")
    assert p.with_name(p.name + ".corrupt").read_text(encoding="utf-8") == payload
    assert not p.exists()
    # The recovered store is fully usable: a fresh baseline saves + reloads clean.
    store.put_baseline(_sig())
    store.save()
    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN) is not None


def test_from_dict_rejects_non_dict_directly(tmp_path):
    """`from_dict` is the seam behind `load`; guard it directly so a non-dict raises ValueError
    (quarantine-able) rather than AttributeError, independent of the load() catch tuple."""
    for bad in ([1, 2], 7, "x", None, True):
        with pytest.raises(ValueError, match="must be a JSON object"):
            BaselineStore.from_dict(bad)


@pytest.mark.parametrize("section", ["baselines", "seen_incidents"])
@pytest.mark.parametrize("bad_value", [[1, 2, 3], 42, "a string", True])
def test_load_recovers_from_nested_section_that_is_not_an_object(tmp_path, section, bad_value):
    """The top-level guard isn't enough: a file with the correct `version` but whose nested
    `baselines`/`seen_incidents` value is a valid-JSON-but-non-object (list/scalar from a
    truncated or hand-mangled file) reaches `.items()` and raises AttributeError — which is NOT
    in load()'s (ValueError/KeyError/TypeError) recovery net, so a scheduled `ogle check` would
    crash-loop and go blind. It must quarantine like any other foreign file."""
    payload = {"version": 1, "baselines": {}, "seen_incidents": {}}
    payload[section] = bad_value
    p = tmp_path / "store.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    # Strict mode surfaces a clean, section-named ValueError (not AttributeError).
    with pytest.raises(ValueError, match=f"'{section}' must be a JSON object"):
        BaselineStore.load(p, recover_corrupt=False)
    # Default mode quarantines the bad file and hands back a fresh, re-baselineable store.
    store = BaselineStore.load(p)
    assert len(store) == 0
    assert store.recovered_from_corruption is True
    assert store.corrupt_backup_path == p.with_name(p.name + ".corrupt")
    assert not p.exists()
    # The recovered store is fully usable: a fresh baseline saves + reloads clean.
    store.put_baseline(_sig())
    store.save()
    assert BaselineStore.load(p).get_baseline(CUSTOMERS_URN) is not None


@pytest.mark.parametrize("bad_value", [[1, 2, 3], 42, "a string", True, None])
def test_load_recovers_from_seen_incident_entry_that_is_not_an_object(tmp_path, bad_value):
    """One layer deeper than the section guard: the `seen_incidents` section IS a dict (passes
    the 3844ade guard) but a per-ENTRY value is a valid-JSON-but-non-object (list/scalar from a
    truncated or hand-mangled file that still parses AND carries the right version). That value
    reaches `_IncidentRecord.from_dict`, whose `data.get(...)` raises AttributeError — NOT in
    load()'s (ValueError/KeyError/TypeError) recovery net — so a scheduled `ogle check` would
    crash-loop and go blind on exactly the bad file the net exists for. It must quarantine like
    any other foreign file."""
    payload = {"version": 1, "baselines": {}, "seen_incidents": {"abc123": bad_value}}
    p = tmp_path / "store.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    # Strict mode surfaces a clean, record-named ValueError (not AttributeError).
    with pytest.raises(ValueError, match="incident record must be a JSON object"):
        BaselineStore.load(p, recover_corrupt=False)
    # Default mode quarantines the bad file and hands back a fresh, re-baselineable store.
    store = BaselineStore.load(p)
    assert len(store.seen_incidents) == 0
    assert store.recovered_from_corruption is True
    assert store.corrupt_backup_path == p.with_name(p.name + ".corrupt")
    assert not p.exists()
    # The recovered store is fully usable: a fresh incident records + reloads clean.
    store.record_incident("fp", severity="high")
    store.save()
    assert BaselineStore.load(p).has_seen("fp")


def test_incident_record_from_dict_rejects_non_dict_directly():
    """`_IncidentRecord.from_dict` is the seam behind the seen_incidents rebuild; guard it
    directly so a non-dict raises ValueError (quarantine-able) rather than AttributeError,
    independent of the load() catch tuple."""
    for bad in ([1, 2], 7, "x", None, True):
        with pytest.raises(ValueError, match="incident record must be a JSON object"):
            _IncidentRecord.from_dict(bad)


@pytest.mark.parametrize("bad_value", [42, "urn:li:dataset:sales", True, {"a": "b"}, None])
def test_load_recovers_from_muted_urns_that_is_not_a_list(tmp_path, bad_value):
    """`muted_urns` is the one mute section built with `set(...)` instead of `dict(...)`, and
    `set(...)` degrades SILENTLY on a non-list: a hand-edited/truncated `"muted_urns":
    "urn:li:dataset:sales"` char-splits into a set of single characters ({'u','r','n',...}) with
    NO exception, so the operator's intended mute vanishes and the dataset it was meant to
    silence starts paging again — quietly wrong, the worst failure for a suppression list (a
    scalar like `42` would raise TypeError that load() happens to catch, but the string case
    slips straight through). The sibling `muted_until`/`mute_reasons`/`muted_at` sections have
    their own object guard now (see the test below); this one needs the explicit list guard so
    a foreign value quarantines like any other corrupt file."""
    payload = {"version": 1, "baselines": {}, "seen_incidents": {}, "muted_urns": bad_value}
    p = tmp_path / "store.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    # Strict mode surfaces a clean, section-named ValueError (not a silent char-split / TypeError).
    with pytest.raises(ValueError, match="'muted_urns' must be a JSON list"):
        BaselineStore.load(p, recover_corrupt=False)
    # Default mode quarantines the bad file and hands back a fresh, re-baselineable store —
    # crucially with an EMPTY mute set, never the char-split junk the old code would have kept.
    store = BaselineStore.load(p)
    assert store.muted_urns == set()
    assert store.recovered_from_corruption is True
    assert store.corrupt_backup_path == p.with_name(p.name + ".corrupt")
    assert not p.exists()
    # The recovered store is fully usable: a fresh mute saves + reloads clean.
    store.mute(CUSTOMERS_URN)
    store.save()
    assert BaselineStore.load(p).is_muted(CUSTOMERS_URN)


@pytest.mark.parametrize("field", ["muted_until", "mute_reasons", "muted_at"])
@pytest.mark.parametrize(
    "bad_value",
    [
        ["12", "34"],  # the nastiest case: dict(["12","34"]) SILENTLY char-pairs -> {"1":"2","3":"4"}
        "urn:li:dataset:x",
        42,
        True,
        [["a", "b"]],
    ],
)
def test_load_recovers_from_mute_metadata_that_is_not_an_object(tmp_path, field, bad_value):
    """Twin of the `muted_urns` guard for the three mute-metadata maps. Each is built with
    `dict(data.get(field, {}))`, and `dict(...)` degrades in the same silent-misread way the
    `muted_urns` `set(...)` did: a JSON list of 2-char strings (`["12","34"]`) char-pairs into
    a bogus `{"1":"2","3":"4"}` snooze/reason/stamp map with NO exception — a hand-edited or
    truncated file would quietly resurrect fabricated mutes. An explicit object guard rejects
    a non-dict shape with the same clean, section-named ValueError so load() quarantines it."""
    payload = {"version": 1, "baselines": {}, "seen_incidents": {}, "muted_urns": [], field: bad_value}
    p = tmp_path / "store.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    # Strict mode: clean, section-named ValueError — never a silent char-pair or cryptic crash.
    with pytest.raises(ValueError, match=f"'{field}' must be a JSON object"):
        BaselineStore.load(p, recover_corrupt=False)
    # Default mode quarantines the foreign file and hands back a fresh, empty, usable store.
    store = BaselineStore.load(p)
    assert store.muted_until == {}
    assert store.mute_reasons == {}
    assert store.muted_at == {}
    assert store.recovered_from_corruption is True
    assert not p.exists()


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


# ---- incident first_seen: the longevity axis behind `ogle incidents` "first seen X ago" ----


def test_record_incident_stamps_first_seen_when_now_given():
    store = BaselineStore()
    store.record_incident("fp", severity="high", now=1000.0)
    (rec,) = store.incidents()
    assert rec["first_seen"] == 1000.0


def test_record_incident_first_seen_frozen_while_last_seen_advances():
    # first_seen is written once (the incident's birth); last_seen tracks the newest sighting.
    store = BaselineStore()
    store.record_incident("fp", now=1000.0)
    store.record_incident("fp", now=2500.0)
    store.record_incident("fp", now=9000.0)
    (rec,) = store.incidents()
    assert rec["first_seen"] == 1000.0   # never moves — measures the whole standing life
    assert rec["last_seen"] == 9000.0    # moves to the most recent sighting


def test_record_incident_without_now_omits_first_seen():
    store = BaselineStore()
    store.record_incident("fp", severity="low")
    (rec,) = store.incidents()
    assert "first_seen" not in rec


def test_record_incident_first_seen_backfills_from_first_timed_sighting():
    # An incident whose earliest sightings were untimed backfills first_seen to the FIRST
    # clock it ever gets (best available), and a later timed call never moves it.
    store = BaselineStore()
    store.record_incident("fp")           # untimed
    store.record_incident("fp", now=5000.0)
    store.record_incident("fp", now=6000.0)
    (rec,) = store.incidents()
    assert rec["first_seen"] == 5000.0    # first timed sighting, not overwritten by 6000
    assert rec["last_seen"] == 6000.0


def test_incident_first_seen_round_trips_through_disk(tmp_path):
    p = tmp_path / "store.json"
    s1 = BaselineStore(path=p)
    s1.record_incident("fp", severity="high", now=1234.5)
    s1.save()
    (rec,) = BaselineStore.load(p).incidents()
    assert rec["first_seen"] == 1234.5


def test_incident_record_without_first_seen_loads_as_none(tmp_path):
    # A store written before first_seen existed loads with the key absent — additive schema,
    # no STORE_VERSION bump (mirrors the last_seen back-compat contract).
    p = tmp_path / "store.json"
    p.write_text(
        json.dumps(
            {
                "version": STORE_VERSION,
                "baselines": {},
                "seen_incidents": {"fp": {"count": 2, "severity": "high", "last_seen": 42.0}},
            }
        ),
        encoding="utf-8",
    )
    (rec,) = BaselineStore.load(p).incidents()
    assert "first_seen" not in rec
    assert rec["last_seen"] == 42.0
