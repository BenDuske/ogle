"""Unit tests for ogle.pipeline — the signature->store->scorer->narrative seam.

Pure end-to-end: signatures are hand-built (as a DataHub walk would yield), the store is
in-memory, and the LLM seam is a fake callable. Fixtures use the Task #2 shape where
`customers` feeds the deployed `churn_predictor` (serving path).
"""

import pytest

from ogle.pipeline import DriftReport, run_drift_check
from ogle.scorer import DriftKind, Severity, build_score_config
from ogle.signature import build_signature
from ogle.store import BaselineStore

CUSTOMERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
ORDERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"


def _sig(urn=CUSTOMERS_URN, **kw):
    kw.setdefault("schema_fields", [("id", "int"), ("email", "string")])
    kw.setdefault("row_count", 1000)
    return build_signature(urn, **kw)


# ---- first-run seeding ------------------------------------------------------------
def test_new_dataset_is_seeded_not_scored():
    store = BaselineStore()
    report = run_drift_check(store, [_sig()])
    assert report.new_urns == [CUSTOMERS_URN]
    assert report.scored_urns == []
    assert report.findings == []
    assert report.incident is None
    assert report.should_alert is False
    # baseline was seeded so the NEXT run can diff
    assert store.get_baseline(CUSTOMERS_URN) is not None


def test_unchanged_dataset_produces_no_drift():
    store = BaselineStore()
    store.put_baseline(_sig())
    report = run_drift_check(store, [_sig()])
    assert report.scored_urns == [CUSTOMERS_URN]
    assert report.findings == []
    assert report.incident is None
    assert "No drift" in report.narrative
    assert report.should_alert is False


# ---- freshness flows through the pipeline's injected `now` ------------------------
def test_freshness_sla_pages_through_pipeline():
    """A stale-but-otherwise-unchanged serving source pages when an SLA is configured,
    using the same `now` the pipeline threads into the scorer."""
    store = BaselineStore()
    stale_stamp = "2020-01-01T00:00:00Z"  # epoch 1577836800
    store.put_baseline(_sig(computed_at=stale_stamp))
    now = 1577836800.0 + 3 * 86_400  # 3 days after the stamp, SLA 1 day
    cfg = build_score_config(freshness_max_age_seconds=86_400)  # 1-day SLA
    report = run_drift_check(
        store, [_sig(computed_at=stale_stamp)], cfg=cfg, now=now, update_baselines=False
    )
    assert report.incident is not None
    assert any(f.kind is DriftKind.FRESHNESS for f in report.findings)
    assert report.should_alert is True


def test_no_freshness_sla_stays_quiet_through_pipeline():
    """Without an SLA the same stale stamp produces no incident — opt-in default holds."""
    store = BaselineStore()
    stale_stamp = "2020-01-01T00:00:00Z"
    store.put_baseline(_sig(computed_at=stale_stamp))
    report = run_drift_check(
        store, [_sig(computed_at=stale_stamp)], now=1577836800.0 + 3 * 86_400,
        update_baselines=False,
    )
    assert report.findings == []
    assert report.should_alert is False


# ---- real drift -------------------------------------------------------------------
def test_volume_collapse_flags_incident():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    report = run_drift_check(store, [_sig(row_count=0)])
    assert report.incident is not None
    assert report.scored_urns == [CUSTOMERS_URN]
    assert report.should_alert is True
    assert report.incident_count == 1


def test_recorded_incident_carries_drift_kinds():
    """The pipeline persists the incident's drift dimensions into memory so `ogle incidents
    --kind` can isolate a failure mode. A schema-change-plus-volume-collapse records both."""
    store = BaselineStore()
    store.put_baseline(_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1000))
    # Drop a column (schema) AND collapse the row count (volume) in one sighting.
    report = run_drift_check(store, [_sig(schema_fields=[("id", "int")], row_count=0)])
    assert report.incident is not None
    (rec,) = store.incidents()
    assert rec["kinds"] == ["schema", "volume"]  # sorted, both dimensions present


def test_serving_path_escalates_severity():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    # same drift, but declared as feeding a deployed model -> escalated
    plain = run_drift_check(
        BaselineStore(baselines={CUSTOMERS_URN: _sig(row_count=1000)}),
        [_sig(row_count=600)],
    )
    escalated = run_drift_check(
        store,
        [_sig(row_count=600)],
        serving_urns=[CUSTOMERS_URN],
    )
    assert escalated.incident.overall_severity.rank > plain.incident.overall_severity.rank
    assert escalated.incident.serving_impacted is True


def test_findings_merged_and_ranked_across_datasets():
    store = BaselineStore()
    store.put_baseline(_sig(urn=CUSTOMERS_URN, row_count=1000))
    store.put_baseline(_sig(urn=ORDERS_URN, row_count=1000))
    report = run_drift_check(
        store,
        [
            _sig(urn=CUSTOMERS_URN, row_count=0),          # collapse -> HIGH
            _sig(urn=ORDERS_URN, row_count=800),           # -20%, under 30% -> no finding
        ],
    )
    assert report.incident is not None
    # findings sorted worst-first
    ranks = [f.severity.rank for f in report.findings]
    assert ranks == sorted(ranks, reverse=True)


# ---- incident dedup / debounce ----------------------------------------------------
def test_same_incident_alerts_once_then_debounces():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))

    first = run_drift_check(store, [_sig(row_count=0)], update_baselines=False)
    assert first.is_new_incident is True
    assert first.should_alert is True
    assert first.incident_count == 1

    second = run_drift_check(store, [_sig(row_count=0)], update_baselines=False)
    assert second.is_new_incident is False
    assert second.should_alert is False           # debounced — same fingerprint
    assert second.incident_count == 2


# ---- muting (known false positives) -----------------------------------------------
def test_muted_dataset_drift_is_suppressed_not_paged():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.mute(CUSTOMERS_URN)
    report = run_drift_check(store, [_sig(row_count=0)], serving_urns=[CUSTOMERS_URN])
    # the collapse is real, but the dataset is muted -> no incident, no page
    assert report.suppressed_urns == [CUSTOMERS_URN]
    assert report.findings == []
    assert report.incident is None
    assert report.should_alert is False
    # still counted as scored (it WAS diffed) so the baseline can advance
    assert report.scored_urns == [CUSTOMERS_URN]


def test_muted_dataset_baseline_still_advances():
    # Muting silences the alert but must NOT freeze tracking, or an un-mute later would
    # diff against stale state.
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.mute(CUSTOMERS_URN)
    run_drift_check(store, [_sig(row_count=0)])
    assert store.get_baseline(CUSTOMERS_URN).row_count == 0


def test_muting_one_dataset_still_pages_another():
    store = BaselineStore()
    store.put_baseline(_sig(urn=CUSTOMERS_URN, row_count=1000))
    store.put_baseline(_sig(urn=ORDERS_URN, row_count=1000))
    store.mute(CUSTOMERS_URN)
    report = run_drift_check(
        store,
        [_sig(urn=CUSTOMERS_URN, row_count=0), _sig(urn=ORDERS_URN, row_count=0)],
    )
    # customers muted -> suppressed; orders still fires the incident
    assert report.suppressed_urns == [CUSTOMERS_URN]
    assert report.incident is not None
    assert report.incident.urns == [ORDERS_URN]
    assert report.should_alert is True


def test_muted_dataset_with_no_drift_is_not_listed_suppressed():
    # Suppression only records a muted URN when it actually drifted — a quiet muted asset
    # shouldn't show up as "silenced".
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.mute(CUSTOMERS_URN)
    report = run_drift_check(store, [_sig(row_count=1000)])
    assert report.suppressed_urns == []


def test_unmuting_restores_paging():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.mute(CUSTOMERS_URN)
    store.unmute(CUSTOMERS_URN)
    report = run_drift_check(store, [_sig(row_count=0)])
    assert report.suppressed_urns == []
    assert report.should_alert is True


def test_suppressed_urns_in_report_dict():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.mute(CUSTOMERS_URN)
    report = run_drift_check(store, [_sig(row_count=0)])
    assert report.to_dict()["suppressed_urns"] == [CUSTOMERS_URN]


def test_active_snooze_suppresses_but_lapsed_snooze_pages():
    # A snooze silences the alert while active; once it expires the same drift pages again,
    # driven purely by the injected `now` so the test is deterministic.
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    store.mute(CUSTOMERS_URN, until=100.0)

    # Read-only probe (no baseline advance) so the collapse is still on the table next run.
    active = run_drift_check(
        store, [_sig(row_count=0)], now=50.0, update_baselines=False
    )
    assert active.suppressed_urns == [CUSTOMERS_URN]
    assert active.should_alert is False

    # Same collapse, but the snooze has now lapsed -> it pages.
    lapsed = run_drift_check(store, [_sig(row_count=0)], now=150.0)
    assert lapsed.suppressed_urns == []
    assert lapsed.should_alert is True


# ---- baseline advancement ---------------------------------------------------------
def test_update_baselines_advances_state_by_default():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    run_drift_check(store, [_sig(row_count=5000)])
    assert store.get_baseline(CUSTOMERS_URN).row_count == 5000


def test_read_only_probe_does_not_touch_baselines():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    run_drift_check(store, [_sig(row_count=5000)], update_baselines=False)
    assert store.get_baseline(CUSTOMERS_URN).row_count == 1000


# ---- llm seam ---------------------------------------------------------------------
def test_llm_narrates_when_provided():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    report = run_drift_check(
        store, [_sig(row_count=0)], llm=lambda prompt: "LLM SUMMARY"
    )
    assert report.narrative.strip() == "LLM SUMMARY"


def test_llm_failure_falls_back_to_deterministic():
    def broken(_prompt):
        raise RuntimeError("model down")

    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    report = run_drift_check(store, [_sig(row_count=0)], llm=broken)
    # deterministic markdown still produced -> alert always goes out
    assert "drift" in report.narrative.lower()
    assert report.incident is not None


# ---- report shape -----------------------------------------------------------------
def test_report_to_dict_is_serializable():
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    report = run_drift_check(store, [_sig(row_count=0)])
    d = report.to_dict()
    assert d["should_alert"] is True
    assert d["incident"]["overall_severity"]
    assert d["is_new_incident"] is True


def test_empty_batch_is_clean_heartbeat():
    store = BaselineStore()
    report = run_drift_check(store, [])
    assert report.findings == []
    assert report.incident is None
    assert report.should_alert is False
    assert "No drift" in report.narrative


# ---- incident memory provenance ---------------------------------------------------
def test_incident_recorded_with_provenance_for_ogle_incidents():
    # A real drift run must persist the incident's human context into store memory so
    # `ogle incidents` can describe it, not just count an opaque fingerprint.
    store = BaselineStore()
    store.put_baseline(_sig(row_count=1000))
    report = run_drift_check(
        store, [_sig(row_count=0)], serving_urns=[CUSTOMERS_URN]
    )
    assert report.incident is not None
    (rec,) = store.incidents()
    assert rec["fingerprint"] == report.incident.fingerprint
    assert rec["count"] == 1
    assert rec["severity"] == report.incident.overall_severity.value
    assert rec["title"] == report.incident.title
    assert rec["datasets"] == len(report.incident.urns)
    assert rec["serving"] is True  # customers is on the serving path
