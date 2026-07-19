"""Unit tests for ogle.writeback — pure plan/apply, dict-backed fake backend.

No `acryl-datahub` import at test time. The `FakeWritebackBackend` stores tag URN sets in
a dict, mirroring the shape the live adapter would expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

import pytest

from ogle.scorer import DriftFinding, DriftKind, Severity
from ogle.walker import WalkResult
from ogle.writeback import (
    OGLE_DRIFT_TAG,
    TagAction,
    WritebackPlan,
    WritebackResult,
    apply,
    plan_writeback,
    severity_tag_urn,
)

# ---- URNs (Task #2 shape) -------------------------------------------------------------
MODEL_CHURN = "urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.churn_predictor,PROD)"
MODEL_DEMAND = "urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.demand_forecast,PROD)"
DS_CUSTOMERS = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
DS_ORDERS = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"
DS_ISOLATED = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.isolated,PROD)"


def _finding(urn=DS_CUSTOMERS, kind=DriftKind.SCHEMA, severity=Severity.HIGH) -> DriftFinding:
    return DriftFinding(urn=urn, kind=kind, severity=severity, message=f"{kind.value} drift")


def _walk(mapping: Dict[str, list]) -> WalkResult:
    return WalkResult(dataset_to_models=mapping)


# ---- Fake backend ---------------------------------------------------------------------
@dataclass
class FakeWritebackBackend:
    """In-memory backend. Tracks the tag URN set per entity + how many times set was called."""

    tags: Dict[str, Set[str]] = field(default_factory=dict)
    write_calls: int = 0
    fail_on_read: Set[str] = field(default_factory=set)
    fail_on_write: Set[str] = field(default_factory=set)

    def existing_tag_urns(self, entity_urn: str) -> Set[str]:
        if entity_urn in self.fail_on_read:
            raise RuntimeError(f"simulated read failure for {entity_urn}")
        return set(self.tags.get(entity_urn, set()))

    def set_tag_urns(self, entity_urn: str, tag_urns: Iterable[str]) -> None:
        if entity_urn in self.fail_on_write:
            raise RuntimeError(f"simulated write failure for {entity_urn}")
        self.write_calls += 1
        self.tags[entity_urn] = set(tag_urns)


# =======================================================================================
# plan_writeback
# =======================================================================================


def test_plan_empty_when_no_findings():
    plan = plan_writeback([])
    assert isinstance(plan, WritebackPlan)
    assert plan.actions == []
    assert len(plan) == 0


def test_plan_tags_dataset_and_downstream_model():
    findings = [_finding(DS_CUSTOMERS)]
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN]})
    plan = plan_writeback(findings, walk)
    urns = [a.entity_urn for a in plan.actions]
    # Dataset is tagged first (findings order), then its downstream model.
    assert urns == [DS_CUSTOMERS, MODEL_CHURN]
    assert {a.tag_urn for a in plan.actions} == {OGLE_DRIFT_TAG}


def test_plan_dedupes_repeated_finding_on_same_dataset():
    """Two findings on the same dataset (e.g., schema AND volume drift) -> one dataset action."""
    findings = [
        _finding(DS_CUSTOMERS, kind=DriftKind.SCHEMA),
        _finding(DS_CUSTOMERS, kind=DriftKind.VOLUME),
    ]
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN]})
    plan = plan_writeback(findings, walk)
    urns = [a.entity_urn for a in plan.actions]
    assert urns.count(DS_CUSTOMERS) == 1
    assert urns.count(MODEL_CHURN) == 1


def test_plan_tags_multiple_downstream_models():
    findings = [_finding(DS_ORDERS)]
    walk = _walk({DS_ORDERS: [MODEL_CHURN, MODEL_DEMAND]})
    plan = plan_writeback(findings, walk)
    urns = {a.entity_urn for a in plan.actions}
    assert urns == {DS_ORDERS, MODEL_CHURN, MODEL_DEMAND}


def test_plan_dedupes_shared_model_across_datasets():
    """Two drifted datasets both feed churn_predictor -> churn tagged once."""
    findings = [_finding(DS_CUSTOMERS), _finding(DS_ORDERS)]
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN], DS_ORDERS: [MODEL_CHURN]})
    plan = plan_writeback(findings, walk)
    urns = [a.entity_urn for a in plan.actions]
    assert urns.count(MODEL_CHURN) == 1


def test_plan_dataset_only_when_no_walk_result():
    """Without a walk_result Ogle still tags the drifted dataset directly."""
    plan = plan_writeback([_finding(DS_ISOLATED)])
    assert len(plan) == 1
    assert plan.actions[0].entity_urn == DS_ISOLATED


def test_plan_dataset_only_when_dataset_absent_from_walk():
    """A drifted dataset that ISN'T in the walk's reverse index still gets tagged itself."""
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN]})  # ISOLATED not in walk
    plan = plan_writeback([_finding(DS_ISOLATED)], walk)
    urns = [a.entity_urn for a in plan.actions]
    assert urns == [DS_ISOLATED]  # no model action added


def test_plan_reason_carries_finding_kind_and_severity():
    plan = plan_writeback([_finding(DS_CUSTOMERS, kind=DriftKind.VOLUME, severity=Severity.MEDIUM)])
    assert "volume" in plan.actions[0].reason
    assert "medium" in plan.actions[0].reason


def test_plan_supports_custom_tag_urn():
    plan = plan_writeback(
        [_finding(DS_CUSTOMERS)],
        _walk({DS_CUSTOMERS: [MODEL_CHURN]}),
        tag_urn="urn:li:tag:custom",
    )
    assert {a.tag_urn for a in plan.actions} == {"urn:li:tag:custom"}


def test_plan_to_dict_shape():
    plan = plan_writeback([_finding(DS_CUSTOMERS)], _walk({DS_CUSTOMERS: [MODEL_CHURN]}))
    dumped = plan.to_dict()
    assert "actions" in dumped
    assert len(dumped["actions"]) == 2
    assert dumped["actions"][0]["entity_urn"] == DS_CUSTOMERS


# =======================================================================================
# severity write-back tags (--write-back-severity)
# =======================================================================================


def test_severity_tag_urn_helper():
    assert severity_tag_urn(Severity.HIGH) == "urn:li:tag:ogle-drift-high"
    assert severity_tag_urn(Severity.MEDIUM) == "urn:li:tag:ogle-drift-medium"
    assert severity_tag_urn(Severity.LOW) == "urn:li:tag:ogle-drift-low"
    # Accepts a raw string value + degrades on empty rather than raising.
    assert severity_tag_urn("high") == "urn:li:tag:ogle-drift-high"
    assert severity_tag_urn("") == "urn:li:tag:ogle-drift-unknown"


def test_severity_off_by_default_only_flat_tag():
    """No --write-back-severity -> the flat tag is the ONLY tag (behavior unchanged)."""
    plan = plan_writeback([_finding(DS_CUSTOMERS, severity=Severity.HIGH)])
    assert {a.tag_urn for a in plan.actions} == {OGLE_DRIFT_TAG}


def test_severity_stamps_flat_plus_severity_tag_on_dataset():
    plan = plan_writeback(
        [_finding(DS_CUSTOMERS, severity=Severity.HIGH)], severity_tags=True
    )
    tags = {a.tag_urn for a in plan.actions if a.entity_urn == DS_CUSTOMERS}
    assert tags == {OGLE_DRIFT_TAG, "urn:li:tag:ogle-drift-high"}


def test_severity_flat_tag_leads_for_an_entity():
    """The coarse flat tag is emitted before the severity tag for a given dataset."""
    plan = plan_writeback(
        [_finding(DS_CUSTOMERS, severity=Severity.LOW)], severity_tags=True
    )
    ds_actions = [a for a in plan.actions if a.entity_urn == DS_CUSTOMERS]
    assert [a.tag_urn for a in ds_actions] == [OGLE_DRIFT_TAG, "urn:li:tag:ogle-drift-low"]


def test_severity_dataset_uses_worst_of_its_findings():
    """customers has a LOW volume finding + a HIGH schema finding -> HIGH severity tag."""
    findings = [
        _finding(DS_CUSTOMERS, kind=DriftKind.VOLUME, severity=Severity.LOW),
        _finding(DS_CUSTOMERS, kind=DriftKind.SCHEMA, severity=Severity.HIGH),
    ]
    plan = plan_writeback(findings, severity_tags=True)
    sev_tags = {
        a.tag_urn for a in plan.actions if a.tag_urn.startswith("urn:li:tag:ogle-drift-")
    } - {OGLE_DRIFT_TAG}
    assert sev_tags == {"urn:li:tag:ogle-drift-high"}


def test_severity_model_inherits_worst_upstream_severity():
    """churn is fed by a MEDIUM dataset + a HIGH dataset -> churn gets the HIGH tag."""
    findings = [
        _finding(DS_CUSTOMERS, severity=Severity.MEDIUM),
        _finding(DS_ORDERS, severity=Severity.HIGH),
    ]
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN], DS_ORDERS: [MODEL_CHURN]})
    plan = plan_writeback(findings, walk, severity_tags=True)
    model_tags = {a.tag_urn for a in plan.actions if a.entity_urn == MODEL_CHURN}
    assert model_tags == {OGLE_DRIFT_TAG, "urn:li:tag:ogle-drift-high"}


def test_severity_tags_apply_end_to_end():
    """A severity-tagged plan lands both tags on the dataset via the fake backend."""
    plan = plan_writeback(
        [_finding(DS_CUSTOMERS, severity=Severity.HIGH)], severity_tags=True
    )
    backend = FakeWritebackBackend()
    apply(plan, backend)
    assert backend.tags[DS_CUSTOMERS] == {OGLE_DRIFT_TAG, "urn:li:tag:ogle-drift-high"}


# =======================================================================================
# apply
# =======================================================================================


def test_apply_writes_missing_tags_and_returns_applied():
    backend = FakeWritebackBackend()
    plan = plan_writeback([_finding(DS_CUSTOMERS)], _walk({DS_CUSTOMERS: [MODEL_CHURN]}))
    result = apply(plan, backend)
    assert set(result.tagged_entities) == {DS_CUSTOMERS, MODEL_CHURN}
    assert len(result.applied) == 2
    assert result.unchanged == []
    assert backend.tags[DS_CUSTOMERS] == {OGLE_DRIFT_TAG}
    assert backend.tags[MODEL_CHURN] == {OGLE_DRIFT_TAG}


def test_apply_is_idempotent_when_tag_already_present():
    """Second apply of the same plan writes nothing new."""
    backend = FakeWritebackBackend()
    plan = plan_writeback([_finding(DS_CUSTOMERS)], _walk({DS_CUSTOMERS: [MODEL_CHURN]}))
    apply(plan, backend)
    writes_after_first = backend.write_calls
    result = apply(plan, backend)
    assert result.applied == []
    assert len(result.unchanged) == 2
    # No additional writes on the second pass.
    assert backend.write_calls == writes_after_first


def test_apply_preserves_existing_unrelated_tags():
    """Ogle must not clobber tags a human put on the entity."""
    backend = FakeWritebackBackend(tags={DS_CUSTOMERS: {"urn:li:tag:pii"}})
    plan = plan_writeback([_finding(DS_CUSTOMERS)])
    apply(plan, backend)
    assert backend.tags[DS_CUSTOMERS] == {"urn:li:tag:pii", OGLE_DRIFT_TAG}


def test_apply_merges_multiple_tags_on_one_entity_in_a_single_write():
    """Two actions targeting the same entity (custom + default tag) -> one write, both tags."""
    backend = FakeWritebackBackend()
    plan = WritebackPlan(
        actions=[
            TagAction(DS_CUSTOMERS, OGLE_DRIFT_TAG, "d1"),
            TagAction(DS_CUSTOMERS, "urn:li:tag:custom", "d2"),
        ]
    )
    result = apply(plan, backend)
    assert backend.write_calls == 1
    assert backend.tags[DS_CUSTOMERS] == {OGLE_DRIFT_TAG, "urn:li:tag:custom"}
    assert len(result.applied) == 2


def test_apply_records_read_failure_without_writing():
    """A backend read failure MUST NOT trigger a write (we could overwrite good tags)."""
    backend = FakeWritebackBackend(fail_on_read={DS_CUSTOMERS})
    plan = plan_writeback([_finding(DS_CUSTOMERS)])
    result = apply(plan, backend)
    assert len(result.failed) == 1
    assert result.failed[0].entity_urn == DS_CUSTOMERS
    assert backend.write_calls == 0


def test_apply_records_write_failure_and_continues_batch():
    """A write failure on one entity doesn't abort the batch."""
    backend = FakeWritebackBackend(fail_on_write={DS_CUSTOMERS})
    plan = WritebackPlan(
        actions=[
            TagAction(DS_CUSTOMERS, OGLE_DRIFT_TAG, "d"),
            TagAction(MODEL_CHURN, OGLE_DRIFT_TAG, "d"),
        ]
    )
    result = apply(plan, backend)
    assert len(result.failed) == 1
    assert result.failed[0].entity_urn == DS_CUSTOMERS
    assert MODEL_CHURN in backend.tags  # second action landed


def test_apply_empty_plan_is_no_op():
    backend = FakeWritebackBackend()
    result = apply(WritebackPlan(), backend)
    assert result.applied == []
    assert result.unchanged == []
    assert result.failed == []
    assert backend.write_calls == 0


def test_writeback_result_tagged_entities_is_deduped_and_ordered():
    """tagged_entities preserves first-write order, no dupes even if same URN was tagged
    with multiple tag URNs."""
    backend = FakeWritebackBackend()
    plan = WritebackPlan(
        actions=[
            TagAction(DS_CUSTOMERS, OGLE_DRIFT_TAG, ""),
            TagAction(DS_CUSTOMERS, "urn:li:tag:custom", ""),
            TagAction(MODEL_CHURN, OGLE_DRIFT_TAG, ""),
        ]
    )
    result = apply(plan, backend)
    assert result.tagged_entities == [DS_CUSTOMERS, MODEL_CHURN]


def test_writeback_result_to_dict_shape():
    backend = FakeWritebackBackend()
    plan = plan_writeback([_finding(DS_CUSTOMERS)], _walk({DS_CUSTOMERS: [MODEL_CHURN]}))
    result = apply(plan, backend)
    dumped = result.to_dict()
    assert set(dumped) == {"applied", "unchanged", "failed", "tagged_entities"}
    assert set(dumped["tagged_entities"]) == {DS_CUSTOMERS, MODEL_CHURN}


# =======================================================================================
# End-to-end: findings -> plan -> apply -> tags visible on entities
# =======================================================================================


def test_full_writeback_flow_end_to_end():
    """The whole outbound story: findings in, tags on datasets + models out."""
    findings = [
        _finding(DS_CUSTOMERS, kind=DriftKind.SCHEMA, severity=Severity.HIGH),
        _finding(DS_ORDERS, kind=DriftKind.VOLUME, severity=Severity.MEDIUM),
    ]
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN], DS_ORDERS: [MODEL_CHURN, MODEL_DEMAND]})
    backend = FakeWritebackBackend()

    plan = plan_writeback(findings, walk)
    result = apply(plan, backend)

    # Every affected entity got the tag once.
    assert backend.tags[DS_CUSTOMERS] == {OGLE_DRIFT_TAG}
    assert backend.tags[DS_ORDERS] == {OGLE_DRIFT_TAG}
    assert backend.tags[MODEL_CHURN] == {OGLE_DRIFT_TAG}
    assert backend.tags[MODEL_DEMAND] == {OGLE_DRIFT_TAG}
    # And churn wasn't tagged twice even though both drifted datasets feed it.
    assert result.applied.count(TagAction(MODEL_CHURN, OGLE_DRIFT_TAG, "downstream of drifted " + DS_CUSTOMERS)) == 1


# =======================================================================================
# plan_retract / apply_retract — the write-side inverse (clear the flag when drift heals)
# =======================================================================================
from ogle.writeback import (  # noqa: E402
    all_ogle_tag_urns,
    apply_retract,
    plan_retract,
    severity_tag_urn as _sev_tag,
)

_SEV_HIGH = _sev_tag(Severity.HIGH)
_SEV_MED = _sev_tag(Severity.MEDIUM)


def test_retract_empty_when_nothing_recovered():
    plan = plan_retract([], active_findings=[_finding(DS_ORDERS)])
    assert isinstance(plan, WritebackPlan)
    assert plan.actions == []


def test_all_ogle_tag_urns_covers_flat_plus_every_severity():
    urns = all_ogle_tag_urns()
    assert OGLE_DRIFT_TAG in urns
    assert {_sev_tag("high"), _sev_tag("medium"), _sev_tag("low"), _sev_tag("unknown")} <= urns


def test_retract_plans_flat_and_all_severity_tags_by_default():
    plan = plan_retract([DS_CUSTOMERS])
    removed = {a.tag_urn for a in plan.actions if a.entity_urn == DS_CUSTOMERS}
    assert removed == all_ogle_tag_urns()


def test_retract_flat_only_when_severity_tags_false():
    plan = plan_retract([DS_CUSTOMERS], severity_tags=False)
    removed = {a.tag_urn for a in plan.actions if a.entity_urn == DS_CUSTOMERS}
    assert removed == {OGLE_DRIFT_TAG}


def test_retract_skips_dataset_still_drifting():
    """A URN that shows up in recovered AND still-drifting is never retracted."""
    plan = plan_retract([DS_CUSTOMERS, DS_ORDERS], active_findings=[_finding(DS_ORDERS)])
    entities = {a.entity_urn for a in plan.actions}
    assert DS_CUSTOMERS in entities
    assert DS_ORDERS not in entities


def test_retract_clears_downstream_model_when_fully_recovered():
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN]})
    plan = plan_retract([DS_CUSTOMERS], walk_result=walk)
    entities = {a.entity_urn for a in plan.actions}
    assert entities == {DS_CUSTOMERS, MODEL_CHURN}


def test_retract_protects_model_still_downstream_of_active_drift():
    """MODEL_CHURN is fed by both a recovered and a still-drifting dataset -> keep its flag."""
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN], DS_ORDERS: [MODEL_CHURN, MODEL_DEMAND]})
    plan = plan_retract(
        [DS_CUSTOMERS], active_findings=[_finding(DS_ORDERS)], walk_result=walk
    )
    entities = {a.entity_urn for a in plan.actions}
    assert DS_CUSTOMERS in entities        # recovered dataset cleared
    assert MODEL_CHURN not in entities     # still downstream of drifting DS_ORDERS -> protected
    assert MODEL_DEMAND not in entities     # not downstream of any recovered dataset


def test_apply_retract_removes_present_tag_and_leaves_others():
    backend = FakeWritebackBackend(tags={DS_CUSTOMERS: {OGLE_DRIFT_TAG, "urn:li:tag:pii"}})
    plan = plan_retract([DS_CUSTOMERS], severity_tags=False)
    result = apply_retract(plan, backend)
    # Ogle's tag stripped; the unrelated pii tag survives.
    assert backend.tags[DS_CUSTOMERS] == {"urn:li:tag:pii"}
    assert [a.tag_urn for a in result.applied] == [OGLE_DRIFT_TAG]


def test_apply_retract_idempotent_when_tag_absent():
    """Re-running on an already-clean entity writes nothing and reports unchanged."""
    backend = FakeWritebackBackend(tags={DS_CUSTOMERS: {"urn:li:tag:pii"}})
    plan = plan_retract([DS_CUSTOMERS], severity_tags=False)
    result = apply_retract(plan, backend)
    assert result.applied == []
    assert [a.tag_urn for a in result.unchanged] == [OGLE_DRIFT_TAG]
    assert backend.write_calls == 0  # no write when there's nothing of ours to remove
    assert backend.tags[DS_CUSTOMERS] == {"urn:li:tag:pii"}


def test_apply_retract_read_failure_strands_entity_not_batch():
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN]})
    backend = FakeWritebackBackend(
        tags={DS_CUSTOMERS: {OGLE_DRIFT_TAG}, MODEL_CHURN: {OGLE_DRIFT_TAG}},
        fail_on_read={DS_CUSTOMERS},
    )
    plan = plan_retract([DS_CUSTOMERS], walk_result=walk, severity_tags=False)
    result = apply_retract(plan, backend)
    failed_entities = {a.entity_urn for a in result.failed}
    applied_entities = {a.entity_urn for a in result.applied}
    assert failed_entities == {DS_CUSTOMERS}     # unreadable entity stranded
    assert applied_entities == {MODEL_CHURN}      # the rest of the batch still cleared
    assert backend.tags[DS_CUSTOMERS] == {OGLE_DRIFT_TAG}   # never blind-written


def test_apply_retract_write_failure_recorded():
    backend = FakeWritebackBackend(
        tags={DS_CUSTOMERS: {OGLE_DRIFT_TAG}}, fail_on_write={DS_CUSTOMERS}
    )
    plan = plan_retract([DS_CUSTOMERS], severity_tags=False)
    result = apply_retract(plan, backend)
    assert {a.entity_urn for a in result.failed} == {DS_CUSTOMERS}
    assert result.applied == []


def test_retract_round_trips_a_full_write_then_clear():
    """Tag with plan_writeback+apply, then heal with plan_retract+apply_retract -> clean."""
    findings = [_finding(DS_CUSTOMERS, severity=Severity.HIGH)]
    walk = _walk({DS_CUSTOMERS: [MODEL_CHURN]})
    backend = FakeWritebackBackend()

    apply(plan_writeback(findings, walk, severity_tags=True), backend)
    assert OGLE_DRIFT_TAG in backend.tags[DS_CUSTOMERS]
    assert _SEV_HIGH in backend.tags[DS_CUSTOMERS]

    apply_retract(plan_retract([DS_CUSTOMERS], walk_result=walk), backend)
    assert backend.tags[DS_CUSTOMERS] == set()
    assert backend.tags[MODEL_CHURN] == set()
