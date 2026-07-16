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
