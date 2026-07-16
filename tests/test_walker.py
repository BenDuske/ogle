"""Unit tests for ogle.walker.

Every test uses a `FakeBackend` — no `acryl-datahub` import required at test time. The
fake mirrors DataHub's aspect shapes closely enough (`.fields`, `.fieldPath`,
`.nativeDataType`, `.rowCount`, `.fieldProfiles[].nullProportion`, `.deployments`,
`.status`, `.mlFeatures`, `.sources`) that a bug in the walker's traversal or signature
build surfaces the same way it would against a live graph.

Fixtures mirror Task #2: `churn_predictor` (deployed via `churn_predictor_endpoint`)
consuming `customer_purchase_features` -> `customers` + `orders`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pytest

from ogle.walker import (
    IN_SERVICE,
    WalkResult,
    build_signature_from_aspects,
    dataset_urns_for_model,
    is_model_serving,
    walk_model,
    walk_models,
)

# ---- URNs (Task #2 shape) -------------------------------------------------------------
MODEL_CHURN = "urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.churn_predictor,PROD)"
MODEL_DEMAND = "urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.demand_forecast,PROD)"
DEPLOY_CHURN = "urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,ogle_demo.churn_predictor_endpoint,PROD)"

FEAT_CLV = "urn:li:mlFeature:(ogle_demo.customer_purchase_features,customer_lifetime_value)"
FEAT_ORDERS_90 = "urn:li:mlFeature:(ogle_demo.customer_purchase_features,orders_last_90d)"
FEAT_UNITS_7 = "urn:li:mlFeature:(ogle_demo.product_demand_features,units_sold_last_7d)"

DS_CUSTOMERS = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
DS_ORDERS = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"
DS_PRODUCTS = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.products,PROD)"
DS_ORPHAN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orphan_no_aspects,PROD)"


# ---- Fake aspect objects --------------------------------------------------------------
@dataclass
class FakeSchemaField:
    fieldPath: str
    nativeDataType: str


@dataclass
class FakeSchema:
    fields: List[FakeSchemaField] = field(default_factory=list)


@dataclass
class FakeFieldProfile:
    fieldPath: str
    nullProportion: Optional[float]


@dataclass
class FakeProfile:
    rowCount: Optional[int] = None
    fieldProfiles: List[FakeFieldProfile] = field(default_factory=list)


@dataclass
class FakeDeploymentProps:
    status: str = IN_SERVICE


@dataclass
class FakeFeatureProps:
    sources: List[str] = field(default_factory=list)


@dataclass
class FakeModelProps:
    mlFeatures: List[str] = field(default_factory=list)
    deployments: List[str] = field(default_factory=list)


@dataclass
class FakeBackend:
    model_props: Dict[str, FakeModelProps] = field(default_factory=dict)
    feature_props: Dict[str, FakeFeatureProps] = field(default_factory=dict)
    deployment_props: Dict[str, FakeDeploymentProps] = field(default_factory=dict)
    schemas: Dict[str, FakeSchema] = field(default_factory=dict)
    profiles: Dict[str, FakeProfile] = field(default_factory=dict)

    def get_model_props(self, urn):
        return self.model_props.get(urn)

    def get_feature_props(self, urn):
        return self.feature_props.get(urn)

    def get_deployment_props(self, urn):
        return self.deployment_props.get(urn)

    def get_schema_metadata(self, urn):
        return self.schemas.get(urn)

    def get_dataset_profile(self, urn):
        return self.profiles.get(urn)


def _task2_backend() -> FakeBackend:
    """Reproduces the Task #2 shape: churn_predictor deployed, consuming 2 features
    -> 2 datasets. Datasets carry a schema + profile."""
    return FakeBackend(
        model_props={
            MODEL_CHURN: FakeModelProps(
                mlFeatures=[FEAT_CLV, FEAT_ORDERS_90],
                deployments=[DEPLOY_CHURN],
            ),
        },
        feature_props={
            FEAT_CLV: FakeFeatureProps(sources=[DS_CUSTOMERS, DS_ORDERS]),
            FEAT_ORDERS_90: FakeFeatureProps(sources=[DS_ORDERS]),  # duplicate on purpose
        },
        deployment_props={DEPLOY_CHURN: FakeDeploymentProps(status=IN_SERVICE)},
        schemas={
            DS_CUSTOMERS: FakeSchema(
                fields=[
                    FakeSchemaField("id", "int"),
                    FakeSchemaField("email", "string"),
                ]
            ),
            DS_ORDERS: FakeSchema(
                fields=[FakeSchemaField("order_id", "int")]
            ),
        },
        profiles={
            DS_CUSTOMERS: FakeProfile(
                rowCount=1000,
                fieldProfiles=[FakeFieldProfile("email", 0.05)],
            ),
            DS_ORDERS: FakeProfile(rowCount=5000),
        },
    )


# =======================================================================================
# build_signature_from_aspects — pure fingerprint core
# =======================================================================================


def test_build_signature_folds_schema_and_profile():
    schema = FakeSchema(
        fields=[FakeSchemaField("id", "int"), FakeSchemaField("email", "string")]
    )
    profile = FakeProfile(
        rowCount=1000,
        fieldProfiles=[
            FakeFieldProfile("id", 0.0),
            FakeFieldProfile("email", 0.05),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", schema, profile)
    assert sig is not None
    assert sig.field_paths == frozenset({"id", "email"})
    assert sig.row_count == 1000
    assert sig.field_null_fractions["email"] == pytest.approx(0.05)


def test_build_signature_returns_none_when_both_aspects_absent():
    assert build_signature_from_aspects("urn:li:dataset:x", None, None) is None


def test_build_signature_ok_with_only_schema():
    schema = FakeSchema(fields=[FakeSchemaField("id", "int")])
    sig = build_signature_from_aspects("urn:li:dataset:x", schema, None)
    assert sig is not None
    assert sig.row_count is None
    assert sig.field_paths == frozenset({"id"})


def test_build_signature_ok_with_only_profile():
    profile = FakeProfile(rowCount=42)
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig is not None
    assert sig.schema_fields == ()
    assert sig.row_count == 42


def test_build_signature_skips_null_fractions_out_of_range():
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            FakeFieldProfile("ok", 0.3),
            FakeFieldProfile("bad_high", 1.2),
            FakeFieldProfile("bad_neg", -0.1),
            FakeFieldProfile("bad_none", None),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_null_fractions) == {"ok"}


def test_build_signature_skips_partial_schema_fields():
    """A field with no path or no nativeDataType is skipped, not defaulted."""
    schema = FakeSchema(
        fields=[
            FakeSchemaField("good", "int"),
            FakeSchemaField(fieldPath=None, nativeDataType="int"),  # type: ignore[arg-type]
            FakeSchemaField(fieldPath="also_bad", nativeDataType=None),  # type: ignore[arg-type]
        ]
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", schema, None)
    assert sig.field_paths == frozenset({"good"})


def test_build_signature_ignores_negative_row_count():
    """A profile aspect with invalid data still yields a signature (URN was seen); the
    invalid row_count just doesn't populate. Downstream scorer skips missing rowCount."""
    profile = FakeProfile(rowCount=-1)
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig is not None
    assert sig.row_count is None  # invalid input rejected
    assert sig.schema_fields == ()


def test_build_signature_carries_computed_at():
    schema = FakeSchema(fields=[FakeSchemaField("id", "int")])
    sig = build_signature_from_aspects("u", schema, None, computed_at="2026-07-16T18:00:00Z")
    assert sig.computed_at == "2026-07-16T18:00:00Z"


# =======================================================================================
# is_model_serving
# =======================================================================================


def test_is_model_serving_true_when_any_deployment_in_service():
    b = _task2_backend()
    assert is_model_serving(b, MODEL_CHURN) is True


def test_is_model_serving_false_when_no_deployment():
    b = _task2_backend()
    b.model_props[MODEL_CHURN] = FakeModelProps(mlFeatures=[FEAT_CLV], deployments=[])
    assert is_model_serving(b, MODEL_CHURN) is False


def test_is_model_serving_false_when_deployment_not_in_service():
    b = _task2_backend()
    b.deployment_props[DEPLOY_CHURN] = FakeDeploymentProps(status="OUT_OF_SERVICE")
    assert is_model_serving(b, MODEL_CHURN) is False


def test_is_model_serving_false_when_model_props_absent():
    b = FakeBackend()
    assert is_model_serving(b, MODEL_CHURN) is False


def test_is_model_serving_ignores_missing_deployment_aspect():
    """A dangling deployment URN (aspect not returned) is treated as unknown, not serving."""
    b = _task2_backend()
    b.deployment_props.pop(DEPLOY_CHURN)  # dangling
    assert is_model_serving(b, MODEL_CHURN) is False


# =======================================================================================
# dataset_urns_for_model
# =======================================================================================


def test_dataset_urns_dedup_across_features():
    """orders is a source of both features -> appears exactly once."""
    urns = dataset_urns_for_model(_task2_backend(), MODEL_CHURN)
    assert urns == [DS_CUSTOMERS, DS_ORDERS]  # first-seen order, no dup


def test_dataset_urns_empty_when_model_missing():
    assert dataset_urns_for_model(FakeBackend(), MODEL_CHURN) == []


def test_dataset_urns_skips_missing_feature_props():
    b = _task2_backend()
    b.feature_props.pop(FEAT_ORDERS_90)  # feature aspect missing
    # Only FEAT_CLV's sources come through
    urns = dataset_urns_for_model(b, MODEL_CHURN)
    assert urns == [DS_CUSTOMERS, DS_ORDERS]  # from FEAT_CLV alone


def test_dataset_urns_empty_when_no_features_listed():
    b = _task2_backend()
    b.model_props[MODEL_CHURN] = FakeModelProps(mlFeatures=[], deployments=[DEPLOY_CHURN])
    assert dataset_urns_for_model(b, MODEL_CHURN) == []


# =======================================================================================
# walk_model — the aggregate call the pipeline uses
# =======================================================================================


def test_walk_model_produces_signatures_and_serving_set():
    result = walk_model(_task2_backend(), MODEL_CHURN)
    urns = {s.urn for s in result.signatures}
    assert urns == {DS_CUSTOMERS, DS_ORDERS}
    assert result.serving_dataset_urns == {DS_CUSTOMERS, DS_ORDERS}
    assert MODEL_CHURN in result.walked_models
    assert result.skipped_urns == []


def test_walk_model_serving_urns_empty_when_not_deployed():
    b = _task2_backend()
    b.deployment_props[DEPLOY_CHURN] = FakeDeploymentProps(status="OUT_OF_SERVICE")
    result = walk_model(b, MODEL_CHURN)
    # Datasets still fingerprinted; just no severity escalation.
    assert {s.urn for s in result.signatures} == {DS_CUSTOMERS, DS_ORDERS}
    assert result.serving_dataset_urns == frozenset()


def test_walk_model_skips_datasets_without_aspects():
    """A source URN Ogle finds with neither schema nor profile is reported in skipped_urns."""
    b = _task2_backend()
    b.feature_props[FEAT_CLV] = FakeFeatureProps(sources=[DS_CUSTOMERS, DS_ORDERS, DS_ORPHAN])
    result = walk_model(b, MODEL_CHURN)
    assert DS_ORPHAN in result.skipped_urns
    assert DS_ORPHAN not in {s.urn for s in result.signatures}
    # The known datasets still land.
    assert {s.urn for s in result.signatures} == {DS_CUSTOMERS, DS_ORDERS}


def test_walk_model_empty_result_when_model_absent():
    result = walk_model(FakeBackend(), MODEL_CHURN)
    assert result.signatures == []
    assert result.serving_dataset_urns == frozenset()


def test_walk_model_computed_at_flows_into_signatures():
    result = walk_model(_task2_backend(), MODEL_CHURN, computed_at="2026-07-16T18:00:00Z")
    assert all(s.computed_at == "2026-07-16T18:00:00Z" for s in result.signatures)


# =======================================================================================
# walk_models — union across many models
# =======================================================================================


def _two_model_backend() -> FakeBackend:
    """churn_predictor (serving) shares `orders` with demand_forecast (not serving)."""
    b = _task2_backend()
    b.model_props[MODEL_DEMAND] = FakeModelProps(
        mlFeatures=[FEAT_UNITS_7],
        deployments=[],  # not deployed
    )
    b.feature_props[FEAT_UNITS_7] = FakeFeatureProps(sources=[DS_ORDERS, DS_PRODUCTS])
    b.schemas[DS_PRODUCTS] = FakeSchema(fields=[FakeSchemaField("sku", "string")])
    b.profiles[DS_PRODUCTS] = FakeProfile(rowCount=200)
    return b


def test_walk_models_deduplicates_shared_datasets():
    b = _two_model_backend()
    result = walk_models(b, [MODEL_CHURN, MODEL_DEMAND])
    urns = [s.urn for s in result.signatures]
    # orders appears in BOTH models -> one signature
    assert urns.count(DS_ORDERS) == 1
    assert set(urns) == {DS_CUSTOMERS, DS_ORDERS, DS_PRODUCTS}


def test_walk_models_unions_serving_set():
    """orders feeds a serving model AND a non-serving one -> still serving."""
    b = _two_model_backend()
    result = walk_models(b, [MODEL_CHURN, MODEL_DEMAND])
    # Serving set comes only from the serving model (churn) — orders should be in it,
    # products should NOT (only fed by non-serving demand_forecast).
    assert DS_ORDERS in result.serving_dataset_urns
    assert DS_CUSTOMERS in result.serving_dataset_urns
    assert DS_PRODUCTS not in result.serving_dataset_urns


def test_walk_models_walked_models_records_traversal():
    b = _two_model_backend()
    result = walk_models(b, [MODEL_CHURN, MODEL_DEMAND])
    assert set(result.walked_models) == {MODEL_CHURN, MODEL_DEMAND}


def test_walk_result_merge_first_signature_wins_on_dup_urn():
    """A dataset seen in two walks keeps the first walk's signature (deterministic)."""
    from ogle.signature import build_signature as bs

    sig_a = bs(DS_CUSTOMERS, schema_fields=[("id", "int")], row_count=1000)
    sig_b = bs(DS_CUSTOMERS, schema_fields=[("id", "int")], row_count=2000)  # would differ
    left = WalkResult(signatures=[sig_a])
    right = WalkResult(signatures=[sig_b])
    merged = left.merge(right)
    assert len(merged.signatures) == 1
    assert merged.signatures[0].row_count == 1000  # first-seen


# ---- dataset_to_models reverse index (feeds W3 writeback) -----------------------------
def test_walk_model_populates_dataset_to_models():
    result = walk_model(_task2_backend(), MODEL_CHURN)
    # Every upstream dataset lists the walked model as a downstream consumer.
    assert result.dataset_to_models[DS_CUSTOMERS] == [MODEL_CHURN]
    assert result.dataset_to_models[DS_ORDERS] == [MODEL_CHURN]


def test_walk_models_dataset_to_models_unions_across_walks():
    """Orders feeds churn AND demand -> reverse index lists both, dedup'd, order stable."""
    b = _two_model_backend()
    result = walk_models(b, [MODEL_CHURN, MODEL_DEMAND])
    assert set(result.dataset_to_models[DS_ORDERS]) == {MODEL_CHURN, MODEL_DEMAND}
    # Datasets fed by only one model list only that model.
    assert result.dataset_to_models[DS_CUSTOMERS] == [MODEL_CHURN]
    assert result.dataset_to_models[DS_PRODUCTS] == [MODEL_DEMAND]


# =======================================================================================
# End-to-end wiring with the pipeline — proves the output is drop-in
# =======================================================================================


def test_walker_output_feeds_pipeline_cleanly():
    """The whole point of the walker: its output IS what run_drift_check needs."""
    from ogle.pipeline import run_drift_check
    from ogle.store import BaselineStore

    result = walk_model(_task2_backend(), MODEL_CHURN)
    store = BaselineStore()
    report = run_drift_check(
        store,
        result.signatures,
        serving_urns=result.serving_dataset_urns,
    )
    # First run -> everything is new, nothing scored, no drift.
    assert set(report.new_urns) == {DS_CUSTOMERS, DS_ORDERS}
    assert report.findings == []
    assert report.should_alert is False


def test_walker_pipeline_flags_serving_escalation_on_second_run():
    """After seeding baselines, a volume collapse on a serving dataset should escalate."""
    from ogle.pipeline import run_drift_check
    from ogle.scorer import Severity
    from ogle.store import BaselineStore

    # First run: seed baselines from a healthy walk.
    healthy = walk_model(_task2_backend(), MODEL_CHURN)
    store = BaselineStore()
    run_drift_check(store, healthy.signatures, serving_urns=healthy.serving_dataset_urns)

    # Second run: DS_CUSTOMERS collapsed to 0 rows.
    b = _task2_backend()
    b.profiles[DS_CUSTOMERS] = FakeProfile(rowCount=0)
    broken = walk_model(b, MODEL_CHURN)

    report = run_drift_check(
        store,
        broken.signatures,
        serving_urns=broken.serving_dataset_urns,
    )
    assert report.should_alert is True
    assert report.incident.serving_impacted is True
    assert report.incident.overall_severity == Severity.HIGH
