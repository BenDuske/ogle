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
    DataHubBackend,
    WalkResult,
    build_signature_from_aspects,
    dataset_urns_for_model,
    extract_owner_names,
    is_model_serving,
    owner_display_name,
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
class FakeQuantile:
    quantile: Optional[str]  # DataHub reports the level as a numeric string ("0.25")
    value: Optional[str]     # ...and the value at that level as a numeric string too


@dataclass
class FakeFieldProfile:
    fieldPath: str
    nullProportion: Optional[float]
    uniqueProportion: Optional[float] = None
    mean: Optional[str] = None  # DataHub reports mean as a numeric string
    stdev: Optional[str] = None  # DataHub reports stdev as a numeric string too
    min: Optional[str] = None    # DataHub reports min/max as numeric strings too
    max: Optional[str] = None
    quantiles: Optional[List["FakeQuantile"]] = None


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
class FakeOwner:
    owner: str  # a corpuser/corpGroup URN, mirroring OwnerClass.owner


@dataclass
class FakeOwnership:
    owners: List[FakeOwner] = field(default_factory=list)


@dataclass
class FakeBackend:
    model_props: Dict[str, FakeModelProps] = field(default_factory=dict)
    feature_props: Dict[str, FakeFeatureProps] = field(default_factory=dict)
    deployment_props: Dict[str, FakeDeploymentProps] = field(default_factory=dict)
    schemas: Dict[str, FakeSchema] = field(default_factory=dict)
    profiles: Dict[str, FakeProfile] = field(default_factory=dict)
    ownership: Dict[str, FakeOwnership] = field(default_factory=dict)

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

    def get_ownership(self, urn):
        return self.ownership.get(urn)


@dataclass
class LegacyBackend(FakeBackend):
    """A backend from before ownership support — deliberately hides `get_ownership` to
    prove `walk_model` tolerates its absence (getattr probe) rather than raising."""

    def __getattribute__(self, name):
        if name == "get_ownership":
            raise AttributeError(name)
        return super().__getattribute__(name)


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
        ownership={
            # customers owned by a person; orders by a team + the same person twice
            # (two ownership types) to exercise dedup.
            DS_CUSTOMERS: FakeOwnership(owners=[FakeOwner("urn:li:corpuser:jane.doe")]),
            DS_ORDERS: FakeOwnership(
                owners=[
                    FakeOwner("urn:li:corpGroup:data-eng"),
                    FakeOwner("urn:li:corpuser:jane.doe"),
                    FakeOwner("urn:li:corpuser:jane.doe"),  # dup -> collapses
                ]
            ),
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


def test_build_signature_folds_unique_fractions():
    """uniqueProportion on a field profile lands in field_unique_fractions."""
    profile = FakeProfile(
        rowCount=1000,
        fieldProfiles=[
            FakeFieldProfile("id", 0.0, uniqueProportion=1.0),
            FakeFieldProfile("region", 0.02, uniqueProportion=0.35),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_unique_fractions == {"id": pytest.approx(1.0), "region": pytest.approx(0.35)}


def test_build_signature_unique_fraction_absent_yields_empty():
    """Older profiles carry nullProportion but no uniqueProportion — degrade to empty."""
    profile = FakeProfile(rowCount=10, fieldProfiles=[FakeFieldProfile("id", 0.0)])
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_null_fractions == {"id": pytest.approx(0.0)}
    assert sig.field_unique_fractions == {}


def test_build_signature_skips_unique_fractions_out_of_range():
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            FakeFieldProfile("ok", 0.0, uniqueProportion=0.5),
            FakeFieldProfile("bad_high", 0.0, uniqueProportion=1.4),
            FakeFieldProfile("bad_neg", 0.0, uniqueProportion=-0.2),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_unique_fractions) == {"ok"}


def test_build_signature_folds_means():
    """`mean` on a field profile (a numeric string) lands in field_means as a float."""
    profile = FakeProfile(
        rowCount=1000,
        fieldProfiles=[
            FakeFieldProfile("amount", 0.0, mean="42.5"),
            FakeFieldProfile("pnl", 0.0, mean="-3.0"),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_means == {"amount": pytest.approx(42.5), "pnl": pytest.approx(-3.0)}


def test_build_signature_mean_absent_yields_empty():
    """Text/categorical columns (no mean) and older profiles degrade to an empty map."""
    profile = FakeProfile(rowCount=10, fieldProfiles=[FakeFieldProfile("region", 0.0)])
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_means == {}


def test_build_signature_skips_non_finite_and_junk_means():
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            FakeFieldProfile("ok", 0.0, mean="1.5"),
            FakeFieldProfile("nan", 0.0, mean="nan"),
            FakeFieldProfile("inf", 0.0, mean="inf"),
            FakeFieldProfile("junk", 0.0, mean="not-a-number"),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_means) == {"ok"}


def test_build_signature_folds_quantiles():
    """`quantiles` ({quantile, value} numeric strings) land sorted in field_quantiles."""
    profile = FakeProfile(
        rowCount=1000,
        fieldProfiles=[
            FakeFieldProfile(
                "amount",
                0.0,
                quantiles=[
                    FakeQuantile("0.75", "30"),
                    FakeQuantile("0.25", "10"),
                    FakeQuantile("0.5", "20"),
                ],
            ),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_quantiles == {
        "amount": ((0.25, 10.0), (0.5, 20.0), (0.75, 30.0))
    }


def test_build_signature_quantiles_absent_yields_empty():
    """Text/categorical columns (no quantiles) and older profiles degrade to an empty map."""
    profile = FakeProfile(rowCount=10, fieldProfiles=[FakeFieldProfile("region", 0.0)])
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_quantiles == {}


def test_build_signature_skips_junk_quantile_points_and_thin_sets():
    """A junk (unparseable / out-of-range level) point is dropped; a set left with < 2 usable
    points is dropped entirely rather than half-recorded."""
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            # one junk value + one out-of-range level -> only one clean point survives -> dropped
            FakeFieldProfile(
                "thin",
                0.0,
                quantiles=[
                    FakeQuantile("0.5", "nope"),
                    FakeQuantile("1.5", "3"),
                    FakeQuantile("0.9", "9"),
                ],
            ),
            # two clean points survive a junk third -> kept
            FakeFieldProfile(
                "kept",
                0.0,
                quantiles=[
                    FakeQuantile("0.25", "1"),
                    FakeQuantile("0.75", "3"),
                    FakeQuantile(None, "5"),
                ],
            ),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_quantiles) == {"kept"}
    assert sig.field_quantiles["kept"] == ((0.25, 1.0), (0.75, 3.0))


def test_build_signature_folds_stdevs():
    """`stdev` on a field profile (a numeric string) lands in field_stdevs as a float."""
    profile = FakeProfile(
        rowCount=1000,
        fieldProfiles=[
            FakeFieldProfile("amount", 0.0, mean="42.5", stdev="12.5"),
            FakeFieldProfile("const", 0.0, mean="7.0", stdev="0.0"),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_stdevs == {"amount": pytest.approx(12.5), "const": pytest.approx(0.0)}


def test_build_signature_stdev_absent_yields_empty():
    """Text/categorical columns (no stdev) and older profiles degrade to an empty map."""
    profile = FakeProfile(rowCount=10, fieldProfiles=[FakeFieldProfile("region", 0.0)])
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_stdevs == {}


def test_build_signature_skips_non_finite_junk_and_negative_stdevs():
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            FakeFieldProfile("ok", 0.0, stdev="1.5"),
            FakeFieldProfile("nan", 0.0, stdev="nan"),
            FakeFieldProfile("inf", 0.0, stdev="inf"),
            FakeFieldProfile("junk", 0.0, stdev="not-a-number"),
            FakeFieldProfile("neg", 0.0, stdev="-2.0"),  # a stdev < 0 is nonsense, skipped
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_stdevs) == {"ok"}


def test_build_signature_folds_mins_maxes():
    """`min`/`max` on a field profile (numeric strings, possibly signed) land as floats."""
    profile = FakeProfile(
        rowCount=1000,
        fieldProfiles=[
            FakeFieldProfile("amount", 0.0, min="-5.0", max="999.5"),
            FakeFieldProfile("id", 0.0, min="1", max="1000"),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_mins == {"amount": pytest.approx(-5.0), "id": pytest.approx(1.0)}
    assert sig.field_maxes == {"amount": pytest.approx(999.5), "id": pytest.approx(1000.0)}


def test_build_signature_min_max_absent_yields_empty():
    """Text/categorical columns (no min/max) and older profiles degrade to empty maps."""
    profile = FakeProfile(rowCount=10, fieldProfiles=[FakeFieldProfile("region", 0.0)])
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert sig.field_mins == {}
    assert sig.field_maxes == {}


def test_build_signature_skips_non_finite_and_junk_min_max():
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            FakeFieldProfile("ok", 0.0, min="0.0", max="10.0"),
            FakeFieldProfile("nan", 0.0, min="nan", max="5.0"),
            FakeFieldProfile("inf", 0.0, min="0.0", max="inf"),
            FakeFieldProfile("junk", 0.0, min="lo", max="hi"),
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_mins) == {"ok"}
    assert set(sig.field_maxes) == {"ok"}


def test_build_signature_drops_inverted_min_max_pair():
    """A profile reporting min > max is nonsense; the pair is dropped, not fatal to the walk."""
    profile = FakeProfile(
        rowCount=1,
        fieldProfiles=[
            FakeFieldProfile("good", 0.0, min="0.0", max="10.0"),
            FakeFieldProfile("bad", 0.0, min="10.0", max="5.0"),  # inverted -> dropped
        ],
    )
    sig = build_signature_from_aspects("urn:li:dataset:x", None, profile)
    assert set(sig.field_mins) == {"good"}
    assert set(sig.field_maxes) == {"good"}


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


# =======================================================================================
# Ownership — "who to page" attribution surfaced through the live walk
# =======================================================================================


def test_owner_display_name_strips_corpuser_and_corpgroup_urns():
    assert owner_display_name("urn:li:corpuser:jane.doe") == "jane.doe"
    assert owner_display_name("urn:li:corpGroup:data-eng") == "data-eng"
    # A bare (non-URN) string is returned stripped, not mangled.
    assert owner_display_name("  ops-team  ") == "ops-team"
    # Empty/degenerate never raises.
    assert owner_display_name("") == ""
    assert owner_display_name("urn:li:corpuser:") == "urn:li:corpuser:"


def test_extract_owner_names_dedups_and_preserves_order():
    ownership = FakeOwnership(
        owners=[
            FakeOwner("urn:li:corpGroup:data-eng"),
            FakeOwner("urn:li:corpuser:jane.doe"),
            FakeOwner("urn:li:corpuser:jane.doe"),  # dup
        ]
    )
    assert extract_owner_names(ownership) == ["data-eng", "jane.doe"]


def test_extract_owner_names_handles_none_and_empty():
    assert extract_owner_names(None) == []
    assert extract_owner_names(FakeOwnership(owners=[])) == []
    # An owner entry with no `.owner` is skipped, not crashed on.
    assert extract_owner_names(FakeOwnership(owners=[FakeOwner(owner=None)])) == []  # type: ignore[arg-type]


def test_walk_model_populates_owners_per_dataset():
    result = walk_model(_task2_backend(), MODEL_CHURN)
    assert result.owners[DS_CUSTOMERS] == ["jane.doe"]
    # data-eng leads (first-seen), jane.doe deduped from the two ownership entries.
    assert result.owners[DS_ORDERS] == ["data-eng", "jane.doe"]


def test_walk_model_owner_map_omits_unowned_datasets():
    b = _task2_backend()
    del b.ownership[DS_ORDERS]  # orders now unowned
    result = walk_model(b, MODEL_CHURN)
    assert DS_CUSTOMERS in result.owners
    assert DS_ORDERS not in result.owners  # cleanly omitted, not an empty list


def test_walk_model_tolerates_backend_without_get_ownership():
    """A pre-ownership custom backend degrades to no owners, never AttributeError."""
    b = LegacyBackend(**_task2_backend().__dict__)
    result = walk_model(b, MODEL_CHURN)
    # Walk still succeeds; owners simply empty.
    assert {s.urn for s in result.signatures} == {DS_CUSTOMERS, DS_ORDERS}
    assert result.owners == {}


def test_walk_models_unions_owners_across_walks():
    """A dataset owned in two walks lists each owner once, first-seen order."""
    left = WalkResult(owners={DS_ORDERS: ["data-eng", "jane.doe"]})
    right = WalkResult(owners={DS_ORDERS: ["jane.doe", "ml-platform"], DS_PRODUCTS: ["sku-team"]})
    merged = left.merge(right)
    assert merged.owners[DS_ORDERS] == ["data-eng", "jane.doe", "ml-platform"]
    assert merged.owners[DS_PRODUCTS] == ["sku-team"]


def test_owners_flow_into_the_narrative_who_to_page_line():
    """End-to-end: a live-shaped walk's owners reach the rendered incident."""
    from ogle.pipeline import run_drift_check
    from ogle.store import BaselineStore

    healthy = walk_model(_task2_backend(), MODEL_CHURN)
    store = BaselineStore()
    run_drift_check(store, healthy.signatures, serving_urns=healthy.serving_dataset_urns)

    b = _task2_backend()
    b.profiles[DS_CUSTOMERS] = FakeProfile(rowCount=0)  # collapse -> HIGH drift
    broken = walk_model(b, MODEL_CHURN)
    report = run_drift_check(
        store,
        broken.signatures,
        serving_urns=broken.serving_dataset_urns,
        owners=broken.owners,
    )
    assert report.should_alert is True
    # The owner reaches the incident and the rendered "who to page" line.
    assert report.incident.owners.get(DS_CUSTOMERS) == ["jane.doe"]
    assert "jane.doe" in report.narrative


# =======================================================================================
# DataHubBackend — live-adapter wiring.
#
# The `acryl-datahub` SDK is NOT installed at test time, so we stub
# `datahub.metadata.schema_classes` in sys.modules and inject a fake `DataHubGraph`. This
# pins the aspect-type -> accessor contract, the DatasetProfile *timeseries* special-case
# (get_latest_timeseries_value, NOT get_aspect), and the serving-filter in
# discover_deployed_models — none of which had any coverage, and all of which break
# silently against a real DataHub if the mapping drifts.
# =======================================================================================


@pytest.fixture
def stub_datahub_sdk(monkeypatch):
    import sys
    import types

    schema = types.ModuleType("datahub.metadata.schema_classes")

    # Distinct sentinel classes so a test can assert *which* aspect type was requested.
    class DatasetProfileClass: ...

    class MLFeaturePropertiesClass: ...

    class MLModelDeploymentPropertiesClass: ...

    class MLModelPropertiesClass: ...

    class OwnershipClass: ...

    class SchemaMetadataClass: ...

    for name, cls in {
        "DatasetProfileClass": DatasetProfileClass,
        "MLFeaturePropertiesClass": MLFeaturePropertiesClass,
        "MLModelDeploymentPropertiesClass": MLModelDeploymentPropertiesClass,
        "MLModelPropertiesClass": MLModelPropertiesClass,
        "OwnershipClass": OwnershipClass,
        "SchemaMetadataClass": SchemaMetadataClass,
    }.items():
        setattr(schema, name, cls)

    # Parent packages must resolve too, or `from datahub.metadata.schema_classes import ...`
    # fails before it reaches the stub.
    monkeypatch.setitem(sys.modules, "datahub", types.ModuleType("datahub"))
    monkeypatch.setitem(sys.modules, "datahub.metadata", types.ModuleType("datahub.metadata"))
    monkeypatch.setitem(sys.modules, "datahub.metadata.schema_classes", schema)
    return schema


class RecordingGraph:
    """Fake `DataHubGraph`. Records every aspect call as (method, urn, aspect_type)."""

    def __init__(self, *, aspect_fn=None, aspect_return=None, timeseries_return=None, urns=None):
        self.calls = []
        self.filter_arg = None
        self._aspect_fn = aspect_fn
        self._aspect_return = aspect_return
        self._timeseries_return = timeseries_return
        self._urns = list(urns or [])

    def get_aspect(self, entity_urn, aspect_type):
        self.calls.append(("get_aspect", entity_urn, aspect_type))
        if self._aspect_fn is not None:
            return self._aspect_fn(entity_urn, aspect_type)
        return self._aspect_return

    def get_latest_timeseries_value(self, entity_urn, aspect_type, filter_criteria_map):
        self.calls.append(("get_latest_timeseries_value", entity_urn, aspect_type))
        self.filter_arg = filter_criteria_map
        return self._timeseries_return

    def get_urns_by_filter(self, entity_types):
        self.filter_arg = entity_types
        return list(self._urns)


def test_datahubbackend_maps_snapshot_aspects_to_get_aspect(stub_datahub_sdk):
    """Each non-timeseries accessor calls get_aspect with the matching sentinel class."""
    schema = stub_datahub_sdk
    sentinel = object()
    graph = RecordingGraph(aspect_return=sentinel)
    backend = DataHubBackend(graph=graph)

    urn = "urn:li:mlModel:(x,y,PROD)"
    cases = [
        (backend.get_model_props, schema.MLModelPropertiesClass),
        (backend.get_feature_props, schema.MLFeaturePropertiesClass),
        (backend.get_deployment_props, schema.MLModelDeploymentPropertiesClass),
        (backend.get_schema_metadata, schema.SchemaMetadataClass),
        (backend.get_ownership, schema.OwnershipClass),
    ]
    for accessor, expected_cls in cases:
        graph.calls.clear()
        assert accessor(urn) is sentinel
        assert graph.calls == [("get_aspect", urn, expected_cls)]


def test_datahubbackend_dataset_profile_uses_timeseries_not_get_aspect(stub_datahub_sdk):
    """DatasetProfile is a timeseries aspect: the SDK refuses get_aspect for it."""
    schema = stub_datahub_sdk
    snapshot = object()
    graph = RecordingGraph(timeseries_return=snapshot)
    backend = DataHubBackend(graph=graph)

    urn = "urn:li:dataset:(dbt,x.orders,PROD)"
    assert backend.get_dataset_profile(urn) is snapshot
    assert graph.calls == [("get_latest_timeseries_value", urn, schema.DatasetProfileClass)]
    # Empty filter map = "latest snapshot as-is"; the SDK crashes on None.
    assert graph.filter_arg == {}
    # And it must NOT fall back to the snapshot-aspect path.
    assert not any(c[0] == "get_aspect" for c in graph.calls)


def test_datahubbackend_discover_returns_only_serving_models(stub_datahub_sdk):
    """discover_deployed_models enumerates mlModels then keeps IN_SERVICE ones."""
    import types as _t

    schema = stub_datahub_sdk
    serving = "urn:li:mlModel:(mlflow,serving,PROD)"
    idle = "urn:li:mlModel:(mlflow,idle,PROD)"

    def aspect_fn(urn, aspect_type):
        if aspect_type is schema.MLModelPropertiesClass:
            deps = ["urn:li:mlModelDeployment:(sm,serving_ep,PROD)"] if urn == serving else []
            return _t.SimpleNamespace(deployments=deps)
        if aspect_type is schema.MLModelDeploymentPropertiesClass:
            return _t.SimpleNamespace(status=IN_SERVICE)
        return None

    graph = RecordingGraph(aspect_fn=aspect_fn, urns=[serving, idle])
    backend = DataHubBackend(graph=graph)

    assert backend.discover_deployed_models() == [serving]
    assert graph.filter_arg == ["mlModel"]  # enumerated the right entity type
