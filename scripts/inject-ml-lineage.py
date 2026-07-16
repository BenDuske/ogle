"""Inject a synthetic ML-lineage layer on top of the showcase-ecommerce datapack.

Why: showcase-ecommerce ships BI lineage only (datasets -> jobs -> charts -> dashboards).
Ogle targets the Production ML Agents track, so we need real mlFeatureTable / mlModel /
mlModelDeployment entities pointing at the real showcase datasets. That gives Ogle's
lineage walker something to walk and gives judges the "recognizable showcase pack + real
ML lineage" combo without hand-rolling a full dataset.

Idempotent by design: DataHub upserts by URN; re-running this script overwrites the same
aspects and never produces duplicates.

Wire diagram (see docs/task-2-ml-lineage.md for the full rationale):

  dbt.customers    ---\
  dbt.orders        --->  feast.customer_purchase_features  --> mlflow.churn_predictor  --> sagemaker.churn_predictor_endpoint
  dbt.order_items  ---/                                                                        (deployment)
  dbt.product_categories

  dbt.products        ---\
  dbt.order_items     --->  feast.product_demand_features  --\
  dbt.inventories     ---/                                     ---> mlflow.demand_forecast
  dbt.warehouses     ----/                                    /
  dbt.promotions     ----/                                   /
                                                            /
  dbt.orders          ---\                                 /
  dbt.order_items     --->  feast.order_risk_features    -/
  dbt.addresses       ---/
  dbt.customers       ---/
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import List

from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    AuditStampClass,
    DeploymentStatusClass,
    MLFeatureDataTypeClass,
    MLFeaturePropertiesClass,
    MLFeatureTablePropertiesClass,
    MLModelDeploymentPropertiesClass,
    MLModelPropertiesClass,
    OwnerClass,
    OwnershipClass,
    OwnershipTypeClass,
    VersionTagClass,
)

# -------- URN builders (kept in one place so drift is impossible) --------

SHOWCASE_PREFIX = "b2fd91"


def dataset_urn(platform: str, name: str, env: str = "PROD") -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:{platform},{SHOWCASE_PREFIX}.{name},{env})"


def feature_table_urn(name: str, env: str = "PROD") -> str:
    # MLFeatureTableKey has only (platform, name) — no env — so URN is 2-part.
    return f"urn:li:mlFeatureTable:(urn:li:dataPlatform:feast,ogle_demo.{name})"


def feature_urn(table: str, feature: str) -> str:
    return f"urn:li:mlFeature:(ogle_demo.{table},{feature})"


def model_urn(name: str, env: str = "PROD") -> str:
    return f"urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.{name},{env})"


def deployment_urn(name: str, env: str = "PROD") -> str:
    return f"urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,ogle_demo.{name},{env})"


def user_urn(name: str) -> str:
    return f"urn:li:corpuser:{name}"


# -------- Design (single source of truth for the ML layer) --------


@dataclass
class Feature:
    name: str
    data_type: str
    description: str


@dataclass
class FeatureTable:
    name: str
    description: str
    upstream_datasets: List[str]  # showcase dataset URNs
    features: List[Feature] = field(default_factory=list)
    owner: str = "ml-platform"


@dataclass
class Model:
    name: str
    description: str
    framework: str
    version: str
    upstream_feature_tables: List[str]  # feature-table URNs
    owner: str = "ml-platform"


@dataclass
class Deployment:
    name: str
    description: str
    model_urn: str
    status: str  # DeploymentStatusClass constant
    owner: str = "ml-platform"


FEATURE_TABLES: List[FeatureTable] = [
    FeatureTable(
        name="customer_purchase_features",
        description=(
            "Customer-level purchase behavior features for churn/next-basket models. "
            "Rolls per-order behavior into stable 30/90d customer signals."
        ),
        upstream_datasets=[
            dataset_urn("dbt", "order_entry_db.order_entry.customers"),
            dataset_urn("dbt", "order_entry_db.order_entry.orders"),
            dataset_urn("dbt", "order_entry_db.order_entry.order_items"),
            dataset_urn("dbt", "order_entry_db.order_entry.product_categories"),
        ],
        features=[
            Feature("customer_lifetime_value", "CONTINUOUS", "Sum of net order totals across history, USD."),
            Feature("avg_order_value_30d", "CONTINUOUS", "Mean net order total over the last 30 days."),
            Feature("orders_last_90d", "ORDINAL", "Count of distinct orders in the last 90 days."),
            Feature("days_since_last_order", "ORDINAL", "Days between now and the most recent order."),
            Feature("preferred_category_id", "NOMINAL", "Modal product category over the last 90 days."),
        ],
    ),
    FeatureTable(
        name="product_demand_features",
        description=(
            "Per-SKU demand-signal features for forecasting and stock planning. "
            "Blends recent unit velocity with warehouse coverage and promo state."
        ),
        upstream_datasets=[
            dataset_urn("dbt", "order_entry_db.order_entry.products"),
            dataset_urn("dbt", "order_entry_db.order_entry.order_items"),
            dataset_urn("dbt", "order_entry_db.order_entry.inventories"),
            dataset_urn("dbt", "order_entry_db.order_entry.warehouses"),
            dataset_urn("dbt", "order_entry_db.order_entry.promotions"),
        ],
        features=[
            Feature("units_sold_last_7d", "ORDINAL", "Total units sold across all warehouses, last 7 days."),
            Feature("units_sold_last_30d", "ORDINAL", "Total units sold across all warehouses, last 30 days."),
            Feature("avg_reorder_days", "CONTINUOUS", "Mean days between successive reorders per SKU."),
            Feature("warehouse_coverage_ratio", "CONTINUOUS", "Warehouses stocking SKU / total warehouses."),
            Feature("promo_active", "BINARY", "1 if an active promotion covers the SKU right now."),
        ],
    ),
    FeatureTable(
        name="order_risk_features",
        description=(
            "Per-order abuse/return-risk features. Blends order shape, first-order flag, "
            "and shipping-address change signal to feed downstream fraud/forecast models."
        ),
        upstream_datasets=[
            dataset_urn("dbt", "order_entry_db.order_entry.orders"),
            dataset_urn("dbt", "order_entry_db.order_entry.order_items"),
            dataset_urn("dbt", "order_entry_db.order_entry.addresses"),
            dataset_urn("dbt", "order_entry_db.order_entry.customers"),
        ],
        features=[
            Feature("order_total_z_score", "CONTINUOUS", "Order total normalized against customer historical mean/std."),
            Feature("address_change_flag", "BINARY", "1 if shipping address differs from previous order."),
            Feature("first_order_flag", "BINARY", "1 if this is the customer's first order."),
            Feature("items_count", "ORDINAL", "Number of line items on the order."),
            Feature("shipping_region_id", "NOMINAL", "Coarse geographic region identifier."),
        ],
    ),
]


MODELS: List[Model] = [
    Model(
        name="churn_predictor",
        description=(
            "Binary churn classifier. Predicts P(customer churns in next 30 days) from "
            "customer-purchase features. Ships weekly, evaluated on time-split holdout."
        ),
        framework="xgboost",
        version="v3",
        upstream_feature_tables=[feature_table_urn("customer_purchase_features")],
    ),
    Model(
        name="demand_forecast",
        description=(
            "Per-SKU 14-day demand regressor. LightGBM over product-demand + order-risk "
            "features. Feeds warehouse restock planning."
        ),
        framework="lightgbm",
        version="v2",
        upstream_feature_tables=[
            feature_table_urn("product_demand_features"),
            feature_table_urn("order_risk_features"),
        ],
    ),
]


DEPLOYMENTS: List[Deployment] = [
    Deployment(
        name="churn_predictor_endpoint",
        description="SageMaker real-time inference endpoint serving churn_predictor to the marketing platform.",
        model_urn=model_urn("churn_predictor"),
        status="IN_SERVICE",
    ),
]


# -------- Emit --------


def _owner_aspect(user: str) -> OwnershipClass:
    return OwnershipClass(
        owners=[OwnerClass(owner=user_urn(user), type=OwnershipTypeClass.TECHNICAL_OWNER)],
        lastModified=AuditStampClass(time=int(time.time() * 1000), actor=user_urn("datahub")),
    )


def _feature_data_type(name: str) -> str:
    return getattr(MLFeatureDataTypeClass, name)


def build_mcps() -> List[MetadataChangeProposalWrapper]:
    mcps: List[MetadataChangeProposalWrapper] = []

    for ft in FEATURE_TABLES:
        table_urn = feature_table_urn(ft.name)

        # Emit each feature entity first so the table can reference them.
        feature_urns: List[str] = []
        for f in ft.features:
            f_urn = feature_urn(ft.name, f.name)
            feature_urns.append(f_urn)
            mcps.append(
                MetadataChangeProposalWrapper(
                    entityUrn=f_urn,
                    aspect=MLFeaturePropertiesClass(
                        description=f.description,
                        dataType=_feature_data_type(f.data_type),
                        sources=ft.upstream_datasets,
                    ),
                )
            )

        # Feature table props: description + membership + upstream datasets as sources.
        mcps.append(
            MetadataChangeProposalWrapper(
                entityUrn=table_urn,
                aspect=MLFeatureTablePropertiesClass(
                    description=ft.description,
                    mlFeatures=feature_urns,
                    mlPrimaryKeys=[],
                ),
            )
        )
        mcps.append(
            MetadataChangeProposalWrapper(entityUrn=table_urn, aspect=_owner_aspect(ft.owner))
        )

    # Invert DEPLOYMENTS into model_urn -> [deployment_urn] so `MLModelProperties.deployments`
    # can be populated. Without this back-reference the Ogle walker can't tell that a model
    # is IN_SERVICE (bit us during Ogle W2c live-walk).
    deployments_by_model: Dict[str, List[str]] = {}
    for d in DEPLOYMENTS:
        deployments_by_model.setdefault(d.model_urn, []).append(deployment_urn(d.name))

    for m in MODELS:
        m_urn = model_urn(m.name)
        mcps.append(
            MetadataChangeProposalWrapper(
                entityUrn=m_urn,
                aspect=MLModelPropertiesClass(
                    description=m.description,
                    version=VersionTagClass(versionTag=m.version),
                    type=m.framework,
                    mlFeatures=[
                        feature_urn(ft.name, f.name)
                        for ft in FEATURE_TABLES
                        if feature_table_urn(ft.name) in m.upstream_feature_tables
                        for f in ft.features
                    ],
                    deployments=deployments_by_model.get(m_urn, []),
                    trainingMetrics=[],
                    hyperParams=[],
                ),
            )
        )
        mcps.append(MetadataChangeProposalWrapper(entityUrn=m_urn, aspect=_owner_aspect(m.owner)))

    for d in DEPLOYMENTS:
        d_urn = deployment_urn(d.name)
        mcps.append(
            MetadataChangeProposalWrapper(
                entityUrn=d_urn,
                aspect=MLModelDeploymentPropertiesClass(
                    description=d.description,
                    status=getattr(DeploymentStatusClass, d.status),
                ),
            )
        )
        mcps.append(MetadataChangeProposalWrapper(entityUrn=d_urn, aspect=_owner_aspect(d.owner)))

    return mcps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gms", default="http://localhost:8080", help="DataHub GMS URL")
    p.add_argument("--dry-run", action="store_true", help="Build MCPs but do not emit")
    args = p.parse_args()

    mcps = build_mcps()
    print(f"Built {len(mcps)} MCPs "
          f"({len(FEATURE_TABLES)} feature tables, "
          f"{sum(len(t.features) for t in FEATURE_TABLES)} features, "
          f"{len(MODELS)} models, "
          f"{len(DEPLOYMENTS)} deployment(s))")

    if args.dry_run:
        for mcp in mcps:
            print(f"  {mcp.entityUrn}  {type(mcp.aspect).__name__}")
        return 0

    emitter = DatahubRestEmitter(gms_server=args.gms)
    emitter.test_connection()

    ok = 0
    for mcp in mcps:
        emitter.emit(mcp)
        ok += 1
    print(f"Emitted {ok}/{len(mcps)} aspects to {args.gms}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
