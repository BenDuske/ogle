# Task #2 — Synthetic ML lineage on showcase-ecommerce

## Why

The DataHub team's `showcase-ecommerce` datapack is BI lineage only: 67 datasets,
23 dataJobs, 12 charts, 3 dashboards — **zero** ML entities (grep audit found no
`mlModel` / `mlFeature` / `mlFeatureTable`). Ogle targets the **Production ML Agents**
track, so the lineage walker needs real ML entities to walk.

Rather than fabricate a whole synthetic dataset (throws away the pack's credibility)
or pivot to the Wildcard track (less differentiated, still needs new data), Task #2
injects a small ML layer that hangs off the existing showcase datasets. Judges get
the "familiar showcase pack + real ML lineage" combo, and Ogle gets a graph that
looks like a real production stack.

## What lands in the graph

```
             (feast)
customers   ─┐                                         (mlflow)                  (sagemaker)
orders      ─┼─▶ customer_purchase_features ──▶ churn_predictor ──▶ churn_predictor_endpoint
order_items ─┤        (5 features)                (xgboost, v3)          (IN_SERVICE)
product_categories ─┘

products    ─┐
order_items ─┼─▶ product_demand_features ─┐
inventories ─┤        (5 features)          │
warehouses  ─┤                              ├─▶ demand_forecast
promotions  ─┘                              │      (lightgbm, v2)
                                            │
orders      ─┐                              │
order_items ─┼─▶ order_risk_features    ────┘
addresses   ─┤        (5 features)
customers   ─┘
```

- **3 mlFeatureTable** (platform: `feast`) — `customer_purchase_features`,
  `product_demand_features`, `order_risk_features`.
- **15 mlFeature** (5 per table) — every feature's `sources` field pins the real
  showcase dataset URNs it derives from, so `mlFeature → dataset` lineage is
  established at the entity level, not stitched.
- **2 mlModel** (platform: `mlflow`) — `churn_predictor` (xgboost, v3) consuming
  the customer table; `demand_forecast` (lightgbm, v2) consuming BOTH the product
  and risk tables (10 features total — deliberate: a wider surface makes the
  drift-detection story more compelling).
- **1 mlModelDeployment** (platform: `sagemaker`) — `churn_predictor_endpoint`
  (status: IN_SERVICE).

All entities carry an `ownership` aspect (`ml-platform` team = technical owner).

## URN shape notes (bit us during Task #2, pinned here)

- `MLFeatureTableKey` = `(platform, name)` — **2 parts**, no env. A 3-part
  `(platform, name, env)` URN was accepted silently but never indexed in search
  (ghost entity, `total: 0`). Fixed by dropping `,PROD`. The `MLFeatureTable`
  URN in this pack is therefore:
  `urn:li:mlFeatureTable:(urn:li:dataPlatform:feast,ogle_demo.customer_purchase_features)`.
- `MLModelKey` and `MLModelDeploymentKey` are `(platform, name, origin)` —
  **3 parts**, env goes in `origin` (`PROD`).
- `mlFeatureKey` = `(featureNamespace, name)` — **2 parts**, no platform, no
  env. Feature namespace = the feature-table name.

## Verify

```bash
# From the workspace with DataHub v1.6.0 quickstart running:
py -3.11 scripts/inject-ml-lineage.py            # emit
py -3.11 scripts/inject-ml-lineage.py --dry-run  # print 27 MCPs without emitting
```

GraphQL sanity checks (login `datahub` / `datahub`):

```graphql
query {
  mft: search(input:{type:MLFEATURE_TABLE, query:"*", start:0, count:100}) { total }  # → 3
  mf:  search(input:{type:MLFEATURE,       query:"*", start:0, count:100}) { total }  # → 15
  mlm: search(input:{type:MLMODEL,         query:"*", start:0, count:100}) { total }  # → 2
}
```

Model → feature → dataset walk (`http://localhost:9002/mlModels`):

- `churn_predictor` lists 5 `mlFeatures`.
- Each `mlFeature` (`.properties.sources`) lists 3–4 dbt dataset URNs.
- Deployment `churn_predictor_endpoint` is reachable via the DataHub SDK
  (`DataHubGraph.get_aspect(urn, MLModelDeploymentPropertiesClass)`). OSS
  GraphQL doesn't expose `MLModelDeployment` as a top-level `Entity` type;
  Ogle's walker uses the SDK path.

## Idempotent

DataHub upserts by URN + aspect. Re-running the script overwrites the same
aspects and produces no duplicates. Safe to rerun after any showcase reingest.

## What this unlocks (W2 preview)

- **Signature computation:** each `mlFeature.sources` dataset gets a lightweight
  row-count + schema-hash signature on a schedule.
- **Anomaly scoring:** compare current signature to the last-known baseline
  (persisted in Aegis's memory store) — flag drift above threshold.
- **Narrative alert:** LLM (qwen-plus via OpenAI-compatible endpoint) writes a
  30-second-actionable alert citing the exact model → feature → dataset that
  drifted, using DataHub ownership metadata to name a human.
- **Tag write-back (W3):** Ogle marks the affected `mlModel` and
  `mlFeatureTable` with a `datahub:tag:ogle/drift-flagged` tag so the next
  DataHub visitor inherits the finding.
