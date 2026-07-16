# Screenshots — pre-drift baseline

Local DataHub OSS v1.6.0 with the Task #2 ML lineage injected on top of
`showcase-ecommerce`. Captured against `http://localhost:9002`. Cropped to remove
browser chrome + Windows taskbar (see `_crop.py`).

These are the healthy baseline the W2 scorer diffs against. When a W3 tag
write-back marks a drifted asset with `datahub:tag:ogle/drift-flagged`, it will
show up in these same views.

| # | File | What it shows |
|---|---|---|
| 00 | `00-search-ml-feast-mlflow.png` | Search "churn" filtered to Feast + MLflow platforms — Task #2 landing surface |
| 01 | `01-search-mlmodels.png` | Both models (`churn_predictor` + `demand_forecast`) filtered to `MLMODEL` |
| 02 | `02-churn-predictor-summary.png` | `churn_predictor` detail — Summary/Documentation/Lineage/Properties/Group/Features/Incidents tabs |
| **03** | **`03-churn-predictor-lineage.png`** | **Money shot** — 4 dbt datasets (customers, orders, order_items, product_categories) → `churn_predictor` (via 5 features) |
| 04 | `04-search-demand-forecast.png` | Search filtered to `demand_forecast` — 1/1 result |
| 05 | `05-demand-forecast-summary.png` | Second model detail |
| **06** | **`06-demand-forecast-lineage.png`** | **Money shot** — richer graph: 8 dbt datasets → 2 feature tables → `demand_forecast` (10 features) |
| 07 | `07-search-feature-tables.png` | All 3 `mlFeatureTable` entities |
| 08 | `08-feature-table-customer-purchase.png` | `customer_purchase_features` — Features tab (5 features w/ types & descriptions) |
| 09 | `09-feature-table-sources.png` | `customer_purchase_features` — Sources tab (4 upstream dbt datasets) |

## Regenerating

1. Ingest Task #2 into a fresh DataHub quickstart (see `docs/task-2-ml-lineage.md`).
2. Drive Chrome via `tools/desktop/` on Halcyon (`hybrid.py shot`, `input.ps1 click`, UI-TARS grounding).
3. Save raw 1920x1080 PNGs into this folder.
4. Run `py -3.14 _crop.py` — crops to `(0, 118, 1920, 1000)` in place.
