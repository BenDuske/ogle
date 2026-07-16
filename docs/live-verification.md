# Live end-to-end verification (DataHub quickstart)

Everything before this doc was proven with in-memory fakes (`FakeBackend`) and offline
signature files. This is the record of Ogle's **first full live run** against a real
DataHub GMS — the loop the judges will see:

```
live DataHub walk  ->  build signatures  ->  diff vs baselines  ->  narrate  ->  alert  ->  tag write-back
```

Run on **2026-07-16** against the DataHub **quickstart** (`v1.6.0`, GMS `http://localhost:8080`,
all containers healthy) with the demo ML lineage from `scripts/inject-ml-lineage.py` already
seeded (`ogle_demo.churn_predictor` IN_SERVICE, `ogle_demo.demand_forecast`).

## Environment

`acryl-datahub` needs a Python with prebuilt wheels — **3.12** works, **3.14 does not**
(pydantic-core has no 3.14 wheel and fails the source build). Use an isolated venv:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[datahub]"   # or: pip install "acryl-datahub>=1.6.0"
```

The pure pipeline (signature/scorer/store/narrative/pipeline) still needs **none** of this —
the SDK is only pulled in lazily inside `walker.DataHubBackend` / `writeback`.

## What was verified (all green)

1. **Connection + discovery** — `DataHubGraph.get_urns_by_filter(entity_types=["mlModel"])`
   returned the two seeded models; `--discover` correctly selected only the IN_SERVICE one
   (`churn_predictor`).
2. **Live walk → signatures** — walking from the deployed model out to its upstream datasets
   produced 4 `DatasetSignature`s from real `SchemaMetadata` aspects
   (`customers`, `orders`, `order_items`, `product_categories`).
3. **Store round-trip** — first run seeded 4 datasets (exit 0); a second identical walk
   diffed against the persisted baselines and reported **no false drift** (exit 0). Proves
   the baseline JSON round-trips real signatures and the debounce holds on live data.
4. **Drift alert path** — injecting a silent column **type change** on a serving-path dataset
   (`customers.credit_limit FLOAT -> TEXT`) made the next check fire a **HIGH** incident on a
   serving path, with the exact cause (`retyped ['credit_limit:FLOAT->TEXT'] [serving]`) and
   **exit code 1**. Reverting the type returned the check to healthy (exit 0).
5. **W3 tag write-back** — `--write-back` on the incident stamped
   `urn:li:tag:ogle-drift-flagged` onto both the drifted dataset **and** its downstream
   serving model (`churn_predictor`); a re-run was an idempotent no-op ("already tagged,
   skipped"), so a scheduled loop won't double-tag.
6. **Console safety** — the cp1252 Windows console never crashed on the emoji/arrow markers
   (the `_emit` encode-with-replace guard held); markers render fully on a UTF-8 terminal.

The graph was mutated only on the throwaway demo entities and **fully restored** afterward
(schema type reverted, test tags stripped, final check back to exit 0) — the quickstart is
left pristine for a live demo.

## Reproduce

```bash
# 0. quickstart up (datahub docker quickstart) + demo lineage seeded:
.venv/Scripts/python scripts/inject-ml-lineage.py --gms http://localhost:8080

# 1. seed baselines from the live graph:
.venv/Scripts/ogle check --gms http://localhost:8080 --discover --store live.json      # exit 0, seeds 4

# 2. (demo) inject a drift, then:
.venv/Scripts/ogle check --gms http://localhost:8080 --discover --store live.json --no-update --write-back
#    -> HIGH drift on the serving path, exit 1, tags written back
```
