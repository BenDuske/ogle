# W2c — DataHub walker

The seam that connects Ogle's pure drift-detection pipeline to a live DataHub graph.

Everything else in Ogle (`signature`, `scorer`, `store`, `narrative`, `pipeline`) is pure:
it takes `DatasetSignature`s in and never touches a network. `ogle.walker` is the one
module that does — it walks DataHub from a deployed model out to its upstream datasets,
fetches the aspects it needs, and folds them into signatures the pipeline can score.

## Traversal

```
mlModelDeployment  (IN_SERVICE?)  <- serving flag
        |
    mlModel  (MLModelProperties.deployments, .mlFeatures)
        |
    mlFeature  (MLFeatureProperties.sources)
        |
    dataset  (SchemaMetadata, DatasetProfile)  -> DatasetSignature
```

**Serving detection** uses `MLModelProperties.deployments` -> each deployment's
`MLModelDeploymentProperties.status`. A model with *any* `IN_SERVICE` deployment marks
every one of its upstream datasets as serving, so downstream findings get bumped one
severity step by the scorer.

`walk_models([model_a, model_b])` unions across multiple models — a dataset feeding
model A (serving) and model B (not serving) is *still* on the serving path.

## Two-layer design

- **Pure core.** `build_signature_from_aspects`, `is_model_serving`,
  `dataset_urns_for_model`, `walk_model`, `walk_models` all take a `WalkerBackend`
  protocol. Every test in `tests/test_walker.py` uses an in-memory `FakeBackend` — no
  `acryl-datahub` import needed at test time. This keeps the walker's traversal logic
  unit-testable at 0.2 s speed.
- **Live adapter.** `DataHubBackend` wraps `acryl-datahub`'s `DataHubGraph` and is
  imported lazily so `ogle.walker` stays importable on a machine without the SDK. Users
  who feed their own signatures never install it. The SDK becomes an optional extra
  (`pip install "ogle[datahub]"`).

## Never guess

Missing-aspect behavior mirrors `scorer.score_dataset` — a signature never carries data
that DataHub did not report.

- `SchemaMetadata` absent + `DatasetProfile` absent -> **no signature** (returned in
  `WalkResult.skipped_urns` for diagnostics; not passed to the scorer).
- Either aspect present -> signature returned with whatever the aspect provided
  (`row_count=None`, `field_null_fractions={}` are legal). The scorer skips dimensions
  with data missing on either side.
- Fields with missing `fieldPath` / `nativeDataType` are skipped.
- Null-fraction values outside `[0, 1]` are rejected (defends against garbage aspects).

## Serving-detection gotcha (pinned)

`MLModelProperties.deployments` is a self-declared back-reference. If a datapack emits a
`mlModelDeployment` entity but leaves `deployments=[]` on the model props, `is_model_serving`
returns False even though the deployment exists. This bit Task #2 during the first live
walk (`scripts/inject-ml-lineage.py` originally didn't wire the back-ref) — fixed there.

Real-world DataHub deployments that don't populate this field can still be discovered via
the graph's downstream-relationships API. Ogle punts on this until it becomes a real
blocker; for the demo, the injector wires it explicitly.

## Live smoke test

Against a running quickstart with Task #2 ingested:

```bash
py -3.11 -c "
import sys; sys.path.insert(0, 'src')
from ogle.walker import DataHubBackend, walk_models
b = DataHubBackend(gms_server='http://localhost:8080')
CHURN = 'urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.churn_predictor,PROD)'
DEMAND = 'urn:li:mlModel:(urn:li:dataPlatform:mlflow,ogle_demo.demand_forecast,PROD)'
r = walk_models(b, [CHURN, DEMAND])
print(f'signatures={len(r.signatures)}  serving={len(r.serving_dataset_urns)}')
# -> signatures=9 (4 churn + 5 demand-only), serving=4 (churn's upstream only)
"
```

`discover_deployed_models()` on the same backend returns just `churn_predictor` — as
expected, `demand_forecast` is intentionally NOT deployed in the demo.

## Where it plugs in

The pipeline diagram in `w2-scorer.md` now has a real live-DataHub entry point:

```
DataHubBackend ─▶ walk_models() ─▶ WalkResult
                                       │  .signatures, .serving_dataset_urns
                                       ▼
                              run_drift_check(store, signatures, serving_urns)
                                       │
                                       ▼
                                 DriftReport (.should_alert, .narrative)
```

## Tests

`tests/test_walker.py` — **28 unit tests**, all using `FakeBackend`:

- Pure signature build: schema+profile fold; missing-aspect handling; invalid-value
  guards (null fraction out of range, negative rowCount, partial schema fields);
  `computed_at` propagation.
- Serving detection: any-IN_SERVICE-wins; no-deployment / not-in-service / missing-aspect
  fall-back to False.
- Traversal: cross-feature dataset dedup (first-seen order); skip missing feature props;
  no-model-props no-op.
- `walk_model`: signature+serving output shape; `skipped_urns` diagnostics; empty result on
  missing model; `computed_at` flows through.
- `walk_models`: shared-dataset dedup (first-seen signature wins); serving-set union.
- End-to-end: `walk_model` output feeds `run_drift_check` cleanly; a volume collapse on a
  serving dataset produces an alert-worthy incident on the second run.

Fault-injection verified: breaking `WalkResult.merge` to overwrite instead of union
flips the "unions_serving_set" test red with the exact assertion.
