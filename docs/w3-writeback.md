# W3 — Tag write-back

The outbound edit. Ogle detects drift; W3 makes it *visible* inside DataHub by stamping
the affected datasets and their downstream `mlModel` entities with a stable
`urn:li:tag:ogle-drift-flagged` tag. The next person or agent opening DataHub inherits
the finding without ever seeing Ogle's alert stream.

## Traversal

```
DriftFinding[]          (from scorer / pipeline)
   │
   ▼
plan_writeback(findings, walk_result)
   │
   ▼
WritebackPlan  ── each action targets ── ▶ drifted dataset URN
                                      └── ▶ every downstream mlModel URN
   │
   ▼
apply(plan, backend)
   │  reads existing GlobalTags
   ▼  merges (never clobbers)
DataHub  🟢 tag visible on entity
```

`walk_result.dataset_to_models` (added in W2c) is the reverse index the plan reads to
find each drifted dataset's downstream models. Without a walk result, `plan_writeback`
falls back to tagging just the datasets.

## Two-layer design (same as walker)

- **Pure core.** `plan_writeback`, `apply`, and the `WritebackBackend` protocol trade
  only in string tag URNs — no SDK types cross the pure/live boundary. Every test in
  `tests/test_writeback.py` uses a dict-backed `FakeWritebackBackend`.
- **Live adapter.** `DataHubWritebackBackend` wraps `acryl-datahub`'s `DataHubGraph`
  (imported lazily). Reads the target entity's `GlobalTags` aspect, adds Ogle's tag if
  missing, emits an MCP. Uses the same optional `datahub` extra as the walker.

## Safety invariants (each has a load-bearing test)

1. **Never clobber existing tags.** `apply` reads the entity's current tag URNs first
   and writes back the union. A human-added `urn:li:tag:pii` survives Ogle's edit.
2. **Read-failure is fatal — for that entity.** If the backend read raises, `apply`
   records the action in `failed` and **skips the write** (guessing "empty" would
   overwrite the real tag set). Verified: dropping this guard flips
   `test_apply_records_read_failure_without_writing`.
3. **Write-failure doesn't abort the batch.** A broken URN in a 10-entity plan tags
   the other 9; the failure is reported.
4. **Idempotent.** A tag already on an entity records in `unchanged`, not `applied`,
   and does not trigger a write. Ogle can safely run every 10 minutes.
5. **One write per entity.** Multiple actions targeting the same entity (e.g. custom
   tag + default tag) are merged into a single `set_tag_urns` call — no self-racing.

## CLI integration

`ogle check --write-back` runs the outbound edit **only** when the run produces a
*new* incident (`should_alert=True`):

```
0 - healthy → nothing to tag
1 - new incident + --write-back → plan + apply, tag entities, exit 1
2 - offline mode with --write-back → refused ("requires a live walk")
```

Debounce: the pipeline's incident fingerprint memory means the *same* drift on the
next tick returns `should_alert=False`, so write-back is skipped ("no new incident
this run"). Only genuinely new drifts hit DataHub.

## Live-verified end to end

Against a running Halcyon quickstart with Task #2 ingested:

```bash
# 1. Seed baselines from a healthy walk.
py -3.11 -m ogle.cli check --store ogle-baselines.json \
    --gms http://localhost:8080 --discover
# → "No drift detected" — 4 datasets seeded (churn's upstream)

# 2. Simulate schema drift: drop a column from `customers`.
py -3.11 -c "
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.ingestion.graph.client import DataHubGraph, DataHubGraphConfig
from datahub.metadata.schema_classes import SchemaMetadataClass
urn = 'urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.order_entry_db.order_entry.customers,PROD)'
g = DataHubGraph(DataHubGraphConfig(server='http://localhost:8080'))
sm = g.get_aspect(entity_urn=urn, aspect_type=SchemaMetadataClass)
sm.fields.pop()  # drop the last column
DatahubRestEmitter('http://localhost:8080').emit(MetadataChangeProposalWrapper(entityUrn=urn, aspect=sm))
"

# 3. Second check with write-back.
py -3.11 -m ogle.cli check --store ogle-baselines.json \
    --gms http://localhost:8080 --discover --write-back
# → HIGH severity schema drift on customers (serving path)
# → Tagged 2 entity(ies) in DataHub:
#   - customers dataset
#   - churn_predictor mlModel
# → exit 1
```

Confirmed via DataHub GraphQL — both entities carry `urn:li:tag:ogle-drift-flagged`.
Screenshot in `docs/screenshots/10-churn-predictor-tagged.png`.

## Tests

`tests/test_writeback.py` — **20 unit tests** on the pure core:

- `plan_writeback`: empty findings, dataset + downstream model tagging, dedup on repeat
  findings, multiple downstream models, shared downstream dedup, custom tag URN, plan
  shape.
- `apply`: writes missing tags, is idempotent, preserves unrelated tags, one write per
  entity, read-failure → no write, write-failure → batch continues, empty plan is no-op.
- End-to-end: findings → plan → apply → tags on datasets and models.

Fault-injection verified: dropping the read-failure guard flips exactly the safety test
without touching any of the happy paths.
