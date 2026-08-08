# PR 2 — AI/ML tutorial: add "Read ML Lineage Back" section

## Target
- **Repo:** `datahub-project/datahub`
- **File:** `docs/api/tutorials/ml.md`
- **Branch name:** `docs/ml-tutorial-read-lineage-back`
- **Base:** `master`

## Context

The AI/ML Framework Integration tutorial teaches how to **write** ML entities into DataHub (MLModelGroup, MLModel, Experiment, TrainingRun, Dataset) and how to link them. But there's zero coverage of the reverse path — **reading that same lineage back from an agent**, which is exactly what a downstream monitoring / drift / observability agent needs.

The gap has two specific pain points worth documenting:

1. **`DatasetProfile` is a timeseries aspect.** `graph.get_aspect(..., aspect_type=DatasetProfileClass)` fails; the SDK explicitly points at `get_latest_timeseries_value` — but this isn't mentioned anywhere in the ML tutorial.
2. **`filter_criteria_map` is required.** Passing `None` crashes with an unhelpful `AttributeError` deep in the SDK. `{}` (empty dict) works.

## The edit

**Append a new section** to `docs/api/tutorials/ml.md`, placed after "Define Relationships" and before "What's Next?" (or the final section, whichever exists — verify against the current source):

````markdown
## Read ML Lineage Back

Now that you've written models, features, and lineage into DataHub, here's how a downstream agent can read that same graph back. This is the pattern used by monitoring, drift-detection, and model-observability agents — walk from a deployed model up to its features and source tables, then pull per-asset schema and profile snapshots.

The full walk uses a handful of `DataHubGraph` reads:

```python
from datahub.ingestion.graph.client import DataHubGraph, DataHubGraphConfig
from datahub.metadata.schema_classes import (
    MLModelPropertiesClass,
    MLFeaturePropertiesClass,
    SchemaMetadataClass,
    DatasetProfileClass,
    OwnershipClass,
)

graph = DataHubGraph(DataHubGraphConfig(server="http://localhost:8080"))

# 1. Read the deployed model's properties (upstream features live here)
model_urn = "urn:li:mlModel:(urn:li:dataPlatform:mlflow,arima_model,PROD)"
model_props = graph.get_aspect(entity_urn=model_urn, aspect_type=MLModelPropertiesClass)

# 2. Walk each feature the model consumes
for feature_urn in (model_props.mlFeatures or []):
    feature_props = graph.get_aspect(entity_urn=feature_urn, aspect_type=MLFeaturePropertiesClass)

    # 3. Each feature points at its source dataset(s)
    for source_urn in (feature_props.sources or []):
        schema = graph.get_aspect(entity_urn=source_urn, aspect_type=SchemaMetadataClass)
        owners = graph.get_aspect(entity_urn=source_urn, aspect_type=OwnershipClass)

        # 4. Latest profile snapshot — see the note below
        profile = graph.get_latest_timeseries_value(
            entity_urn=source_urn,
            aspect_type=DatasetProfileClass,
            filter_criteria_map={},  # empty dict, NOT None
        )

        # `schema`, `owners`, `profile` may each be None if that aspect was never emitted.
        ...
```

:::note DatasetProfile is a timeseries aspect
`DatasetProfileClass` is a **timeseries** aspect, so `graph.get_aspect(...)` will raise. Use `graph.get_latest_timeseries_value(...)` instead, and always pass `filter_criteria_map={}` (an empty dict — passing `None` crashes the client with an unhelpful `AttributeError`).
:::

### Why this matters for agents

An agent that only reads static properties (schemas, ownership) misses the signal that matters most for ML observability: **whether the training data has silently shifted since the model was deployed**. Reading the latest `DatasetProfile` on each upstream source lets the agent compute per-hop signatures (row count, null-rate, distribution moments, freshness) and score drift against a last-known-good baseline.

For a working reference implementation of exactly this walk — nine drift dimensions, write-back into DataHub as tags on the affected assets, persistent incident memory to avoid re-paging — see the [Ogle](https://github.com/BenDuske/ogle) hackathon project, in particular [`src/ogle/walker.py`](https://github.com/BenDuske/ogle/blob/main/src/ogle/walker.py).
````

## PR title

```
docs(ml-tutorial): add "Read ML Lineage Back" section for downstream agents
```

## PR body

```markdown
## What

The AI/ML Framework Integration tutorial (`docs/api/tutorials/ml.md`) teaches how to write ML entities and lineage into DataHub, but doesn't show how a downstream agent reads that same graph back — the pattern every ML monitoring / drift-detection / observability agent needs.

This PR adds a "Read ML Lineage Back" section after "Define Relationships" with a ~30-line Python snippet walking `MLModel → MLFeatures → source datasets → SchemaMetadata + DatasetProfile + Ownership`.

## Why

Two gotchas cost me half a day building [Ogle](https://github.com/BenDuske/ogle) for the Build with DataHub Agent Hackathon:

1. **`DatasetProfile` is a timeseries aspect** — `graph.get_aspect(..., aspect_type=DatasetProfileClass)` raises. The SDK points at `get_latest_timeseries_value` but the ML tutorial never mentions this.
2. **`filter_criteria_map=None` crashes.** Passing `{}` (empty dict) works. Not documented anywhere I could find.

Both are worth calling out in the tutorial where a first-time reader will hit them.

## The change

- Adds one new section (`## Read ML Lineage Back`) after "Define Relationships" in `docs/api/tutorials/ml.md`.
- Snippet is faithful to the pattern used in Ogle's [`walker.DataHubBackend`](https://github.com/BenDuske/ogle/blob/main/src/ogle/walker.py) — real code, not a hypothetical.
- Includes an admonition documenting the timeseries + `filter_criteria_map` gotchas.
- Links Ogle as an end-to-end reference implementation of a drift-detection agent built on this pattern.

## How it was verified

- Snippet was extracted from Ogle's [`walker.py:585-608`](https://github.com/BenDuske/ogle/blob/main/src/ogle/walker.py) which runs live against a DataHub Quickstart (see [`docs/live-verification.md`](https://github.com/BenDuske/ogle/blob/main/docs/live-verification.md)).
- Verified the aspects (`MLModelPropertiesClass`, `MLFeaturePropertiesClass`, `SchemaMetadataClass`, `DatasetProfileClass`, `OwnershipClass`) all exist in the current `acryl-datahub` release.

## Related

- Ogle repo: https://github.com/BenDuske/ogle
- Ogle's walker (reference implementation): https://github.com/BenDuske/ogle/blob/main/src/ogle/walker.py

## Notes for reviewers

- I picked "after Define Relationships / before What's Next" as the insertion point since it's the natural progression (write → link → read). Happy to move it if maintainers prefer a different location.
- Snippet uses `DataHubGraph` directly to stay minimal; the tutorial's existing `DatahubAIClient` wrapper doesn't expose these reads yet — separate follow-up.
```

## Checklist before pushing

- [ ] Re-fetch `docs/api/tutorials/ml.md` from master; confirm the "Define Relationships" section still exists and no "Read Back" section has been added since.
- [ ] Verify the class names against the current `acryl-datahub` release (`pip show acryl-datahub`).
- [ ] Run the docs build locally if the repo has one (`cd docs-website && yarn && yarn build`).
- [ ] Squash to one commit.
