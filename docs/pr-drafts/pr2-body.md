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

- Snippet was extracted from Ogle's [`walker.py`](https://github.com/BenDuske/ogle/blob/main/src/ogle/walker.py) which runs live against a DataHub Quickstart (see [`docs/live-verification.md`](https://github.com/BenDuske/ogle/blob/main/docs/live-verification.md)).
- Verified the aspects (`MLModelPropertiesClass`, `MLFeaturePropertiesClass`, `SchemaMetadataClass`, `DatasetProfileClass`, `OwnershipClass`) all exist in the current `acryl-datahub` release.

## Related

- Ogle repo: https://github.com/BenDuske/ogle
- Ogle's walker (reference implementation): https://github.com/BenDuske/ogle/blob/main/src/ogle/walker.py

## Notes for reviewers

- I picked "after Define Relationships / before Update Properties" as the insertion point since it's the natural progression (write → link → read → update). Happy to move it if maintainers prefer a different location.
- Snippet uses `DataHubGraph` directly to stay minimal; the tutorial's existing `DatahubAIClient` wrapper doesn't expose these reads yet — separate follow-up.
