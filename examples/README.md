# Examples

Sample outputs from Ogle — populated as the engine comes online.

Planned layout:

- `alerts/` — narrative alert markdown files produced by the LLM narrative
  writer, one per flagged asset.
- `walks/` — JSON dumps of full lineage-walk traces (asset list, hop
  depths, signature diffs, anomaly scores).
- `writebacks/` — DataHub tag/annotation payloads Ogle wrote back to the
  graph (rubric line: *"contribute back to the graph"*).
- `demo-dataset/` — the seeded 3-table → 2-feature → 1-model dataset used
  in the demo, plus the drift-simulation script's before/after diffs.

**Why these matter for judging:** the Devpost rules explicitly call out
sample outputs as a way for judges to evaluate quality *without* running
the code. Every artifact in this folder is a judgment shortcut.
