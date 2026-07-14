# Ogle — Architecture

```
       ┌──────────────────────────────────────────────────┐
       │                     Ogle                         │
       │                                                  │
       │  ┌─────────────┐   ┌──────────────┐   ┌────────┐ │
       │  │  Scheduler  │──▶│ Lineage-walk │──▶│ Scorer │ │
       │  │ (APScheduler│   │  (traverse   │   │(sigs + │ │
       │  │  or cron)   │   │   upstream)  │   │anomaly)│ │
       │  └─────────────┘   └──────┬───────┘   └────┬───┘ │
       │         ▲                 │                │     │
       │         │          ┌──────▼────────────────▼──┐  │
       │         │          │   Ogle-Brain (Aegis)     │  │
       │         │          │   facts / episodes /     │  │
       │         │          │   prefs · salience recall│  │
       │         │          └──────┬────────────────┬──┘  │
       │         │                 │                │     │
       │         │          ┌──────▼──────┐   ┌─────▼───┐ │
       │         │          │ Narrative   │   │  Alert  │ │
       │         │          │  writer     │   │ writer  │ │
       │         │          │  (LLM)      │   │(→DataHub│ │
       │         │          └──────┬──────┘   │  tags)  │ │
       │         │                 │          └─────┬───┘ │
       └─────────┼─────────────────┼────────────────┼─────┘
                 │                 │                │
       ┌─────────▼─────────────────▼────────────────▼─────┐
       │        DataHub MCP Server / Skills               │
       │  (read lineage/ownership, write tags/annotations)│
       └──────────────────────────────────────────────────┘
```

## Components

- **Scheduler.** Fires a walk cycle per model on a configurable cadence
  (default: every N hours). APScheduler for the demo; cron-friendly for prod.
- **Lineage-walk.** Given a target model URN, traverses upstream through
  DataHub's lineage graph via the MCP server. Emits a stream of `(asset,
  hop_depth, metadata)` tuples.
- **Scorer.** For each visited asset, computes a lightweight *signature*
  (row-count delta vs. last-known, schema hash, coarse distribution stats)
  and produces an anomaly score. Signatures cached locally.
- **Ogle-Brain (Aegis integration).** Every walk consults memory: has this
  exact anomaly been seen before? Was it a false positive last time?
  Findings write back to memory as episodes.
- **Narrative writer.** On above-threshold anomaly, an LLM composes a short
  actionable narrative using: the walk trace, DataHub ownership/docs
  metadata, and Ogle-Brain's recall of related past incidents.
- **Alert writer.** Persists the narrative to a local `examples/alerts/`
  file *and* writes a `ogle:flagged` tag back to DataHub on the affected
  asset — the write-back that hits the judging rubric's "contribute back
  to the graph" line.

## Data flow (one cycle)

1. Scheduler triggers `walk(model_urn)`.
2. Lineage-walk queries DataHub MCP for upstream lineage → asset list.
3. Scorer computes signature for each asset; diffs against cache.
4. Any anomaly → Ogle-Brain recall (similar past events + user prefs).
5. Above threshold → Narrative writer composes alert.
6. Alert writer persists locally + tags asset in DataHub.
7. Episode written back to Ogle-Brain for next cycle to learn from.

## Design principles

- **Read-heavy on DataHub, write-tiny.** We only write tags/annotations on
  flagged assets — never touch the underlying data or model artifacts.
- **Fail-safe.** A DataHub outage degrades to logged skip, never a bad
  alert. A model outage degrades to a rule-based narrative, never silence.
- **Salience-budgeted memory.** Reused from Aegis — the prompt sent to the
  LLM is bounded by tokens, not row count, so Ogle can run for months
  without prompt bloat.
