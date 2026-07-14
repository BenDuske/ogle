# Ogle

*The ML lineage agent that just can't stop staring.*

Ogle walks your DataHub lineage graph on a schedule, detects silent training-data drift
and stale-feature deployments *before* they hit production, writes a root-cause narrative
your on-call engineer can act on in 30 seconds, and remembers what it's already flagged
so it doesn't spam.

Built for the **DataHub Agent Hackathon** (Track: Production ML Agents).

## Why

Every ML team lives one silent training-data change away from a bad model in prod. The
lineage exists in DataHub. What's missing is an agent that *walks* that lineage on a
schedule, catches the change before the deploy, and files a PR-quality alert with the
evidence — and *remembers* past incidents and false positives so it gets sharper over time.

## What it does

1. **Drift-walk.** Given a deployed model in DataHub, Ogle walks upstream through the
   lineage graph (model → features → source tables). For each hop it computes a
   lightweight signature (row-count delta, schema hash, distribution proxy) and compares
   against the last known state. Anomalies get scored.
2. **Root-cause narrative.** When something flags, Ogle uses an LLM plus DataHub
   ownership/documentation context to write a short, actionable narrative: what changed,
   when, who owns it, which downstream models are exposed, and the direct link to inspect.
3. **Memory of past incidents.** Ogle's brain is a salience-ranked, forgetful memory
   store (based on [Aegis MemoryAgent](https://github.com/BenDuske/qwen-memoryagent)).
   It remembers past false positives ("this dashboard bounces every Monday, ignore") and
   past real incidents ("last time table X row count dropped 40%, the ETL job Y had
   silently failed — check that first"). Findings are written *back* into DataHub as tags
   on the affected assets, so the next person or agent inherits the knowledge.

## Quickstart

```bash
# 1. Bring up a local DataHub + Ogle stack (Docker required)
docker compose up -d

# 2. Seed the demo dataset (3 tables → 2 features → 1 model)
python scripts/seed_demo_dataset.py

# 3. Run one lineage-walk cycle
python -m ogle walk --model demo.recommender.v1

# 4. Simulate a drift event and re-walk
python scripts/simulate_drift.py
python -m ogle walk --model demo.recommender.v1
```

Expected: on the second walk, Ogle flags the drifted upstream table, writes a narrative
alert (see `examples/alerts/`), and tags the asset in DataHub.

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Status

🚧 In active build for the DataHub Agent Hackathon.
Submission window: Jul 6 – **Aug 10, 2026 @ 5 PM ET**.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
