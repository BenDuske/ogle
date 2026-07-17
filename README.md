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

### Against a live DataHub (Docker)

```bash
# 1. Bring up a local DataHub quickstart (Docker required)
datahub docker quickstart

# 2. Seed the demo ML lineage (source tables → feature tables → serving models)
python scripts/inject-ml-lineage.py --gms http://localhost:8080

# 3. Run one drift-check cycle (walks lineage, seeds baselines on first run, exit 0)
ogle check --gms http://localhost:8080 --discover --store live.json

# 4. Re-check the unchanged graph — no false drift (exit 0)
ogle check --gms http://localhost:8080 --discover --store live.json
```

Steps 3–4 prove Ogle seeds baselines then reports **no false drift** on a stable graph
(both exit 0). To watch the alert path actually fire — schema/volume/quality drift on a
serving-path table — run the offline demo below; it's the same drift-check code path,
fully reproducible without Docker, and its captured alert lives in
[`examples/alerts/churn-orders-drift.md`](examples/alerts/churn-orders-drift.md).

### Without Docker (offline signatures)

**One command, zero setup, no API key** — run the whole loop over bundled fixtures:

```bash
ogle demo
```

It seeds healthy baselines (exit 0), then re-checks a drifted fixture and fires the HIGH
serving-path alert (exit 1) — the same drift-check code path a live DataHub walk feeds,
reproducing [`examples/alerts/churn-orders-drift.md`](examples/alerts/churn-orders-drift.md)
verbatim. Nothing is written to your working directory.

`ogle check` also runs against pre-computed signatures — no SDK, no quickstart — which is
how it's unit-tested and how a scheduled job can feed signatures it pulled elsewhere:

```bash
# Seed baselines from the healthy demo fixture (exit 0)
ogle check --store demo.json --signatures examples/demo/healthy-signatures.json

# Re-check against the drifted fixture — fires a HIGH serving-path alert (exit 1)
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
```

The second command reproduces [`examples/alerts/churn-orders-drift.md`](examples/alerts/churn-orders-drift.md)
verbatim. Point `--signatures` at your own file to feed signatures pulled elsewhere.

The signatures file is a JSON list of `DatasetSignature` dicts, or
`{"signatures": [...], "serving_urns": [...]}`. Exit codes let a cron/Task wrapper branch:
**0** = healthy (may include first-run seeding), **1** = a *new* incident fired (alert),
**2** = usage/input error. Re-running an unchanged drift is debounced to **0** — you're paged
once per incident, not every tick. Add `--json` for machine output, `--no-update` for a
read-only probe.

### Running on a schedule (`ogle watch`)

`ogle watch` is one scheduler tick: it runs `ogle check`, then acts on the exit code —
**page once on a new incident (1), stay quiet when healthy (0)**. The scheduler owns the
loop (a cron line / a Windows Scheduled Task); because the pipeline debounces standing
drift to `0`, you're paged once per incident, not on every tick. Put the flags for
`ogle check` after `--`:

```bash
# stderr pager (default) — a PAGE: block is printed only on a new incident
ogle watch -- --store baselines.json --signatures my-signatures.json

# wire a real pager: the narrative is handed to your command on stdin
ogle watch --notify-cmd mail -s "ogle drift" you@host \
  -- --gms http://localhost:8080 --discover --write-back
```

`--page-on-error` also pages on exit 2 (input/live-walk failure); by default those are
logged but not paged, so a transient GMS outage doesn't cry wolf. `ogle watch` preserves
the underlying `ogle check` exit code, so a cron/Task can still branch on it. Example
cron line (every 15 min):

```cron
*/15 * * * * cd /srv/ogle && ogle watch --notify-cmd /usr/local/bin/page-me -- \
  --store /var/lib/ogle/baselines.json --gms http://localhost:8080 --discover
```

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Status

🚧 In active build for the DataHub Agent Hackathon.
Submission window: Jul 6 – **Aug 10, 2026 @ 5 PM ET**.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
