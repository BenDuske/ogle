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

Add `--narrate` to also see the **LLM root-cause summary** (feature #2) in the same keyless
command — it uses a local Ollama model (`qwen3:latest`) by default and falls back to the
deterministic summary if none is reachable, so it stays zero-key:

```bash
ogle demo --narrate
```

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

**Tuning sensitivity per deployment.** Defaults are quiet-on-noise, loud-on-breakage
(volume drift at ±30%, quality drift at a +0.20 null-fraction jump), but a noisy dimension
table and a stable serving-path source don't want the same band. Override per run:

```bash
ogle check --signatures sigs.json \
  --volume-threshold 0.15 \   # flag row-count moves past ±15%
  --null-threshold 0.10 \     # flag a null-fraction jump of +0.10
  --no-serving-escalation     # don't bump severity for serving-path sources
```

Thresholds are validated up front — a nonsensical value (volume ≤ 0, or a null band
outside `(0, 1]`) exits **2** before any walk. Schema drift (a removed/retyped column) is
always flagged regardless of these knobs.

**Gating a CI pipeline on severity (`--fail-on`).** By default *any* new incident exits
**1**. When you want a build to go red only on the worst drift, `--fail-on {low,medium,high}`
raises the bar: a new incident below that floor is still reported (and still tagged with
`--write-back`) but the process exits **0**, so a gate can page on HIGH while merely logging
the rest. No new incident is always **0**.

```bash
ogle check --signatures sigs.json --fail-on high   # exit 1 only on a HIGH incident
```

### Muting known false positives (`ogle mute`)

Some assets are chronically noisy — a dashboard that bounces every Monday, a staging table
that gets truncated and reloaded nightly. Debounce alone won't help there: each flap is a
*genuinely new* incident fingerprint, so it pages every time. Tell Ogle to remember it's a
false positive instead:

```bash
# stop paging on a known-noisy dataset (persists into the store `ogle check` reads)
ogle mute 'urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)'

# ...or snooze it temporarily — auto-expires so "quiet it for now" never becomes a
# permanent blind spot:
ogle mute '<urn>' --for 7          # snooze 7 days
ogle mute '<urn>' --for-hours 4    # snooze 4 hours

ogle muted            # list what's currently silenced (snoozes show their expiry)
ogle unmute '<urn>'   # let it page again
```

A muted dataset is **still tracked** — its baseline keeps advancing, so an `unmute` later
diffs against fresh state, not stale — it just never contributes to an incident. A **snooze**
(`--for` / `--for-hours`) is a mute that lapses on its own: once the expiry passes it pages
again automatically, and `ogle check` self-cleans the dead entry from the store. A permanent
mute always wins over a snooze for the same URN. `ogle check` reports how many muted datasets
it silenced (`silenced N muted dataset(s)`) and lists them under `suppressed_urns` in `--json`,
so the suppression is visible, never a silent black hole. This is feature #3 (memory of past
false positives) as a first-class operator control.

### Inspecting what Ogle remembers (`ogle incidents`)

Ogle keeps a cross-run memory of every incident it has seen — that's what lets it page
*once* on a new problem instead of every tick. `ogle incidents` makes that memory
inspectable, so an operator can see the open drift Ogle is tracking without re-walking
DataHub:

```bash
ogle incidents                          # what drift Ogle currently remembers, worst-severity first
ogle incidents --json                   # same, machine-readable
ogle incidents --min-severity high      # triage: only high-severity incidents (drops unknown/legacy)
ogle incidents --serving-only           # only incidents that touch a serving path
ogle incidents --min-count 3            # only chronic/flapping drift seen 3+ times
ogle incidents --min-severity high --serving-only   # filters compose (AND)
ogle incidents --summary                # aggregate rollup instead of the per-incident list
```

The `--min-severity {low,medium,high}`, `--serving-only`, and `--min-count N` filters mirror
`check --fail-on` so a busy operator can focus on what pages first — `--min-count` surfaces the
chronic drift that keeps recurring despite being "seen." When a filter empties a non-empty memory,
Ogle says so (`no incidents match the filter (N remembered)`) rather than implying nothing is
tracked.

`--summary` swaps the per-incident list for an at-a-glance rollup — total remembered, a count
per severity, how many touch a serving path, how many are recurring (seen ≥2×), and the total
sighting count. It describes the *filtered* set, so it composes with the triage flags (e.g.
`ogle incidents --summary --serving-only` summarizes only the serving-path drift). Add `--json`
for a machine-readable `{"summary": {...}}` shape.

Each line shows the incident's **severity**, a human **headline**, how many times it has
**recurred** (`seen 3×` — the "still happening" signal), how many datasets it spans, whether
it touches a **serving path**, and its stable fingerprint for cross-reference:

```
**2 remembered incident(s):**
- 🔴 **high** — HIGH drift across 1 dataset on a serving path · seen 3× · 1 dataset(s) · ⚠️ serving  `fd6f829c77ff9fb4`
- 🟡 **low** — LOW drift across 1 dataset · seen 1× · 1 dataset(s)  `a1b2c3d4e5f60718`
```

It's read-only — it never advances baselines or pages — and the provenance is additive, so a
store written by an older Ogle (which recorded only a recurrence count) still loads and lists.

### Closing the loop (`ogle resolve`)

Once the upstream drift is actually fixed in prod, tell Ogle to drop the incident from
memory — it stops appearing in `ogle incidents`, and if the same drift shape reappears
later Ogle pages **fresh** (resolve is not a mute — it doesn't suppress a *future* problem,
it just retires the current one). Accepts the full 16-hex fingerprint from `ogle incidents`
or an unambiguous prefix, like a git short SHA:

```bash
ogle resolve fd6f829c                              # short-prefix, like a git SHA
ogle resolve fd6f829c77ff9fb4 a1b2c3d4e5f60718     # batch — hits and misses report per token
```

Ambiguous prefixes fail loud (exit 2) with the list of candidates so the operator retypes
with more characters — Ogle never guesses which incident to drop. Unknown/already-forgotten
tokens report `_not remembered_` but aren't an error (exit 0), so replaying a list is safe.

### Writing findings back into DataHub (`--write-back`)

On a **new** incident, `ogle check --write-back` stamps every drifted dataset and its
downstream `mlModel`s with `urn:li:tag:ogle-drift-flagged` in DataHub, so the next person
or agent browsing the graph inherits the finding without re-running Ogle. It only writes
when *this* run fires a new incident (not on every tick), the merge is idempotent (a tag
already present is skipped), and it requires a live walk (`--gms`).

```bash
ogle check --gms http://localhost:8080 --discover --store live.json --write-back
ogle check --gms http://localhost:8080 --discover --store live.json --write-back --write-back-severity
```

`--write-back-severity` adds a **per-severity** tag (`ogle-drift-high` / `-medium` / `-low`)
alongside the flat one, so an operator can filter DataHub straight to the worst drift. A
dataset's severity tag is the worst of its own findings; a **model inherits the worst
severity of the drifted datasets feeding it** — the finding that would page you is the one
that colours the model. The flat tag is always stamped too, so coarse "everything Ogle
flagged" grouping keeps working.

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
