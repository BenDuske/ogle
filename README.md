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

# ...and record WHY, so the silence isn't a mystery weeks later:
ogle mute '<urn>' --reason 'dashboard bounces every Monday — upstream ETL retry'

ogle muted            # list what's currently silenced (each shows its age, expiry + reason)
ogle muted --permanent  # only the standing blind spots (no expiry) — the audit view
ogle muted --snoozed    # only the timed mutes that lapse on their own
ogle muted --unexplained  # only mutes with no --reason — the undocumented silences
ogle unmute '<urn>'   # let it page again

# audit-and-lift every standing blind spot at once:
ogle muted --permanent --urns | xargs -n1 ogle unmute

# find the standing blind spots nobody documented — permanent AND reasonless:
ogle muted --permanent --unexplained
```

A muted dataset is **still tracked** — its baseline keeps advancing, so an `unmute` later
diffs against fresh state, not stale — it just never contributes to an incident. A **snooze**
(`--for` / `--for-hours`) is a mute that lapses on its own: once the expiry passes it pages
again automatically, and `ogle check` self-cleans the dead entry from the store. A permanent
mute always wins over a snooze for the same URN. **`ogle muted --permanent`** / **`--snoozed`**
split the list into its two risk kinds (the same `⛔ permanent · 💤 snoozed` cross-tab `status`
shows): a *permanent* mute silences drift with no end date — a standing blind spot to audit and
justify — while a *snooze* clears itself, so `--permanent` is the view for exactly the mutes that
never self-lapse, and it composes with `--urns` to feed a bulk `unmute`. An optional **`--reason`** attaches a human
note to the mute — the "why" that a bare URN can't tell you — surfaced by `ogle muted` and
`ogle show` and carried in their `--json`; it can annotate an already-muted URN after the
fact, and is cleared automatically when the mute is lifted (unmute / forget / snooze expiry).
**`ogle muted --unexplained`** is the accountability flip-side of `--reason`: it keeps only the
mutes with *no* note recorded, so `ogle muted --permanent --unexplained` surfaces the
standing blind spots nobody justified — the ones to document or lift (a blank/whitespace
`--reason` never counts as explained, since `mute` stores it as no note)
so a reason never outlives the mute it explains. Each mute also carries **how long it's been
standing** — `ogle muted` reads `muted 3d ago` per line and `--json` a `since` epoch — because
a permanent mute set weeks ago is a bigger blind spot than a fresh one; the stamp is set when
the silence begins, preserved across re-annotation, and cleared with the mute (a legacy/undated
mute reads its age as unknown rather than faking one). `ogle check` reports how many muted datasets
it silenced (`silenced N muted dataset(s)`) and lists them under `suppressed_urns` in `--json`,
so the suppression is visible, never a silent black hole. This is feature #3 (memory of past
false positives) as a first-class operator control.

### Store health at a glance (`ogle status`)

The store has two halves plus a mute list, each with its own view (`baselines`, `incidents`,
`muted`). `ogle status` is the top-level rollup that unifies all three into one snapshot, so an
operator — or a scheduled wrapper — can answer "what is Ogle holding right now?" in a single
read, without re-walking DataHub:

```bash
ogle status                    # watch-list size + field/row totals, incident memory by severity, active mutes
ogle status --json             # same snapshot, machine-readable (baselines/incidents/muted rollup)
ogle status --fail-on high     # exit 1 if any remembered incident is high+ (CI/cron health gate)
ogle status --stale-after 6h   # exit 1 if the store hasn't been written in 6h (monitor-went-dark gate)
ogle status --orphan-after 2d  # exit 1 if any watched dataset's baseline hasn't refreshed in 2d (orphan gate)
```

The human view is four lines: **watching** (tracked datasets · total schema fields · total rows,
with an `(N unknown)` note when some baselines have no captured row count), **incidents
remembered** (broken out 🔴 high / 🟠 medium / 🟡 low / • unknown), **serving-path / recurring /
total sightings** — where the serving-path count, when anything is serving, appends its own
severity split `(🔴 high · 🟠 medium · 🟡 low · • unknown)` so the load-bearing 🔴 *high-serving*
page (a deployed model being fed drifted data right now) can't hide inside a flat total, the same
cross-tab the `ogle_incidents_serving_by_severity` gauge exposes (the four buckets sum back to
`serving-path`) — and **muted** — which, when anything is silenced, splits into `⛔ N permanent`
(a *standing* blind spot: drift suppressed with no end date) and `💤 N snoozed` (self-expiring), the
same distinction the `ogle_muted_permanent` gauge exposes, so a forever-muted serving table can't
hide inside a bland "N active". When a snooze is pending, the line appends `⏰ next lifts in <age>`
— the countdown to the *soonest* active snooze lapsing (when that silenced dataset's drift returns
to paging), the human twin of the `ogle_muted_snooze_next_expiry_seconds` gauge. `--json` carries
the split as `muted_permanent`/`muted_snoozed` (disjoint — they sum back to `muted`) plus
`muted_snooze_next_expiry_seconds` (null when no snooze is active, so a consumer can tell "no snooze
pending" from "lapses in 0s"). The severity
and serving counts reuse the same rollup as `incidents --summary`, so the two always agree on the
same store. An untouched store reports `is empty — run ogle check to start watching` rather than a
wall of zeros.

`--fail-on {low,medium,high}` turns the rollup into a CI/scheduled **health gate**: `ogle status`
exits `1` when any remembered incident sits at or above the floor (and prints why), so a nightly
job that already runs `status` for the snapshot can page off the same call — no second command.
Unlike `incidents --fail-on`, it gates on the *whole* store (`status` has no filters), and it stays
`0` by default so the plain snapshot never pages unless a floor is asked for. Mirrors the exit-code
contract of `check --fail-on` / `incidents --fail-on`.

`--stale-after <age>` is the **dead-man's-switch** the severity gate structurally can't be. Every
count `--fail-on` reads is a point-in-time store *level*, so if the scheduled `ogle check` crash-loops
or its cron is pulled, the store freezes and keeps reporting its last (below-floor) incidents forever
— the severity gate stays green while drift silently goes undetected. `--stale-after 6h` exits `1`
when the store file hasn't been written within the window (its mtime is a true heartbeat — it only
advances when a check actually persists an update), catching Ogle *itself* going dark; a **missing**
store (Ogle never ran, or the file was deleted) trips it too. It uses the same compact-duration
grammar as `incidents --stale` (`30m`, `6h`, `2d`, `1w`; a bad value is a hard exit `2`), the CLI gate
on top of the `ogle_store_age_seconds` gauge. It **composes with `--fail-on`** — either gate fails the
run — so one scheduled `ogle status --fail-on high --stale-after 6h` pages on *both* a high incident
and the monitor going quiet. `--json` carries the verdict as `heartbeat_stale` (`true`/`false`, or
`null` when `--stale-after` isn't given, so a consumer can tell "not checked" from "checked, healthy").

`--orphan-after <age>` is the **per-dataset twin** of `--stale-after`. The heartbeat gate catches Ogle
going dark *wholesale* — no store write at all — but it stays green when the walk still runs and rewrites
the store every tick yet *one* dataset silently drops out of it (renamed, de-provisioned upstream, or
filtered away): that URN's baseline just ages while everything else looks healthy. `--orphan-after 2d`
exits `1` when any baseline's capture age exceeds the window (Ogle refreshes a signature on every clean
walk that still sees the dataset, so an aging capture stamp *is* the "stopped seeing it" signal), turning
the orphan hint `status` already shows (the *oldest baseline capture* line) into something a cron/CI
wrapper can page on. **Untimed** baselines (no parseable `computed_at`) can't be proven stale, so they
never trip it — the same "never guess an age" rule the capture-age bounds and `baselines --stale` follow.
Same compact-duration grammar (`30m`, `12h`, `2d`, `1w`; a bad value is a hard exit `2`), and it composes
with the other gates — *any one* fails the run. `--json` carries the count as `stale_baselines` (an int,
or `null` when `--orphan-after` isn't given, so a consumer can tell "not checked" from "checked, none stale").

Each gate boolean above attributes *which* gate fired; when you just want the folded verdict, `--json`
also carries `exit_rc` — the exact process exit code (`0` pass / `1` any gate tripped), so a consumer need
not OR the three nullable gate fields together. It also survives a stdout capture forwarded over a
log/message bus, where the OS exit code is lost.

Where `exit_rc` folds *whether* to page, `gates_tripped` folds *why*: a `--json`-only list of the gate
names that fired, in evaluation order (`["drift", "heartbeat", "orphan"]`). It's always present — an empty
list means nothing tripped (`nonempty` iff `exit_rc == 1`) — so an alert router can dispatch a `drift` trip
to the model owner and a `heartbeat`/`orphan` trip to the monitor's SRE, without re-deriving each gate's
`null`-vs-`false` semantics. A new gate simply appends its name.

### Prometheus metrics (`ogle metrics`)

`ogle status --json` answers "what is Ogle holding right now?" for a script; `ogle metrics`
answers it for a **monitoring stack**. It renders the identical watch-list + incidents + mutes
rollup as Prometheus text exposition format, so you can graph Ogle's drift memory over time and
alert on it — the production-observability counterpart to the human snapshot:

```bash
ogle metrics                                   # Prometheus text exposition on stdout
ogle metrics -o /var/lib/node_exporter/textfile/ogle.prom   # atomic write for the collector
```

For a **node_exporter textfile collector**, prefer `-o/--output` over a `> ogle.prom` redirect:
the collector polls its directory on its own clock, so a plain redirect can be scraped
mid-write and drop the whole file. `--output` writes to a temp file in the same directory and
`os.replace`s it into place, so the collector always sees either the old file or the complete
new one — never a torn one. The confirmation line goes to stderr, keeping stdout (and the
`.prom` file) clean.

Every series is a **gauge** (point-in-time store levels, so none carry the `_total` suffix
Prometheus reserves for counters): `ogle_up`, `ogle_watching_datasets` / `_fields` / `_rows` /
`_rows_unknown`, `ogle_incidents_remembered{severity="high|medium|low|unknown"}`,
`ogle_incidents_serving` (split by severity in
`ogle_incidents_serving_by_severity{severity="…"}`) / `_recurring` / `_sightings`,
`ogle_muted_active` (split into
`ogle_muted_permanent` + the snooze countdown `ogle_muted_snooze_next_expiry_seconds`), the
incident staleness ages (`ogle_incidents_last_seen_{min,max}_age_seconds`), the incident
standing ages (`ogle_incidents_first_seen_{min,max}_age_seconds` — the longevity twin: alert
`ogle_incidents_first_seen_max_age_seconds` climbing past a weeks threshold to page on a
chronic, never-resolved incident), the baseline
capture ages (`ogle_baseline_{newest,oldest}_capture_age_seconds` + the `ogle_baseline_capture_age_unknown`
coverage gap), and a monitor heartbeat `ogle_store_age_seconds`. The numbers are the same rollups `status` prints (verified against
`status --json` in the test suite), so a Grafana panel and the CLI snapshot never disagree.
Unlike `status --fail-on`, `metrics` **always exits 0** — a scrape must not fail on data levels;
keep gating on `status`/`incidents --fail-on`.

**Monitor the monitor.** Every gauge above is a point-in-time store *level*, so if the scheduled
`ogle check` crash-loops or its cron is removed, those gauges freeze at their last value while
`ogle_up` keeps asserting `1` — the dashboard looks healthy while drift goes undetected.
`ogle_store_age_seconds` is the dead-man's-switch: it's `now − store-file-mtime`, and the store
file's mtime only advances when `ogle check` actually runs, so alerting on
`ogle_store_age_seconds > 2 × check_interval` fires when Ogle itself goes dark, independent of
whether any drift is present. Before the first store write (no file yet) the family is declared
but emits **no sample** — an honest "no data", not a fabricated zero age.

**Orphan detection.** `ogle_store_age_seconds` catches Ogle going dark *everywhere at once*, but not a
*single* dataset silently dropping out of the walk (renamed / de-provisioned / lineage pruned) while its
baseline lingers as a blind spot. A clean walk refreshes a signature every time it still sees the dataset,
so the baseline capture ages localize that: `ogle_baseline_oldest_capture_age_seconds` is the stalest
watched signature — alert `ogle_baseline_oldest_capture_age_seconds > 2 × walk_interval` to page on an
orphaned baseline, the Prometheus counterpart to `ogle baselines --sort age`/`--stale`.
`ogle_baseline_newest_capture_age_seconds` bounds the fresh end, and `ogle_baseline_capture_age_unknown`
is the coverage companion (like `ogle_watching_rows_unknown`): baselines with no parseable `computed_at`
whose age can't be asserted, so a small oldest-age next to a large unknown reads as "most of the watch-list
can't be checked for orphaning". Both age families are declared but emit **no sample** when no baseline
carries a stamp — an honest "no data", never a fake zero. The human `ogle status` snapshot carries the
same signal as its twin: a `🕰️ oldest baseline capture: 1w ago · newest: 2h ago · N untimed` line
(stalest first — the orphan candidate; the untimed count trails as a coverage caveat), shown only when
at least one baseline is timestamped and mirrored in `--json` as `oldest_baseline_capture_age_seconds` /
`newest_baseline_capture_age_seconds` / `baseline_capture_age_unknown` — so a glance at `status` catches
an orphaning watch-list without a scrape, and the snapshot and the gauge can never disagree (shared code).

**Mutes aren't all equal.** `ogle_muted_active` counts every silenced dataset as one number, but
a *permanent* mute is a chosen **standing blind spot** — drift on that dataset is suppressed with
no end date, so a serving table quietly muted months ago keeps hiding real drift forever.
`ogle_muted_permanent` surfaces that count on its own (alert if it creeps up on serving assets),
while `ogle_muted_snooze_next_expiry_seconds` counts down to the soonest *snooze* lapsing so you
can anticipate drift returning to the page. The two split the active total exactly
(`permanent + snoozed == ogle_muted_active`); the countdown emits **no sample** when nothing is
snoozed rather than a fabricated `0` that would read as "expiring right now".

**The page-worthy cell is serving × severity.** The single alert that matters most for a
production ML monitor is "a *high-severity* drift is hitting a *serving* path right now" — a
deployed model being fed drifted data. The flat `ogle_incidents_serving` (any severity) and
`ogle_incidents_remembered{severity="high"}` (serving or not) **can't express it**: one
low-severity serving incident next to one high-severity *non*-serving incident reads as
`serving 1` and `remembered{high} 1`, yet there are **zero** high-severity serving incidents.
`ogle_incidents_serving_by_severity{severity="…"}` is that cross-tab — alert on
`ogle_incidents_serving_by_severity{severity="high"} > 0` for the real page. It's emitted for
all four severity buckets (honest `0`s, so the alert series always exists) and sums exactly to
`ogle_incidents_serving`, mirroring the `muted_active` + `muted_permanent` "keep the total, add
the risk-highlighting split" shape.

### Corruption-resilient store (unattended-safe)

The store is written atomically (temp file + `os.replace`), so Ogle itself never leaves a
half-written file. But a store on disk can still go bad from the outside — a truncated
cloud-sync, a hand-edit, a disk fault, or a file from a future/foreign Ogle version. A
scheduled `ogle check` that simply crashed on that would **crash-loop and go silently blind
to drift** — the worst failure mode for a production monitor.

Instead, `ogle check` recovers: an unreadable store (bad JSON, wrong version, or malformed
shape) is **quarantined** to `<store>.corrupt` (deterministic `.corrupt.1`, `.corrupt.2`, …
if a prior forensic copy already exists — never clobbered), a **loud warning** is printed to
stderr, and the run **re-baselines from scratch** and exits `0` rather than failing. The next
walk repopulates baselines, and the operator keeps the bad file for forensics.

```
ogle check: WARNING baseline store at live.json was unreadable (corrupt/foreign);
quarantined to live.json.corrupt and re-baselining from scratch — this run cannot
detect drift against prior state.
```

The warning goes to **stderr**, so `--json` on stdout stays clean for a wrapper to parse.
Strict callers that would rather see the raw error can use `BaselineStore.load(path,
recover_corrupt=False)`.

### Inspecting the watch-list (`ogle baselines`)

The store has two halves. `ogle incidents` shows the drift Ogle *remembers*; `ogle baselines`
shows what it's *watching* — every dataset it has captured a signature for and will diff the
next walk against. It answers "is this dataset actually under Ogle's eye?" and "how many am I
tracking?" without re-walking DataHub:

```bash
ogle baselines                       # every tracked dataset: field count, row count, capture age, schema hash
ogle baselines --json                # machine-readable (urn, fields, row_count, schema_hash, computed_at, age_seconds)
ogle baselines --grep serving        # find tracked datasets by URN keyword (case-insensitive)
ogle baselines --sort fields         # widest-schema datasets first (highest blast radius)
ogle baselines --sort rows           # highest-volume datasets first
ogle baselines --sort age            # stalest capture first — datasets most likely to have dropped out of the walk
ogle baselines --stale 7d            # only baselines captured ≥7d ago — the orphan filter
ogle baselines --urns                # plain URNs, one per line — a selector to pipe into a write-side command
ogle baselines --grep staging --urns | xargs -n1 ogle mute   # mute a whole class of tracked datasets
ogle baselines --stale 14d --urns | xargs -n1 ogle forget    # prune baselines the walk stopped refreshing
```

`--sort {urn,fields,rows,age}` picks the ordering axis. The default `urn` is alphabetical (the
stable order for scripting); `fields` and `rows` surface the **highest-blast-radius** watched
datasets first — the widest schemas and highest-volume tables, where a silent schema or volume
shift does the most damage; `age` surfaces the **stalest captures** first. Ties break on URN
ascending (deterministic run to run), and a baseline with no signature / unknown row count /
unknown capture age sinks last. `--sort` is honored by `--urns` and `--json` too, so
`ogle baselines --sort rows --urns | head -5` is "the five biggest tables I'm watching."

`--stale DURATION` (e.g. `7d`, `12h`, `2w`) keeps only baselines whose capture stamp is at least
that old — the **orphan filter**. Ogle refreshes a signature on every clean walk that still sees
the dataset, so a stamp older than your walk cadence means Ogle *stopped seeing* that URN (dropped
from the walk, renamed, de-provisioned) while its baseline lingers — a per-dataset blind spot the
store-wide freshness heartbeat can't localize. Baselines with **no capture timestamp are excluded**
(staleness can't be asserted, so Ogle never guesses one). A bad duration is a hard error (exit 2).

`--urns` mirrors `incidents --fingerprints`: it turns the read-side watch-list into a selector
for the write side (`ogle mute`, or feeding `ogle check --models`). It honors `--grep`/`--sort`,
overrides `--json`, and stays **silent on an empty set** so a pipe gets a clean stream. An
all-whitespace `--grep "   "` matches nothing (a slip, not a wildcard), and a filter that hides
everything says so (`no baselines match the filter`) rather than reading as an empty store.

### Drilling into one dataset (`ogle show`)

`ogle baselines` lists the watch-list a summary line each; `ogle show <urn>` opens a **single**
dataset and prints the full signature Ogle memorized — the detail no other view exposes. After a
page ("Ogle flagged `orders`"), it answers "show me exactly what it has on that table":

```bash
ogle show "urn:li:dataset:(dbt,shop.orders,PROD)"          # full field list + types + null% + rows + hash + mute state
ogle show "urn:li:dataset:(dbt,shop.orders,PROD)" --json   # the whole signature, machine-readable
ogle baselines --grep orders --urns | head -1 | xargs ogle show   # drill in straight from the selector
```

The human view lists every field as `path : native_type`, tags each field's **null fraction**
(the quality signal behind QUALITY drift — a column that used to be full and is now mostly null),
and shows the row count, the **full** schema hash, the capture provenance, and whether the
dataset is muted/snoozed. Fields are sorted by path so the same baseline always renders
identically. It keys on an **exact** URN (as `baselines --urns` emits them) and exits **1** when
the dataset isn't watched — a scriptable "no such baseline" so `ogle show X >/dev/null && …`
branches cleanly. A dataset's remembered *drift* lives in `ogle incidents --grep <name>` (incidents
are keyed by drift event, not URN); `show` is strictly the baseline signature + mute state.

### Explaining a drift (`ogle diff`)

Where `ogle show` prints the *stored* baseline, `ogle diff <urn>` compares that baseline against a
**candidate** signatures file — the exact same JSON `ogle check --signatures` reads — and prints the
field-level delta. It's the read-only investigative step after a page: "the baseline says one thing,
this fresh dump says another — what **exactly** changed?" Unlike `check`, it records no incident and
advances no baseline; it only reports:

```bash
ogle diff "urn:li:dataset:(dbt,shop.orders,PROD)" --signatures fresh.json          # human field-level diff
ogle diff "urn:li:dataset:(dbt,shop.orders,PROD)" --signatures fresh.json --json   # machine-readable delta
ogle diff X --signatures fresh.json && echo "no drift"                              # scriptable gate (exit 0/1)
```

The diff calls out fields **added** (`➕`), **removed** (`➖`), and **retyped** (`🔀 int → bigint`),
**null-fraction** moves on surviving fields (`25.0% → 60.0% null`, with a 0.1pp rounding gate so
re-profiling jitter isn't reported as drift), the **row-count** change with its delta, and whether the
**schema hash** flipped. Exit codes are the drift verdict — **0** identical, **1** differences found,
**2** can't compare (URN not watched, absent from the file, or a malformed file) — so preconditions
stay off the 0/1 path and `ogle diff X --signatures f.json && …` branches cleanly.

### Pruning the watch-list (`ogle forget`)

`ogle forget` is the write-side counterpart to `ogle baselines`: when a dataset is
decommissioned in DataHub, its signature would otherwise sit in the watch-list forever (and
any mute on it becomes an orphan pointing at nothing). `forget` drops the baseline **and**
clears any mute/snooze on that URN so `ogle baselines` and the next walk stay honest:

```bash
ogle forget "urn:li:dataset:(dbt,old.orders,PROD)"          # prune one decommissioned dataset
ogle forget URN_A URN_B                                       # batch — hits and misses report per token
ogle baselines --grep staging --urns | ogle forget -         # prune a whole class, no xargs (native on Windows)
ogle baselines --grep old --urns | ogle forget --dry-run -   # preview what WOULD be pruned, change nothing
```

Where `ogle resolve` drops a drift *event* by fingerprint, `forget` drops the *dataset* by
URN — matched **exactly** (the `--urns` selector emits them whole, so there's no prefix to
disambiguate). A lone `-` reads URNs from stdin so the `baselines --urns` pipe runs without
`xargs`; an unknown or empty token is a reportable miss (`not watched`), never a mass wipe.
Incidents are left untouched — a remembered drift outlives the dataset row and is cleared via
`ogle resolve`, not here. `--dry-run` previews the exact per-token outcome without writing.

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
ogle incidents --grep customers         # find drift by keyword (title or fingerprint, case-insensitive)
ogle incidents --stale 7d               # only drift NOT seen in the last 7 days (resolve candidates)
ogle incidents --fresh 1h               # only drift seen within the last hour (currently-active set)
ogle incidents --fresh 7d --stale 1h    # window: seen between 7d and 1h ago (--fresh + --stale compose)
ogle incidents --min-severity high --serving-only   # filters compose (AND)
ogle incidents --sort count             # order by recurrence (most-recurring drift first)
ogle incidents --sort datasets          # order by blast radius (most datasets first)
ogle incidents --sort recent            # order by freshness (most-recently-seen drift first)
ogle incidents --limit 5                # triage cap: only the top 5 worst (severity, then recurrence)
ogle incidents --summary                # aggregate rollup instead of the per-incident list
ogle incidents --fail-on high           # exit 1 if any remembered incident is HIGH+ (health gate)
ogle incidents --serving-only --fingerprints | xargs ogle resolve   # batch-resolve a filtered set
ogle incidents --stale 30d --fingerprints | ogle resolve -          # prune drift that stopped recurring
```

Each incident line carries the **age of its most recent sighting** (`last seen 3h ago`, or
`just now` for a fresh one) when Ogle has a timestamp for it — the temporal signal that tells a
recurring, still-happening problem apart from one that quietly stopped. Incidents remembered by an
older Ogle (before age-tracking) simply omit the age rather than fake one.

Once a drift has **recurred** (seen ≥2×), the line also carries **how long it has been standing**
since its *first* sighting (`first seen 2w ago · last seen 3h ago`) — the **longevity** axis. This
is written once at first detection and never moved, so it measures the whole life of the drift,
which recurrence `count` alone can't: a burst seen 5× in an hour and a drift festering unresolved
for three weeks both read as "seen 5×" without it. A single-sighting incident omits the clause
(first == last, so it would just echo "last seen"); `--json` still carries the raw `first_seen`
epoch for every timed record (`null`/absent on a legacy store), alongside `last_seen`.

`--summary` collapses the (filtered) set into an aggregate rollup — total remembered, the
severity break-out, the serving-path count with its own severity split, recurring / total
sightings, and the **open-drift age span** (`oldest open drift: 12d ago · freshest: 30m ago`):
the same stalest/freshest `last_seen`-derived ages `ogle status` reports, so the two rollups read
identically and an operator can tell a live incident (freshest = minutes) from a resolve-candidate
festering for weeks (stalest = 12d) without leaving the summary. `--json` carries the raw seconds
as `oldest_incident_age_seconds`/`freshest_incident_age_seconds` (both `null` on a legacy/untimed
store — "no data", never a fabricated age).

Alongside that recency span, the rollup also carries the **standing-age span** — the rollup-level
counterpart to the per-incident longevity axis: `longest-standing: 3w ago · newest: 3h ago`, the
oldest/newest `first_seen`-derived ages. Where the recency span says how recently drift *recurred*,
this says how long it has been *standing* since first appearing, so a chronic incident recurring
30m ago but first seen three weeks ago (`oldest open drift: 30m ago` yet `longest-standing: 3w ago`)
reads as the festering problem it is — the two spans diverge exactly when it matters. Because
`first_seen ≤ last_seen`, the longest-standing age is always ≥ the stalest recency age. `--json`
carries `longest_standing_incident_age_seconds`/`newest_incident_standing_age_seconds` (`null` on an
untimed store), and `ogle status` prints the same line + fields so snapshot and summary stay in
lockstep.

The `--min-severity {low,medium,high}`, `--serving-only`, and `--min-count N` filters mirror
`check --fail-on` so a busy operator can focus on what pages first — `--min-count` surfaces the
chronic drift that keeps recurring despite being "seen." When a filter empties a non-empty memory,
Ogle says so (`no incidents match the filter (N remembered)`) rather than implying nothing is
tracked.

`--grep TEXT` is the text axis: it keeps only incidents whose **title or fingerprint** contains
`TEXT` (case-insensitive) — the way to pull one dataset or drift phrase out of a large memory
(`ogle incidents --grep customers`). Because it also matches the fingerprint, a fingerprint
**prefix** works as a needle, mirroring how `ogle resolve` accepts prefixes. It ANDs with every
other filter (`--grep customers --min-severity high` = high-severity customers drift only). An
all-whitespace needle matches nothing rather than everything, so a fat-fingered `--grep ""` never
masquerades as "all incidents."

`--stale AGE` is the *staleness* axis: it keeps only incidents **last seen longer ago than
AGE** — the drift that stopped recurring, i.e. the resolve/forget candidates. `AGE` is a compact
duration (`30m`, `12h`, `7d`, `2w`; a bare number is rejected so `--stale 3` can't ambiguously mean
seconds or days). It composes with every other filter and with `--fingerprints`, so
`ogle incidents --stale 30d --fingerprints | ogle resolve -` batch-clears drift Ogle hasn't seen in
a month. A legacy incident with **no recorded age** can't be proven stale, so `--stale` skips it
rather than guessing.

`--fresh AGE` is the mirror image: it keeps only incidents **last seen within AGE** — the drift
still recurring lately, i.e. the currently-active set an operator triages during a live incident
(`ogle incidents --fresh 1h --serving-only` = serving-path drift seen in the last hour). Same
duration grammar and the same "no recorded age can't be proven fresh, so it's skipped" rule as
`--stale`. Because the two bound opposite ends, they compose into a **window**:
`ogle incidents --fresh 7d --stale 1h` shows only drift last seen between 7 days and 1 hour ago.

`--sort {severity,count,datasets,recent}` picks the ordering axis. The default `severity` is the
triage order (worst first, recurrence as tiebreak); `count` surfaces the most-recurring
(chronic/flapping) drift first; `datasets` surfaces the widest blast radius first; `recent`
surfaces the freshest sighting first (untimed/legacy records sink last). Every axis
falls back to severity then fingerprint for a stable, deterministic order. It also redefines
what `--limit` calls the "top N" (e.g. `--sort count --limit 3` = the three most-recurring),
and is ignored by `--summary`/`--fail-on` (a rollup and a floor gate don't depend on order).

`--limit N` caps the list to the top N after the chosen `--sort` — a triage shortcut for
"just show me the N that matter most" on a store with a long tail. It composes with the filters
(applied *after* them) and the header says `Top N of M` so a capped view never reads as the whole
set. It's deliberately ignored by `--summary` (the rollup describes the whole filtered set — capping
it would under-count).

`--summary` swaps the per-incident list for an at-a-glance rollup — total remembered, a count
per severity, how many touch a serving path, how many are recurring (seen ≥2×), and the total
sighting count. When anything is serving, the serving-path count appends its own severity split
`(🔴 high · 🟠 medium · 🟡 low · • unknown)` — the same page-worthy cross-tab `ogle status` and the
`ogle_incidents_serving_by_severity` gauge surface, so the 🔴 *high-serving* page can't hide inside
a flat total (the four buckets sum back to `serving-path`). It describes the *filtered* set, so it
composes with the triage flags (e.g. `ogle incidents --summary --serving-only` summarizes only the
serving-path drift). Add `--json` for a machine-readable `{"summary": {...}}` shape.

`--fail-on {low,medium,high}` turns the read-only listing into a **health gate on the drift
*memory***. Where `check --fail-on` gates on *new* drift surfaced this run, `incidents --fail-on`
exits **1** whenever Ogle still *remembers* open drift at or above the floor — so a nightly job
stays red until the drift is actually fixed (`ogle resolve`d, or its fingerprint stops recurring),
even on ticks that surface nothing new. It evaluates the *filtered* set (composes with
`--min-severity`/`--serving-only`/`--min-count`) but is **independent of `--limit`** — a display cap
never changes the pass/fail verdict — and works with the list, `--summary`, and `--json` views alike.

`--fingerprints` turns the read side into a **selector for the write side**: instead of the
human list it prints just the surviving fingerprints, one per line, honoring every filter plus
`--sort`/`--limit`. Pipe them straight into `ogle resolve` to batch-triage a whole class of drift
in one command — `ogle incidents --serving-only --min-severity high --fingerprints | xargs ogle
resolve` (or `| ogle resolve -` to skip `xargs`, native on Windows). It stays **silent on an
empty set** (so a pipe gets a clean empty stream, not a bogus token), overrides
`--summary`/`--json` (this *is* the machine form), and still returns the `--fail-on` gate code so
one invocation can both list a batch and signal a failing gate. (`ogle resolve` trims surrounding
whitespace, so the pipe works even where a shell adds a trailing CR.)

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
ogle incidents --serving-only --fingerprints | ogle resolve -   # stdin pipe, no xargs
```

A lone `-` token reads fingerprints from **stdin** (whitespace-separated), so the selector
pipe runs **without `xargs`** — the native form on Windows, where `xargs` isn't a built-in.
Blank lines and trailing CRs are dropped, and an empty stdin resolves nothing (never a mass
wipe); `-` composes with literal tokens and `--dry-run` just like any other argument.

Ambiguous prefixes fail loud (exit 2) with the list of candidates so the operator retypes
with more characters — Ogle never guesses which incident to drop. Unknown/already-forgotten
tokens report `_not remembered_` but aren't an error (exit 0), so replaying a list is safe.

`--dry-run` **previews** exactly what a resolve would drop without touching the store —
each hit prints the full fingerprint it would retire (a prefix resolves to its full id, so
the preview is exact), while misses and ambiguity behave identically. Handy as a safety
check before committing a batch pulled from `ogle incidents --fingerprints`:

```bash
ogle incidents --serving-only --min-severity high --fingerprints \
  | xargs ogle resolve --dry-run     # see what WOULD be dropped, change nothing
```

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

A single unreachable or rejected entity never strands the batch: `--write-back` catches a
per-entity backend error, records it, and keeps going. But a swallowed write is a silent
blind spot — the asset you think is flagged isn't. So any failed write is **loud on stderr**,
naming every entity (and tag) that did **not** reach DataHub, while stdout keeps the terse
`(N failed — see stderr)` note and `--json` stays a clean parseable blob. The same applies to
`--retract-cleared`. Never trust "done" without checking stderr.

#### Clearing the flag when drift heals (`--retract-cleared`)

A tag you never take back becomes noise: after a fix ships, `ogle-drift-flagged` lingers on
an asset that's healthy again, and operators learn to ignore it. `--retract-cleared` closes
the loop — on any run (including a **healthy** one, which is exactly when recovery happens),
it **removes** Ogle's drift tag from every dataset that was checked and is now clean, plus
each downstream `mlModel` that is no longer fed by *any* still-drifting dataset. A model that
still sits downstream of live drift keeps its flag, so clearing never hides a real incident.

```bash
# stamp on drift, and clear the stamp once an asset recovers — the tag stays trustworthy
ogle check --gms http://localhost:8080 --discover --store live.json --write-back --retract-cleared
```

Retraction is idempotent (an asset Ogle never tagged is a cheap no-op via a read-before-write),
strips the flat tag **and** every `ogle-drift-<severity>` variant, and requires a live walk
(`--gms`). It reads DataHub as the source of truth, so it needs no local record of what was
flagged before.

#### Previewing catalog writes first (`--catalog-dry-run`)

Letting an agent write to a shared production catalog is a trust decision. `--catalog-dry-run`
lets you make it with eyes open: it computes the **exact** tags `--write-back` / `--retract-cleared`
would stamp or clear and prints them, then stops — **nothing is written to DataHub**. This is the
outward-facing twin of the `--dry-run` on `ogle resolve` / `ogle forget` (which preview local
store edits); it still needs a live walk (`--gms`), since the plan follows the real dataset→model
lineage.

```bash
# See what Ogle WOULD tag on a new incident — change nothing:
ogle check --gms http://localhost:8080 --discover --store live.json --write-back --catalog-dry-run

# Preview a retraction the same way; --json adds "dry_run": true so a wrapper can tell it from an apply:
ogle check --gms http://localhost:8080 --discover --store live.json --retract-cleared --catalog-dry-run --json
```

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
the underlying `ogle check` exit code, so a cron/Task can still branch on it.

**Delivery is verified — a broken pager can't hide behind a green "PAGED".** If the
`--notify-cmd` command can't be spawned (not on `PATH`) or exits non-zero, `ogle watch`
does **not** silently report success: it writes a loud `PAGE DELIVERY FAILED — <reason>`
line to stderr **and** falls back to printing the full drift page to stderr, so the alert
is never lost. The `ogle check` exit code is still preserved (a delivered-or-not page
doesn't rewrite the contract), so wrap the tick in your own alert on that stderr marker if
your scheduler swallows stderr. Example cron line (every 15 min):

```cron
*/15 * * * * cd /srv/ogle && ogle watch --notify-cmd /usr/local/bin/page-me -- \
  --store /var/lib/ogle/baselines.json --gms http://localhost:8080 --discover
```

**Riding out a transient pager blip (`--notify-retries`).** A momentary pager outage — a
5xx from the paging API, a DNS hiccup — shouldn't turn a *recoverable* failure into a
permanently dropped alert. `--notify-retries N` re-attempts a failed delivery up to `N`
times with a linear backoff (1s, 2s, …) before falling back to the loud stderr page. Only
genuine delivery failures are retried: a bug in a custom notifier (a `ValueError`, not a
`NotifyError`) is *not* re-run, because it won't self-heal. When every attempt fails, the
`delivery_error` names how many were made (`… (after 3 attempts)`), so the exhausted-retry
case is distinguishable from a single try.

```bash
ogle watch --notify-cmd /usr/local/bin/page-me --notify-retries 2 -- \
  --store baselines.json --signatures my-signatures.json
```

**Bounding a hung pager (`--notify-timeout`).** A retry rides out a pager that *fails*
fast, but a pager that *hangs* — a network send with no timeout of its own — would wedge
the whole watch tick and stall the scheduler behind it. `--notify-timeout SECONDS` bounds
each delivery attempt: the pager child is killed at the limit and the timeout is treated as
a transient failure, so it flows through the same fallback and — paired with
`--notify-retries` — is re-attempted rather than dropped. Applies only to `--notify-cmd`;
default is no limit.

```bash
ogle watch --notify-cmd /usr/local/bin/page-me --notify-timeout 15 --notify-retries 2 -- \
  --store baselines.json --signatures my-signatures.json
```

**Structured output for a monitor (`--json`).** A scheduler that wants to gate on the
tick without scraping the human line adds `--json`: the outcome goes to stdout as one
object, the `PAGE:` fallback and delivery-failure notice still go to stderr, and the
`ogle check` exit code is preserved.

```bash
ogle watch --json --notify-cmd /usr/local/bin/page-me -- \
  --store baselines.json --signatures my-signatures.json
```

```json
{
  "watch": {
    "action": "page",
    "exit_rc": 1,
    "paged": true,
    "page_delivered": false,
    "delivery_error": "could not run notifier ['page-me']: ...",
    "delivery_failed": true,
    "report_text": "## 🔴 HIGH drift across 1 dataset ...",
    "error_text": ""
  }
}
```

`delivery_failed` is the load-bearing field: `true` means a page was dispatched but never
delivered (the silently-dropped-alert failure the stderr marker also flags) — gate on it
directly instead of parsing prose. `exit_rc` folds the `ogle check` contract into the
payload so the verdict survives a stdout capture forwarded over a log/message bus, and
`report_text` carries the same narrative the notifier would have — a JSON consumer can
forward the drift story itself.

**Validating the wiring without paging anyone (`--dry-run`).** Before you trust a new
watch line, you want to know it *would* fire — without actually paging a human while you
test. `--dry-run` runs the check and makes the full paging decision, but never invokes the
notifier: no `--notify-cmd`, no stderr `PAGE:` block. The `ogle check` exit code is still
preserved, so a `1` still tells you an incident is standing.

```bash
ogle watch --dry-run --json -- --store baselines.json --signatures my-signatures.json
# -> "would_page": true, "paged": false, "dry_run": true, "exit_rc": 1  (nothing delivered)
```

The human line reads `WOULD PAGE (dry-run, no page sent)` instead of `PAGED`. In `--json`,
`would_page` is the paging *decision* decoupled from dispatch: outside `--dry-run` it always
equals `paged`; under it, `would_page` can be `true` while `paged` is `false`. (Note the
check itself still updates its incident memory — `--dry-run` suppresses the *page*, not the
store bookkeeping, so a debounced standing incident won't re-fire on the next real tick.)

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Status

🚧 In active build for the DataHub Agent Hackathon.
Submission window: Jul 6 – **Aug 10, 2026 @ 5 PM ET**.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
