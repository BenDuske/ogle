<!-- Real captured output of `ogle check --freshness-max-age 24h` on the offline demo's
freshness scenario. Reproduce: run `ogle demo` (keyless, no DataHub) and read section 3,
"The 9th dimension: freshness" — this block is exactly what it emits. Not hand-written.

This is the silent stall the other eight dimensions can't see. The source below is
byte-for-byte unchanged from its baseline — same schema, rows, nulls, means, spread and
range, so schema/volume/quality/distribution/mean/stdev/range/shape all stay green — but
its profile timestamp stopped advancing. An hourly ETL stalled ~3.75 days ago; every model
retraining on this table is quietly learning a frozen world. Only freshness fires. It is
opt-in and clock-driven (a nightly table and a streaming source have very different SLAs),
so it arms per-deployment via `--freshness-max-age`; here 24h. See
`examples/alerts/churn-orders-drift.md` for the loud/silent multi-dimension companion. -->

## 🔴 HIGH drift across 1 dataset on a serving path

**1 finding** across 1 dataset — 1 🔴 high · ⚠️ serving path impacted

### b2fd91.events_hourly
- 🔴 **freshness** — data is stale: last profiled 2026-07-19T02:00:00Z (90.0h ago, SLA 24.0h) — feed likely stalled [serving]

**What to check**
- check whether the upstream load/profile job is still running — a stale timestamp with unchanged rows usually means the feed silently stopped, so every retrain is learning yesterday's data

_incident d32cb30722ae04fc_

_checked 1 dataset(s)._
