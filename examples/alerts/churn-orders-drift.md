<!-- Real captured output of `ogle check` on the offline demo fixtures.
Reproduce: seed with examples/demo/healthy-signatures.json, then check against
examples/demo/drifted-signatures.json (see README Quickstart). Not hand-written —
this is exactly what a scheduler would page to an on-call engineer. -->

## 🔴 HIGH drift across 1 dataset on a serving path

**3 findings** across 1 dataset — 3 🔴 high · ⚠️ serving path impacted

### b2fd91.orders
- 🔴 **schema** — schema changed: retyped ['order_total:DOUBLE->STRING'] [serving]
- 🔴 **volume** — row count shrank 1250000 -> 40000 (-97%) [serving]
- 🔴 **quality** — null rate spiked on order_total (0%->42%) [serving]

**What to check**
- check the upstream transform that renamed/dropped the column, and any feature reading it before the next training run
- check whether the upstream load job stopped, backfilled, or double-wrote before trusting downstream features
- check the source for a partial/failed load — a null spike usually means an upstream join or extract broke, not real data

_incident fd6f829c77ff9fb4_

_checked 2 dataset(s)._
