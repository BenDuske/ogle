<!-- Real captured output of `ogle check` on the offline demo fixtures.
Reproduce: seed with examples/demo/healthy-signatures.json, then check against
examples/demo/drifted-signatures.json (see README Quickstart). Not hand-written —
this is exactly what a scheduler would page to an on-call engineer. Two serving-path
tables drift at once: orders breaks loudly (schema/volume/quality) while churn_features
drifts silently in its value distributions (distribution/mean/stdev/range) — 7 of Ogle's
8 dimensions in one alert. -->

## 🔴 HIGH drift across 2 datasets on a serving path

**7 findings** across 2 datasets — 6 🔴 high, 1 🟠 medium · ⚠️ serving path impacted

### b2fd91.orders
- 🔴 **schema** — schema changed: retyped ['order_total:DOUBLE->STRING'] [serving]
- 🔴 **volume** — row count shrank 1250000 -> 40000 (-97%) [serving]
- 🔴 **quality** — null rate spiked on order_total (0%->42%, p=0) [serving]

### b2fd91.churn_features
- 🔴 **distribution** — distinct-value fraction dropped on customer_id (100%->55%) [serving]
- 🔴 **stdev** — numeric spread shifted on tenure_days (stdev 420->90, -79%, p=0, 95% CI [0.214x, 0.215x]) [serving]
- 🔴 **range** — numeric range breached on num_support_calls ([0, 12]->[0, 45], +275% of span, 22.0σ past) [serving]
- 🟠 **mean** — numeric mean shifted on monthly_charges (64.8->88.2, +36%, d=+0.7 medium, P(new>old)=70%, H=0.26 small, PSI=0.57 significant, W2=23.5915 moderate, W1emp=21.3889 moderate, KS=0.34 moderate, JS=0.17 small, p=0, 95% CI [+23.2435, +23.5565]) [serving]

**What to check**
- check the upstream transform that renamed/dropped the column, and any feature reading it before the next training run
- check whether the upstream load job stopped, backfilled, or double-wrote before trusting downstream features
- check the source for a partial/failed load — a null spike usually means an upstream join or extract broke, not real data
- check the upstream transform for a stuck default or a fan-out join — a collapsed distinct-value count means the feature lost signal or rows got duplicated
- check the source for a stuck sensor, a clipped/saturated range, or a noisier feed — a collapsed or exploded spread with an intact mean and schema is scale drift that moves feature variance under the model without touching its average
- check the source for an overflow, a unit bug on a subset of rows, or a new outlier regime — values escaping the historical min/max envelope while the mean and spread hold are out-of-bounds features that can silently break a model's input assumptions
- check the source for a unit/scale change, sensor recalibration, or a genuine population shift — a moved mean with intact schema and row count is covariate drift that quietly degrades the model until it's retrained on the new values

_incident d232226d661c10d6_

_checked 3 dataset(s)._
