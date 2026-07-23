# W2 — Signature + anomaly scorer

The analytic core Ogle runs on a schedule to catch silent drift in the datasets feeding
production ML models. Two pure modules, no live DataHub required to develop or test:

- `ogle.signature` — `DatasetSignature`, the cheap fingerprint Ogle persists between runs
  (schema shape + row count + per-field null fractions + per-field distinct-value
  fractions). `schema_hash` is order-independent so two fetches of an unchanged schema
  compare equal. Round-trips through `to_dict` / `from_dict` for the baseline store (Aegis
  memory in W2b); `field_unique_fractions` is optional, so pre-existing baselines load clean.
- `ogle.scorer` — `score_dataset(baseline, current, cfg, serving, now)` returns `DriftFinding`s
  across six dimensions, sorted most-severe first.

## Drift dimensions

| Kind         | Fires when                                                        | Severity logic |
|--------------|-------------------------------------------------------------------|----------------|
| SCHEMA       | a source column is removed / retyped (added-only = LOW)           | remove/retype = HIGH |
| VOLUME       | row count changes past `volume_rel_threshold` (default ±30%)      | banded by how far past threshold; collapse-to-empty = HIGH |
| QUALITY      | a field's null fraction rises ≥ `null_fraction_abs_threshold` (0.20) | banded by max delta |
| DISTRIBUTION | a field's distinct-value fraction *drops* ≥ `unique_fraction_drop_threshold` (0.30) | banded by max drop |
| MEAN         | a numeric field's mean shifts (either way) ≥ `mean_rel_threshold` (0.25) vs baseline | banded by max relative shift |
| FRESHNESS    | a dataset's profile timestamp (`computed_at`) ages past `freshness_max_age_seconds` relative to `now` | banded by how far past the SLA |

**MEAN** is the numeric-covariate-shift dimension: a feature's values move (sensor
recalibration, a unit/currency change, or a genuine population shift) while its schema, row
count, null rate and cardinality all stay green — so every other score is quiet, yet each
retrain learns a distribution the deployed model never saw. It scores the *relative* shift of
each numeric field's mean and flags **both directions** (a feature that doubled or halved has
moved either way), unlike DISTRIBUTION's drop-only rule. A field with no mean on either side is
skipped, and a baseline mean whose magnitude is below `mean_zero_floor` (~0) is skipped too —
a relative shift against zero is undefined and would page on trivial wiggle. Never guessed.
When both sides carry a stdev, each flagged move is annotated with the pooled-sigma **Cohen's
d** effect size and its conventional band (`d=+4.0 large`, `d=+0.1 negligible`) — the first
two-sample signal, so an operator can tell a genuine population shift from a relative move that
is noise against the field's own spread. Alongside the band, the same `d` is rendered as a
**probability of superiority** — `P(new>old)`, the common-language effect size (McGraw & Wong's
CLES) — the chance a value drawn from the new distribution outranks one drawn from the old
(`P(new>old) = 0.5*(1 + erf(d/2))`, computed from `d` alone). It turns the abstract `d=+0.1
negligible` into `P(new>old)=53%` (a coin flip — likely a false page) and `d=+4.0 large` into
`P(new>old)=100%` (a near-certain shift), so an operator triages without carrying Cohen's
thresholds in their head. Purely enrichment: it labels the finding, never gates it (a field
without a stdev is still flagged, just without a `d`).

**DISTRIBUTION** is the cardinality half of true distribution drift: it catches a
categorical/feature column collapsing onto one value (a stuck upstream default — the model
keeps training on a feature that now carries no signal) and an id/join key losing uniqueness
(a fan-out join duplicating rows). Only a *drop* pages — cardinality rising is usually benign
variety, and flagging it would be noise on the serving path we work to keep quiet.

**FRESHNESS** is the silent-stall dimension the others structurally cannot see: when an
ETL quietly stops, the rows, schema, null and unique fractions all stay put, so every other
score is green — yet the data is stale and each retrain learns yesterday's world. The one
signal that moves is the profile timestamp. It is **opt-in** (`freshness_max_age_seconds` /
`--freshness-max-age`, default OFF) because a nightly table and a streaming source have very
different SLAs, and **clock-injected** (`now` is passed in, never read inside the scorer) so
the module stays pure and deterministic. Age is measured against `now`; an unparseable/absent
`computed_at`, no SLA, or no `now` all leave it unscored (never guessed), and a future stamp
(clock skew) clamps to age 0 rather than reading negative.

**Serving escalation:** when the dataset feeds a deployed (IN_SERVICE) model — e.g. the
Task #2 `churn_predictor` behind `churn_predictor_endpoint` — every finding is bumped one
severity step (`escalate_when_serving`, on by default). Drift on a serving path is
production-affecting; that is the whole reason Ogle exists.

## Guarantees

- **Pure / deterministic** — no network or LLM, and no clock *read* inside the scorer: the
  freshness dimension takes `now` as an injected argument, so same inputs → same findings and
  a finding computed on Halcyon reproduces in CI.
- **Never guesses** — a dimension with data missing on either side (no profile, new field
  with no baseline) is skipped, not flagged.

## Where it plugs in

```
DataHub walk ──▶ build_signature() ──▶ baseline store (Aegis memory)
                                    └─▶ score_dataset() ──▶ DriftFinding[]
                                                              ├─▶ narrate() ──▶ incident    [W2b ✅]
                                                              └─▶ tag write-back            [W3]
```

## W2b — Narrative writer (`ogle.narrative`)

`narrate(findings, llm=None, owners=None)` turns a scoring run into one human-facing
incident. Two layers, so the useful part never depends on a model being reachable:

- **Deterministic core** — `build_incident(findings, owners=None)` folds findings into an
  `Incident` (overall severity = worst finding, serving flag, deduped datasets, ranked
  "what to check" actions, and a stable `fingerprint`). `render_markdown` prints it. Pure →
  unit-testable with no DataHub and no LLM.
- **LLM polish (optional)** — pass an `llm` callable and `narrate` hands it a *grounded*
  prompt (`build_llm_prompt`) built from the already-computed facts, with an explicit
  "use only these facts, don't invent severity/datasets" instruction. If the model is
  absent, raises, or returns empty, it falls back to the deterministic markdown — an alert
  always goes out. Model-agnostic (Aegis-local Qwen or Anthropic fallback).

**Owner attribution (who to page)** — `build_incident`/`narrate` take an optional
`owners` map (`urn -> owner names`, from DataHub's **Ownership** aspect). Each affected
dataset renders a `👤 owner(s): …` line, and the grounded LLM prompt is told to name who
to page (but forbidden to invent an owner). Ownership is **presentation only**: it is
deliberately *not* part of the `fingerprint`, so re-assigning an owner never re-pages a
still-open incident. Owners are normalized (restricted to the incident's own URNs, stripped,
deduped, empties dropped) so a stray or blank owner can't leak into an alert. `run_drift_check`
threads the same `owners` map through to both the incident object and the narrative.

The live path now populates it end-to-end: `walker.walk_model` fetches each dataset's
**Ownership** aspect (`WalkerBackend.get_ownership`, added to `DataHubBackend`), folds it to
display names via `extract_owner_names` (`urn:li:corpuser:jane.doe → jane.doe`,
`urn:li:corpGroup:data-eng → data-eng`, deduped, order-preserved) and carries it on
`WalkResult.owners` (`urn -> names`, unioned across walks). `ogle check`'s live branch passes
`walk_result.owners` into `run_drift_check`, so a live serving-path incident renders the
`👤 owner:` line automatically; offline `--signatures` mode has no owner source and cleanly
omits it. `get_ownership` is probed with `getattr`, so a pre-ownership custom backend degrades
to "no owners" rather than erroring.

**`fingerprint`** = order-independent SHA over the set of `(urn, kind, severity)` triples.
It lets Aegis's salience memory dedup a recurring incident across scheduled runs (same
datasets + same drift = one open issue, not a new alert every tick) and changes exactly
when Ben would call it a different situation (drift resolves, worsens, or a new dataset
joins). `tests/test_narrative.py` (34): short-name parsing, severity rollup, serving flag,
fingerprint order-independence + change-on-worsen, deterministic markdown, action dedup,
grounded prompt, the LLM seam (used / fallback-on-raise / fallback-on-empty), and owner
attribution (rendering, plural grammar, normalization, URN-restriction, fingerprint
invariance, `to_dict`, and prompt grounding).

## W2b — Baseline store (`ogle.store`) + pipeline (`ogle.pipeline`)

Drift detection is a diff, so Ogle needs memory between runs. `BaselineStore` is that memory
and the concrete "Aegis memory" backing:

- **Baselines** — the last `DatasetSignature` per URN, so the next run can diff against it.
- **Seen incidents** — the set of incident fingerprints already reported, with an observation
  count, so a scheduled loop pages Ben *once* per drift, not every 10 minutes.
- **Durable** — a single JSON file written atomically (temp + `os.replace`), so a crash
  mid-walk can't corrupt good baselines. Versioned on disk (refuses to misread a stale file).
  Clock-free and diffable. When Aegis salience memory lands (W3), `BaselineStore` is the seam
  that swaps a JSON path for an Aegis-backed KV without the scorer/pipeline changing.

`run_drift_check(store, current, serving_urns, cfg, llm, update_baselines)` is the I/O-free
end-to-end seam the live DataHub walk plugs into — it takes the freshly-pulled signatures and
the store and returns a `DriftReport`:

- **New datasets** (no baseline) are seeded, never scored — you can't diff against nothing.
- **Scored datasets** are diffed via `score_dataset` (serving URNs escalated), findings merged
  and ranked worst-first across all datasets.
- **`should_alert`** is the single field a scheduled loop needs: true only on a *new* incident
  (dedup runs against the store), so repeats debounce automatically.
- **Baselines advance** to the current state after scoring (skippable with `update_baselines=False`
  for a read-only probe); the advance happens only after scoring so a mid-batch failure can't
  half-update state.

```
DataHub walk ──▶ build_signature() ──▶ BaselineStore.get_baseline()   [W2b ✅]
                                    └─▶ run_drift_check() ──▶ score_dataset() ──▶ narrate()
                                                          └─▶ DriftReport{should_alert}
```

## Tests

`tests/test_signature.py` (11) + `tests/test_scorer.py` (18): order-independent hashing,
add-vs-remove-vs-retype severity, volume collapse + thresholds, null-spike vs improvement,
serving escalation on/off, finding ordering, degrade-gracefully on missing profiles.
`tests/test_store.py` (18): put/get/upsert, incident record+count+forget, save/load roundtrip,
atomic write (no tmp leftover), missing-file→fresh store, version rejection, parent-dir creation.
`tests/test_pipeline.py` (16): first-run seeding, unchanged→no drift, volume-collapse incident,
serving escalation, cross-dataset merge+rank, alert-once-then-debounce, baseline advance vs
read-only probe, LLM seam used/fallback-on-raise, serializable report, empty-batch heartbeat.
Fault-injection verified (breaking escalation *and* the debounce each flip a test red). 80 tests
green. Run: `py -3.14 -m pytest -q`.
