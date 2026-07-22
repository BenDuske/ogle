# W2 — Signature + anomaly scorer

The analytic core Ogle runs on a schedule to catch silent drift in the datasets feeding
production ML models. Two pure modules, no live DataHub required to develop or test:

- `ogle.signature` — `DatasetSignature`, the cheap fingerprint Ogle persists between runs
  (schema shape + row count + per-field null fractions + per-field distinct-value
  fractions). `schema_hash` is order-independent so two fetches of an unchanged schema
  compare equal. Round-trips through `to_dict` / `from_dict` for the baseline store (Aegis
  memory in W2b); `field_unique_fractions` is optional, so pre-existing baselines load clean.
- `ogle.scorer` — `score_dataset(baseline, current, cfg, serving)` returns `DriftFinding`s
  across four dimensions, sorted most-severe first.

## Drift dimensions

| Kind         | Fires when                                                        | Severity logic |
|--------------|-------------------------------------------------------------------|----------------|
| SCHEMA       | a source column is removed / retyped (added-only = LOW)           | remove/retype = HIGH |
| VOLUME       | row count changes past `volume_rel_threshold` (default ±30%)      | banded by how far past threshold; collapse-to-empty = HIGH |
| QUALITY      | a field's null fraction rises ≥ `null_fraction_abs_threshold` (0.20) | banded by max delta |
| DISTRIBUTION | a field's distinct-value fraction *drops* ≥ `unique_fraction_drop_threshold` (0.30) | banded by max drop |

**DISTRIBUTION** is the cardinality half of true distribution drift: it catches a
categorical/feature column collapsing onto one value (a stuck upstream default — the model
keeps training on a feature that now carries no signal) and an id/join key losing uniqueness
(a fan-out join duplicating rows). Only a *drop* pages — cardinality rising is usually benign
variety, and flagging it would be noise on the serving path we work to keep quiet.

**Serving escalation:** when the dataset feeds a deployed (IN_SERVICE) model — e.g. the
Task #2 `churn_predictor` behind `churn_predictor_endpoint` — every finding is bumped one
severity step (`escalate_when_serving`, on by default). Drift on a serving path is
production-affecting; that is the whole reason Ogle exists.

## Guarantees

- **Pure / deterministic** — no clock, network, or LLM. Same inputs → same findings, so a
  finding computed on Halcyon reproduces in CI.
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
