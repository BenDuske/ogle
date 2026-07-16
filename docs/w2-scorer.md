# W2 — Signature + anomaly scorer

The analytic core Ogle runs on a schedule to catch silent drift in the datasets feeding
production ML models. Two pure modules, no live DataHub required to develop or test:

- `ogle.signature` — `DatasetSignature`, the cheap fingerprint Ogle persists between runs
  (schema shape + row count + per-field null fractions). `schema_hash` is order-independent
  so two fetches of an unchanged schema compare equal. Round-trips through `to_dict` /
  `from_dict` for the baseline store (Aegis memory in W2b).
- `ogle.scorer` — `score_dataset(baseline, current, cfg, serving)` returns `DriftFinding`s
  across three dimensions, sorted most-severe first.

## Drift dimensions

| Kind    | Fires when                                                        | Severity logic |
|---------|-------------------------------------------------------------------|----------------|
| SCHEMA  | a source column is removed / retyped (added-only = LOW)           | remove/retype = HIGH |
| VOLUME  | row count changes past `volume_rel_threshold` (default ±30%)      | banded by how far past threshold; collapse-to-empty = HIGH |
| QUALITY | a field's null fraction rises ≥ `null_fraction_abs_threshold` (0.20) | banded by max delta |

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

`narrate(findings, llm=None)` turns a scoring run into one human-facing incident. Two
layers, so the useful part never depends on a model being reachable:

- **Deterministic core** — `build_incident` folds findings into an `Incident` (overall
  severity = worst finding, serving flag, deduped datasets, ranked "what to check" actions,
  and a stable `fingerprint`). `render_markdown` prints it. Pure → unit-testable with no
  DataHub and no LLM.
- **LLM polish (optional)** — pass an `llm` callable and `narrate` hands it a *grounded*
  prompt (`build_llm_prompt`) built from the already-computed facts, with an explicit
  "use only these facts, don't invent severity/datasets" instruction. If the model is
  absent, raises, or returns empty, it falls back to the deterministic markdown — an alert
  always goes out. Model-agnostic (Aegis-local Qwen or Anthropic fallback).

**`fingerprint`** = order-independent SHA over the set of `(urn, kind, severity)` triples.
It lets Aegis's salience memory dedup a recurring incident across scheduled runs (same
datasets + same drift = one open issue, not a new alert every tick) and changes exactly
when Ben would call it a different situation (drift resolves, worsens, or a new dataset
joins). `tests/test_narrative.py` (20): short-name parsing, severity rollup, serving flag,
fingerprint order-independence + change-on-worsen, deterministic markdown, action dedup,
grounded prompt, and the LLM seam (used / fallback-on-raise / fallback-on-empty).

## Tests

`tests/test_signature.py` (11) + `tests/test_scorer.py` (18): order-independent hashing,
add-vs-remove-vs-retype severity, volume collapse + thresholds, null-spike vs improvement,
serving escalation on/off, finding ordering, degrade-gracefully on missing profiles.
Fault-injection verified (breaking escalation flips a test red). Run: `py -3.14 -m pytest -q`.
