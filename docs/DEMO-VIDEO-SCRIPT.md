# Ogle — Demo Video Script & Shot-List (< 3 min)

**Purpose:** turnkey recording guide for the Devpost submission demo video
(Track: *Production ML Agents*, deadline **Aug 10, 2026, 5 PM ET**). Target length
**2:45–2:55** (hard cap 3:00, YouTube public/unlisted). One take is fine; everything here is
keyless and reproducible — no Docker, no API key, no live DataHub required for the core beats.

**Setup before recording (30 sec, off-camera):**
- Terminal at repo root `C:\Users\bendu\ogle`, venv active, font bumped so text is legible at 1080p.
- Have `docs/screenshots/` open in an image viewer for the B-roll beat (Scene 2).
- Optional: start local Ollama (`qwen3:latest`) if you want the *live* LLM narration in Scene 3;
  if it's not up, `--narrate` falls back to the deterministic summary — still looks clean, just say
  "root-cause summary" instead of "LLM-written."

Narration is written to be read at a calm pace. On-screen action and spoken line are aligned per scene.

---

### Scene 0 — Cold open (0:00–0:15)
**On screen:** title card or just the README top (`The ML lineage agent that just can't stop staring.`).
**Say:**
> "Every ML team lives one silent training-data change away from a bad model in prod. The lineage
> already exists in DataHub — what's missing is an agent that *watches* it. This is Ogle."

---

### Scene 1 — The problem, in the graph (0:15–0:45)
**On screen:** the DataHub lineage screenshots — flip through
`docs/screenshots/03-churn-predictor-lineage.png` → `06-demand-forecast-lineage.png` →
`09-feature-table-sources.png` (model → feature tables → source tables).
**Say:**
> "Here's a real serving model in DataHub — a churn predictor — walked back through its feature
> tables to the source tables that feed it. If one of those upstream tables silently shifts —
> a schema change, a volume drop, a distribution that quietly moves — the model degrades and
> nobody gets paged. Ogle walks this exact graph on a schedule and catches that *before* the deploy."

---

### Scene 2 — One command, the alert fires (0:45–1:35)
**On screen:** clear terminal, type and run:
```bash
ogle demo
```
Let the full report render. Scroll to the 🔴 HIGH serving-path alert (7 findings / 2 datasets).
**Say (while it runs):**
> "No Docker, no API key, one command. Ogle seeds healthy baselines — exit zero — then re-checks a
> drifted snapshot through the *same* code path a live DataHub walk feeds."
**Say (pointing at the alert):**
> "And it fires. Two serving-path tables drifted at once — one *loudly*: schema, volume, quality.
> One *silently*, in its value distributions — cardinality, mean, standard deviation, range. That's
> seven of Ogle's nine drift dimensions in a single HIGH alert. The silent covariate shift is the one
> that normally ships to prod unnoticed — Ogle scores it green-on-schema but red-on-distribution."
> *(beat)* "Exit code one — that's what pages your on-call."

---

### Scene 3 — Root cause + write-back, still keyless (1:35–2:20)
**On screen:** run:
```bash
ogle demo --narrate --write-back --write-back-severity
```
Show the root-cause paragraph, then the `urn ← tag` write-back preview.
**Say:**
> "Add two flags and you get the other two flagship features in the same keyless command. First, a
> root-cause narrative your on-call can act on in about thirty seconds — what moved, where, and why it
> matters." *(if Ollama up: "written by a local LLM"; else: "a plain-English root-cause summary")*
> "Second, the tag write-back — the exact `ogle-drift-flagged` and `ogle-drift-high` annotations Ogle
> would stamp back onto the drifted dataset *in* DataHub, shown here as a dry run that touches no
> catalog. The alert becomes evidence that lives next to the data."

---

### Scene 4 — It remembers (debounce) (2:20–2:45)
**On screen:** run the hand-driven signatures path to show debounce:
```bash
ogle check --store demo.json --signatures examples/demo/healthy-signatures.json   # seed, exit 0
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json   # fires, exit 1
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json   # debounced, exit 0
```
Point at the three exit codes: `0 → 1 → 0`.
**Say:**
> "And it remembers what it already flagged. First run seeds. The drift fires once — exit one. Run the
> *same* drift again and it's debounced to exit zero. You get paged once per incident, not every tick.
> That's the difference between an alert and alert *spam*."

---

### Scene 5 — Close (2:45–2:55)
**On screen:** back to README title / repo URL `github.com/BenDuske/ogle`.
**Say:**
> "Ogle. It walks your lineage, catches the silent drift, writes the root cause, and remembers.
> Apache-2.0, on GitHub. Thanks for watching."

---

## Command cheat-sheet (paste-ready, in order)
```bash
ogle demo
ogle demo --narrate --write-back --write-back-severity
ogle check --store demo.json --signatures examples/demo/healthy-signatures.json
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
```
> `demo.json` is written to the current dir by Scene 4's first command; `rm demo.json` after recording
> to keep the tree clean (it's gitignored, so it won't show in `git status`, but tidy anyway).

## Timing guardrails
- If you run long, **cut Scene 4 to two commands** (seed + fire) and mention debounce verbally — that
  buys ~15 sec.
- Keep total under 3:00; Devpost rejects over-length. Aim to land at ~2:50 with breathing room.

## What to upload / where it plugs in
- Upload unlisted or public to YouTube; paste the link into `docs/DEVPOST-SUBMISSION.md` (the
  "Demo video" checklist row, currently 🟡) and into the Devpost submission's video field.
