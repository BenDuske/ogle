# Ogle — Demo Video Script & Shot-List (< 3 min)

**Purpose:** turnkey recording guide for the Devpost submission demo video
(Track: *Production ML Agents*, deadline **Aug 10, 2026, 5 PM ET**). Target length
**2:45–2:55** (hard cap 3:00, YouTube public/unlisted).

**Two-track format.** Each scene is split into a **SCREEN** block (what the recording
shows + exact commands, with cue-timing) and a **VOICE** block (verbatim narration,
written for text-to-speech read at ~155 wpm in Ben's cloned voice — no branches, no
stage directions, no parenthetical asides). Total narration budget ≈ 430 words.

**Global assumptions (pinned so narration is deterministic):**
- Local Ollama `qwen3:latest` is up before recording. Narration says "written by a
  local model" and it will be. If Ollama is down at record time, bring it up — do
  not rewrite the script around a fallback.
- Terminal at repo root `C:\Users\bendu\ogle`, venv active, monospace font sized so
  text is readable at 1080p, buffer cleared between scenes.
- `docs/screenshots/` open in an image viewer, ready to flip through for Scene 1.
- Numbers in the VOICE track are spelled out where TTS trips on digits.

---

### Scene 0 — Cold open (0:00 → 0:12)

**SCREEN**
- 0:00 Title card OR the README top line: `The ML lineage agent that just can't stop staring.`
- Hold static — no motion.

**VOICE** (30 words · ~12 s)
> Every ML team lives one silent training-data change away from a bad model in
> production. The lineage already exists in DataHub. What's missing is an agent that
> watches it. This is Ogle.

---

### Scene 1 — The problem, in the graph (0:12 → 0:42)

**SCREEN**
- 0:12 Cut to DataHub lineage screenshots. Flip through, roughly ten seconds each:
  1. `docs/screenshots/03-churn-predictor-lineage.png`
  2. `docs/screenshots/06-demand-forecast-lineage.png`
  3. `docs/screenshots/09-feature-table-sources.png`
- Zoom or highlight the edge from model → feature table → source table on each frame.

**VOICE** (76 words · ~29 s)
> Here is a real serving model in DataHub — a churn predictor — walked back through
> its feature tables to the source tables that feed it. If one of those upstream
> tables silently shifts — a schema change, a volume drop, a distribution that
> quietly moves — the model degrades and nobody gets paged. Ogle walks this exact
> graph on a schedule and catches the shift before the next deploy.

---

### Scene 2 — One command, the alert fires (0:42 → 1:32)

**SCREEN**
- 0:42 Cut to a clean terminal.
- 0:44 Type and run:
  ```bash
  ogle demo
  ```
- Let the seed + drifted re-check render. Auto-scroll to the 🔴 HIGH serving-path
  alert (7 findings / 2 datasets).
- 1:15 Pause on the alert block; briefly highlight `EXIT CODE: 1`.

**VOICE** (128 words · ~50 s)
> No Docker. No API key. One command. Ogle seeds healthy baselines — exit zero —
> then re-checks a drifted snapshot through the same code path a live DataHub walk
> feeds. And it fires. Two serving-path tables drifted at once. One loudly — schema,
> volume, and quality. One silently, in its value distributions — cardinality, mean,
> standard deviation, and range. That's seven of Ogle's nine drift dimensions in a
> single high-severity alert. The silent covariate shift is the one that normally
> ships to production unnoticed. Ogle scores it green on schema, red on
> distribution. Exit code one — that is what pages your on-call.

---

### Scene 3 — Root cause + write-back, still keyless (1:32 → 2:15)

**SCREEN**
- 1:32 Clear the terminal. Type and run:
  ```bash
  ogle demo --narrate --write-back --write-back-severity
  ```
- Scroll to the root-cause narrative paragraph; hold ~4 seconds.
- Scroll to the `urn ← tag` write-back preview showing `ogle-drift-flagged` and
  `ogle-drift-high`; hold ~4 seconds.

**VOICE** (95 words · ~37 s)
> Add two flags and you get the other two flagship features from the same keyless
> command. First, a root-cause narrative your on-call engineer can act on in about
> thirty seconds — what moved, where, and why it matters — written by a local
> model. Second, the tag write-back — the exact ogle-drift-flagged and
> ogle-drift-high annotations Ogle would stamp back onto the drifted dataset inside
> DataHub, shown here as a dry run that touches no catalog. The alert becomes
> evidence that lives next to the data.

---

### Scene 4 — It remembers (debounce) (2:15 → 2:42)

**SCREEN**
- **PRECONDITION (do this before rolling):** ensure `demo.json` does **not** exist
  yet — `rm -f demo.json`. Scene 4 depends on a fresh store: the first `check` seeds
  the baseline (exit 0), the second pages the new drift (exit 1), the third is
  debounced (exit 0). A leftover `demo.json` from a rehearsal already holds the
  incident, so the very first `check` finds pre-existing drift and the sequence
  inverts to `1 → 0 → 0` — silently breaking the "it remembers" narration on camera.
  (Verified 2026-08-04: stale store → `1 → 0 → 0`; fresh store → `0 → 1 → 0`.)
- 2:15 Clear the terminal. Run these three commands back-to-back, letting each
  finish:
  ```bash
  ogle check --store demo.json --signatures examples/demo/healthy-signatures.json
  ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
  ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
  ```
- 2:35 Highlight the three exit codes in sequence: `0 → 1 → 0`.

**VOICE** (66 words · ~26 s)
> And it remembers what it already flagged. The first run seeds. The drift fires
> once — exit one. Run the same drift again and it is debounced to exit zero. You
> get paged once per incident, not every tick. That is the difference between an
> alert and alert spam.

---

### Scene 5 — Close (2:42 → 2:55)

**SCREEN**
- 2:42 Cut back to the README title card or the repo URL: `github.com/BenDuske/ogle`.
- Hold static.

**VOICE** (30 words · ~12 s)
> Ogle. It walks your lineage, catches the silent drift, writes the root cause,
> and remembers. Apache two point zero, on GitHub. Thanks for watching.

---

## Word-count budget (audit)

| Scene | Words | Read @155 wpm | Cume |
|---|---:|---:|---:|
| 0 Cold open | 30 | 12 s | 0:12 |
| 1 Problem in graph | 76 | 29 s | 0:41 |
| 2 Alert fires | 128 | 50 s | 1:31 |
| 3 Narrative + write-back | 95 | 37 s | 2:08 |
| 4 Debounce | 66 | 26 s | 2:34 |
| 5 Close | 30 | 12 s | 2:46 |
| **Total narration** | **425** | **2:46** | — |

Screen-only pauses (Scene 1 flips, Scene 2 command render, Scene 3 hold, Scene 4
exit-code highlight) absorb another 4–8 s of dead air — landing target is **2:50 ±
5 s**, comfortably under Devpost's hard 3:00 cap.

## Command cheat-sheet (paste-ready, in order)

```bash
ogle demo
ogle demo --narrate --write-back --write-back-severity
rm -f demo.json   # Scene 4 needs a FRESH store or the exit codes invert to 1→0→0
ogle check --store demo.json --signatures examples/demo/healthy-signatures.json
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json
```

> `demo.json` is written to the current dir by Scene 4's first command; delete it
> after recording (it is gitignored, so it will not show in `git status`, but tidy
> anyway).

## Recording pipeline (Argo-driven, Ben's cloned voice)

1. **Narration audio.** Feed each Scene's VOICE block verbatim to ElevenLabs voice
   `VMEzE95otIZyOaJaKcL0` (Ben Duske clone, IVC), `eleven_multilingual_v2`, stability
   0.5 / similarity 0.8. Produce six WAVs — one per scene — for clean cut points.
2. **Screen capture.** Record the terminal at 1080p, cursor visible, per the SCREEN
   blocks above. Auto-timed against each scene's read length ± the pause budget.
3. **Mux.** Line each WAV up to the SCREEN cue-timing; add a 250 ms fade between
   scenes; render a single MP4 under 3:00.
4. **Publish.** Upload unlisted to YouTube. Paste the URL into the "Demo video"
   row of `docs/DEVPOST-SUBMISSION.md` and into the Devpost submission's video
   field.

## Timing guardrails

- If the render lands over 2:55, cut Scene 4 to two commands (seed + fire) and
  keep the "get paged once per incident" line in Scene 4's VOICE — that buys ~10 s
  with no substance loss.
- Never let the render exceed 3:00. Devpost rejects over-length.
