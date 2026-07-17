# Devpost Submission — Ogle (Track: Production ML Agents)

*Copy-paste-ready draft for the Devpost form. Deadline: **Aug 10, 2026 @ 5 PM ET**.*

---

## Project title
**Ogle — the ML lineage agent that just can't stop staring**

## Tagline (elevator pitch)
Walks your DataHub lineage graph on a schedule, catches silent training-data drift
before it hits prod, writes a root-cause narrative your on-call engineer can act on
in 30 seconds, and remembers past incidents so it gets sharper over time.

## Track
**Production ML Agents** *(Wildcard-eligible fallback)*

---

## Project Story

### Inspiration
*(Ben's voice — final personal pass in W3. Factual seed: every ML team lives one
silent training-data change away from a bad model in prod. The lineage already
exists in DataHub; what's missing is an agent that walks it on a schedule and
catches the change before the deploy.)*

### What it does
Ogle walks your DataHub lineage graph on a schedule and catches silent
training-data drift before it reaches production. Given a deployed model, it
traverses upstream through the lineage graph (model → feature tables → source
tables) and, at each hop, computes a lightweight **signature** — row-count delta,
schema hash, and coarse distribution stats — which it compares against the last
known-good state. When an above-threshold anomaly on a serving-path asset is
detected, Ogle writes a root-cause **narrative** an on-call engineer can act on in
30 seconds: what changed, when, who owns it, which downstream models are exposed,
and a direct link to inspect. Findings are written **back into DataHub as tags** on
the affected assets, and into a salience-ranked memory so the agent stops re-paging
on incidents it has already reported. It runs as a one-shot `ogle check` or as a
scheduled `ogle watch` loop that pages **once per incident**.

### How we built it
Ogle is a Python agent with five composable stages (see `docs/architecture.md`):
a **scheduler** (cron/APScheduler-friendly `watch` tick), a **lineage-walk** that
traverses DataHub's graph via the MCP server / Skills layer, a **scorer** that
computes per-asset signatures and anomaly scores, a **narrative writer** that turns
a flagged walk plus DataHub ownership/docs context into an actionable alert, and an
**alert writer** that persists the narrative and writes an `ogle:flagged` tag back
to the graph. The agent's memory — "Ogle-Brain" — is built on the
[Aegis MemoryAgent](https://github.com/BenDuske/qwen-memoryagent): a forgetful,
salience-ranked store of facts, episodes, and preferences so past false positives
and real incidents sharpen future walks. The whole suite is keyless and
Docker-free to test — every network call is monkeypatched — so `pytest -q` runs
green with no DataHub and no API key (167 tests at submission).

### Challenges we ran into
- **Scoping drift to what matters.** A naïve diff flags every table that moves.
  Ogle scores against the *serving path* and severity so a Monday-bouncing
  dashboard doesn't page while a genuine schema+volume+quality shift on a serving
  table does — verified with the offline demo, which drifts one table's
  serving-path signals while leaving a sibling clean to prove the scoping.
- **Making the judge path actually runnable.** Early quickstarts referenced scripts
  and commands that didn't exist; we rewrote both the README and this submission's
  quickstart to verified commands and shipped a keyless, reproducible offline drift
  demo (`examples/demo/` → captured alert in `examples/alerts/`).
- **DataHub write-back maturity + local Quickstart on Windows/WSL** — surfaced and
  documented in `docs/live-verification.md` and `docs/DEPLOY.md`.

### What we learned
Statelessness is the ceiling on most lineage tooling — the moment an agent
*remembers* which alerts were noise, its signal-to-noise flips from "another
dashboard" to "a triage assistant." And a hackathon judge's first five minutes are
the demo: a keyless, no-Docker reproduction of the core alert is worth more than any
architecture diagram, so we invested in making `pytest` and the offline demo run
clean from a fresh clone.

### What's next
Richer scorers (true distribution tests beyond the coarse proxy), agent-to-agent
Ogle deployments that share incident memory across teams, and publishing Ogle's
DataHub Skill wrapper back upstream as an OSS contribution.

---

## Testing Instructions

```bash
# 1. Run tests — no Docker, no keys (all network calls are monkeypatched)
pip install -e ".[dev]"
pytest -q

# 2. Offline demo — reproduces the sample drift alert end-to-end
ogle check --store demo.json --signatures examples/demo/healthy-signatures.json   # seeds, exit 0
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json   # alerts, exit 1

# 3. (optional) Live demo against a real DataHub quickstart
datahub docker quickstart
python scripts/inject-ml-lineage.py --gms http://localhost:8080
ogle check --gms http://localhost:8080 --discover --store live.json
```

The offline demo in step 2 needs no DataHub and no API key; its captured output is
`examples/alerts/churn-orders-drift.md`. See `docs/live-verification.md` for a full
transcript of the live path against the DataHub quickstart.

---

## Submission checklist

| Requirement | Status |
|---|---|
| Public repo | https://github.com/BenDuske/ogle |
| **Apache 2.0** license | ✅ present at repo root |
| Demo video (< 3 min, YouTube public/unlisted) | 🟡 W3 |
| Text description | 🟢 drafted (technical story done; Inspiration awaits Ben's voice pass) |
| Setup instructions in README | 🟡 W1 → refined W3 |
| Sample outputs in `examples/` | ✅ `examples/alerts/` + runnable `examples/demo/` fixtures |
| Live demo URL or Docker Compose one-liner | 🟢 `ogle demo` — one keyless command, zero setup |
| Optional: OSS contribution back to DataHub | 🟡 W2–W3 stretch |

---

## Judging-rubric mapping

| Criterion | How Ogle scores |
|---|---|
| **Use of DataHub** | Reads lineage + ownership via MCP · writes tags back to the graph (rubric explicitly rewards this) |
| **Technical Execution** | End-to-end walk in one command · keyless pytest suite · Docker Compose repro |
| **Originality** | Memory-augmented lineage agent — most submissions will be stateless |
| **Real-World Usefulness** | Catches silent drift before deploy; every ML team ships without this |
| **Submission Quality** | Under-3-min video, clean README, working `examples/` |
| **Bonus: OSS contribution** | Ogle's DataHub Skill wrapper published upstream (stretch) |
