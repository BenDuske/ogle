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
*(To be written in W3, once we have the demo working end-to-end.)*

### What it does
*(To be written in W3 — pulls from README.md.)*

### How we built it
*(To be written in W3 — DataHub MCP client, lineage-walk scheduler,
signature/anomaly scorers, Aegis-based memory, LLM narrative, tag write-back.)*

### Challenges we ran into
*(To be written in W3 — placeholders: DataHub Quickstart on Windows/WSL,
MCP write-back maturity, distinguishing real drift from expected seasonality.)*

### What we learned
*(To be written in W3.)*

### What's next
*(To be written in W3 — richer scorers, agent-to-agent Ogle deployments,
publish the DataHub Skill contribution back upstream.)*

---

## Testing Instructions

```bash
# 1. Bring up the stack (Docker required)
docker compose up -d

# 2. Seed the demo dataset
python scripts/seed_demo_dataset.py

# 3. Run tests (keyless — network calls monkeypatched)
pip install -e ".[dev]"
pytest -q

# 4. Live demo — run one walk cycle end-to-end
python -m ogle walk --model demo.recommender.v1
```

Optional live-LLM path documented in `docs/DEPLOY.md`.

---

## Submission checklist

| Requirement | Status |
|---|---|
| Public repo | https://github.com/BenDuske/ogle |
| **Apache 2.0** license | ✅ present at repo root |
| Demo video (< 3 min, YouTube public/unlisted) | 🟡 W3 |
| Text description | 🟡 W3 (this file) |
| Setup instructions in README | 🟡 W1 → refined W3 |
| Sample outputs in `examples/` | 🟡 W2 |
| Live demo URL or Docker Compose one-liner | 🟡 W1 |
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
