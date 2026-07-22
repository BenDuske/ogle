# Deploying Ogle

Two ways to run Ogle, in order of setup cost:

1. **Offline / keyless** — no DataHub, no Docker, no API key. Feeds pre-computed
   signature files. This is how the test suite runs and how a scheduled job can feed
   signatures it pulled elsewhere.
2. **Live** — walks a real DataHub GMS. Needs the DataHub quickstart (Docker) and the
   `acryl-datahub` SDK.

The pure pipeline (`signature` / `scorer` / `store` / `narrative` / `pipeline`) needs
**none** of the DataHub SDK — it is only imported lazily inside `walker.DataHubBackend`
and `writeback`, so the offline path installs nothing beyond Ogle itself.

---

## 1. Offline / keyless deploy

```bash
pip install -e ".[dev]"
pytest -q                                   # 660+ tests, no Docker, no keys

# reproduce the sample drift alert end-to-end
ogle check --store demo.json --signatures examples/demo/healthy-signatures.json   # seeds, exit 0
ogle check --store demo.json --signatures examples/demo/drifted-signatures.json   # alerts, exit 1
```

The store file (`demo.json`) is Ogle's memory between runs — last-known-good signature
per dataset plus seen-incident fingerprints for debounce. It is written atomically
(temp + `os.replace`), so a killed run never leaves a half-updated store.

---

## 2. Live deploy against a DataHub quickstart

### Python version matters

`acryl-datahub` needs a Python with prebuilt wheels — **3.12 works, 3.14 does not**
(pydantic-core has no 3.14 wheel and the source build fails). Stand up an isolated venv:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -e ".[datahub]"     # pulls acryl-datahub>=1.6.0
```

On Linux/WSL use `.venv/bin/python` instead of `.venv/Scripts/python`.

### Bring up DataHub + seed demo lineage

```bash
datahub docker quickstart                                 # 6 containers, GMS on :8080
.venv/Scripts/python scripts/inject-ml-lineage.py --gms http://localhost:8080
```

This seeds the demo ML lineage (`ogle_demo.churn_predictor` IN_SERVICE +
`ogle_demo.demand_forecast`) on top of the showcase datasets so a walk finds real
model → feature-table → source-table lineage.

### Walk it

```bash
# seed baselines from the live graph
.venv/Scripts/ogle check --gms http://localhost:8080 --discover --store live.json    # exit 0

# probe for drift (read-only; write tags back on a real incident)
.venv/Scripts/ogle check --gms http://localhost:8080 --discover --store live.json \
    --no-update --write-back
```

`--discover` selects only IN_SERVICE models. `--write-back` stamps
`urn:li:tag:ogle-drift-flagged` on a drifted dataset **and** its downstream serving
model; re-runs are idempotent no-ops, so a scheduled loop won't double-tag.

See [`live-verification.md`](live-verification.md) for the full transcript of the six
verified live stages (connection, walk, store round-trip, drift alert, tag write-back,
console safety).

---

## 3. Running on a schedule

`ogle watch` is one scheduler tick: it runs `ogle check` and acts on the exit code —
**page once on a new incident (exit 1), stay quiet when healthy (exit 0)**, log-only on
a usage/input error (exit 2). The scheduler (cron / Windows Task Scheduler / APScheduler)
owns the loop; Ogle owns the single tick and the page-once debounce.

```bash
# every 15 min: page a real pager, ride out transient pager outages, bound a hung pager
*/15 * * * * cd /srv/ogle && ogle watch \
    --notify-cmd /usr/local/bin/page-me --notify-retries 2 --notify-timeout 15 -- \
    --store /srv/ogle/baselines.json --gms http://localhost:8080 --discover
```

- Arguments after `--` forward verbatim to the underlying `ogle check`.
- The notifier is injected, so the Apache-2.0 repo stays messaging-agnostic. Wire an
  OpenClaw `message send` (or `mail`, PagerDuty, etc.) at the `--notify-cmd` layer.
- `--json` emits a structured tick outcome for a monitor; `--dry-run` decides-but-suppresses
  to validate the wiring without paging a human.

---

## Windows console note

On a Windows cp1252 console, redirected/piped stdout defaults to cp1252, which would
mangle the emoji severity markers. Ogle promotes stdout/stderr to UTF-8 at startup
(`errors='replace'` as a net) and per-write encodes safely, so `ogle demo > alert.md`
and every check/status pipe produce the same UTF-8 bytes as the committed fixtures on
any terminal. No configuration required.
