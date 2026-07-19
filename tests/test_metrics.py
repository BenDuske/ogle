"""Tests for `ogle metrics` — the Prometheus text-exposition sibling of `status`.

`metrics` renders the same watch-list + incidents + mutes rollup as `status`, but in
Prometheus v0.0.4 text format so a monitoring stack can scrape Ogle's drift memory over
time. These tests pin the format contract (HELP/TYPE per family, gauge type, no `_total`
suffix, label escaping) and the numeric values against a known store.
"""

import json

from ogle.cli import _render_prometheus, build_parser, main
from ogle.signature import build_signature
from ogle.store import BaselineStore

CUSTOMERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
ORDERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"


def _sig(urn=CUSTOMERS_URN, **kw):
    kw.setdefault("schema_fields", [("id", "int"), ("email", "string")])
    kw.setdefault("row_count", 1000)
    return build_signature(urn, **kw)


def _parse_prom(text):
    """Parse exposition text into ({sample_key: value}, {family: type}, help_lines)."""
    samples = {}
    types = {}
    helps = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("# HELP "):
            helps.add(line.split()[2])
            continue
        if line.startswith("# TYPE "):
            _, _, name, typ = line.split(maxsplit=3)
            types[name] = typ
            continue
        key, _, value = line.rpartition(" ")
        samples[key] = value
    return samples, types, helps


def _seed_store(store_path):
    """Two baselines (one unknown-row) + three incidents (one recurring/serving) + a mute."""
    s = BaselineStore(path=store_path)
    s.baselines[CUSTOMERS_URN] = _sig(row_count=1000)  # 2 fields, 1000 rows
    s.baselines[ORDERS_URN] = _sig(ORDERS_URN, row_count=None)  # 2 fields, unknown rows
    s.record_incident("hi", severity="high", title="HIGH", datasets=3, serving=True)
    s.record_incident("hi", severity="high", title="HIGH", datasets=3, serving=True)  # recurring
    s.record_incident("md", severity="medium", title="MED", datasets=2, serving=False)
    s.record_incident("lo", severity="low", title="LOW", datasets=1, serving=False)
    s.save()
    return s


# ---- format contract ---------------------------------------------------------------
def test_metrics_empty_store_is_valid_all_zero(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    assert main(["metrics", "--store", str(store_path)]) == 0
    out = capsys.readouterr().out
    samples, types, helps = _parse_prom(out)

    # ogle_up is always present and 1 — a scrape target liveness signal even on an empty store.
    # (Look up by family prefix: the store-path label is escaped, so an exact key is brittle.)
    up = [v for k, v in samples.items() if k.startswith("ogle_up{")]
    assert up == ["1"]
    # Every level gauge reads 0 on a fresh store (no wall of ambiguity — real zeros).
    assert samples["ogle_watching_datasets"] == "0"
    assert samples["ogle_incidents_serving"] == "0"
    assert samples["ogle_muted_active"] == "0"
    for s in ("high", "medium", "low", "unknown"):
        assert samples[f'ogle_incidents_remembered{{severity="{s}"}}'] == "0"


def test_metrics_every_family_declares_help_and_type_once(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    assert main(["metrics", "--store", str(store_path)]) == 0
    out = capsys.readouterr().out
    _, types, helps = _parse_prom(out)

    families = {
        "ogle_up",
        "ogle_watching_datasets",
        "ogle_watching_fields",
        "ogle_watching_rows",
        "ogle_watching_rows_unknown",
        "ogle_incidents_remembered",
        "ogle_incidents_serving",
        "ogle_incidents_recurring",
        "ogle_incidents_sightings",
        "ogle_muted_active",
    }
    # Every family has exactly one HELP and one TYPE, and every one is a gauge.
    assert helps == families
    assert set(types) == families
    assert all(t == "gauge" for t in types.values())
    # Each HELP/TYPE header appears once — no duplicate family declarations.
    assert out.count("# TYPE ogle_incidents_remembered ") == 1


def test_metrics_no_counter_total_suffix(tmp_path, capsys):
    # These are point-in-time levels (gauges); Prometheus reserves `_total` for counters.
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    assert main(["metrics", "--store", str(store_path)]) == 0
    out = capsys.readouterr().out
    metric_names = [
        ln.split()[2] for ln in out.splitlines() if ln.startswith("# TYPE ")
    ]
    assert metric_names, "expected at least one metric family"
    assert not any(n.endswith("_total") for n in metric_names)


# ---- numeric values against a known store ------------------------------------------
def test_metrics_values_match_store(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    assert main(["metrics", "--store", str(store_path)]) == 0
    samples, _, _ = _parse_prom(capsys.readouterr().out)

    # watch-list rollup: 2 datasets, 4 fields, 1000 known rows, 1 unknown-row baseline.
    assert samples["ogle_watching_datasets"] == "2"
    assert samples["ogle_watching_fields"] == "4"
    assert samples["ogle_watching_rows"] == "1000"
    assert samples["ogle_watching_rows_unknown"] == "1"
    # incidents rollup: high(recurring, 2 sightings) + medium + low.
    assert samples['ogle_incidents_remembered{severity="high"}'] == "1"
    assert samples['ogle_incidents_remembered{severity="medium"}'] == "1"
    assert samples['ogle_incidents_remembered{severity="low"}'] == "1"
    assert samples['ogle_incidents_remembered{severity="unknown"}'] == "0"
    assert samples["ogle_incidents_serving"] == "1"  # only "hi" is serving
    assert samples["ogle_incidents_recurring"] == "1"  # "hi" seen twice
    assert samples["ogle_incidents_sightings"] == "4"  # 2 + 1 + 1


def test_metrics_matches_status_json_numbers(tmp_path, capsys):
    """metrics is a re-shape of status — the numbers must be identical, not drift apart."""
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)

    assert main(["status", "--store", str(store_path), "--json"]) == 0
    st = json.loads(capsys.readouterr().out)["status"]

    assert main(["metrics", "--store", str(store_path)]) == 0
    samples, _, _ = _parse_prom(capsys.readouterr().out)

    assert samples["ogle_watching_datasets"] == str(st["baselines"]["watching"])
    assert samples["ogle_watching_fields"] == str(st["baselines"]["fields"])
    assert samples["ogle_muted_active"] == str(st["muted"])
    assert (
        samples['ogle_incidents_remembered{severity="high"}']
        == str(st["incidents"]["by_severity"]["high"])
    )
    assert samples["ogle_incidents_serving"] == str(st["incidents"]["serving"])


# ---- pure render helper: label escaping + exit code --------------------------------
def test_render_escapes_store_label():
    totals = {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0}
    inc = {
        "total": 0,
        "by_severity": {"high": 0, "medium": 0, "low": 0, "unknown": 0},
        "serving": 0,
        "recurring": 0,
        "total_sightings": 0,
    }
    # A Windows path carries backslashes + a quote, both illegal raw in a label value.
    text = _render_prometheus(totals, inc, 0, r'C:\ogle\store".json')
    # Backslashes doubled, double-quote backslash-escaped — a scraper won't mis-tokenize.
    assert r'ogle_up{store="C:\\ogle\\store\".json"} 1' in text


def test_metrics_always_exits_zero_with_high_incidents(tmp_path, capsys):
    # Unlike status/incidents --fail-on, a metrics scrape must never fail on data levels.
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)  # has a high-severity serving incident
    assert main(["metrics", "--store", str(store_path)]) == 0
    assert capsys.readouterr().out.strip()  # produced output


def test_parser_wires_metrics_to_cmd(tmp_path):
    ns = build_parser().parse_args(["metrics", "--store", "x.json"])
    assert ns.func.__name__ == "cmd_metrics"
    assert ns.store == "x.json"
    assert ns.output is None  # stdout by default


# ---- --output (atomic textfile-collector target) -----------------------------------
def test_output_flag_parses_with_short_alias():
    ns = build_parser().parse_args(["metrics", "-o", "/tmp/ogle.prom"])
    assert ns.output == "/tmp/ogle.prom"
    ns = build_parser().parse_args(["metrics", "--output", "/tmp/ogle.prom"])
    assert ns.output == "/tmp/ogle.prom"


def test_output_writes_file_not_stdout(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    out = tmp_path / "ogle.prom"
    rc = main(["metrics", "--store", str(store_path), "--output", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    # stdout carries no metrics — they went to the file; a collector redirecting stdout
    # elsewhere must not also get the exposition body.
    assert captured.out.strip() == ""
    # a confirmation lands on stderr (won't pollute a piped stdout / the .prom file)
    assert str(out) in captured.err
    body = out.read_text(encoding="utf-8")
    assert "ogle_up{" in body
    assert 'ogle_incidents_serving 1' in body


def test_output_matches_stdout_render(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    # stdout render
    main(["metrics", "--store", str(store_path)])
    stdout_body = capsys.readouterr().out
    # file render
    out = tmp_path / "ogle.prom"
    main(["metrics", "--store", str(store_path), "--output", str(out)])
    file_body = out.read_text(encoding="utf-8")
    # Same exposition, modulo the trailing newline _emit and the file both append.
    assert file_body.strip() == stdout_body.strip()
    assert file_body.endswith("\n")  # exposition wants a newline-terminated final sample


def test_output_is_atomic_no_temp_left(tmp_path):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    out = tmp_path / "sub" / "ogle.prom"  # also exercises parent-dir creation
    assert main(["metrics", "--store", str(store_path), "--output", str(out)]) == 0
    assert out.exists()
    # temp file used during the atomic write is cleaned up — only the final file remains
    leftovers = [p.name for p in out.parent.iterdir() if p.name.startswith(".ogle-metrics-")]
    assert leftovers == []


def test_output_overwrites_existing_atomically(tmp_path):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)
    out = tmp_path / "ogle.prom"
    out.write_text("stale content that must be fully replaced\n", encoding="utf-8")
    assert main(["metrics", "--store", str(store_path), "--output", str(out)]) == 0
    body = out.read_text(encoding="utf-8")
    assert "stale content" not in body
    assert "ogle_up{" in body
