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
        "ogle_muted_permanent",
        "ogle_muted_snooze_next_expiry_seconds",
        "ogle_incidents_last_seen_min_age_seconds",
        "ogle_incidents_last_seen_max_age_seconds",
        "ogle_store_age_seconds",
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


def _drop_wallclock(body):
    """Strip the wall-clock heartbeat SAMPLE line so two invocations compare deterministically.

    `ogle_store_age_seconds` = now - store mtime, sampled per invocation, so the stdout and
    file renders (two separate `main()` calls microseconds apart) legitimately differ on that
    one value. The HELP/TYPE headers stay — only the value line is clock-dependent.
    """
    return "\n".join(
        ln for ln in body.splitlines()
        if not (ln and not ln.startswith("#") and ln.split()[0] == "ogle_store_age_seconds")
    )


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
    # Same exposition, modulo the trailing newline _emit and the file both append and the
    # per-invocation wall-clock heartbeat value.
    assert _drop_wallclock(file_body).strip() == _drop_wallclock(stdout_body).strip()
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


# ---- staleness age gauges (freshest/stalest incident) ------------------------------
def _inc_summary(records):
    from ogle.cli import _incident_summary

    return _incident_summary(records)


def test_age_bounds_none_when_no_timed_incidents():
    from ogle.cli import _incident_age_bounds

    # legacy/untimed incidents (no last_seen) → no age series at all
    assert _incident_age_bounds([{"count": 1}, {"count": 2}], now=100.0) is None


def test_age_bounds_min_is_freshest_max_is_stalest():
    from ogle.cli import _incident_age_bounds

    now = 1000.0
    records = [
        {"last_seen": 990.0},  # 10s ago — freshest
        {"last_seen": 700.0},  # 300s ago — stalest
        {"last_seen": 950.0},  # 50s ago
        {"count": 1},          # untimed — ignored
    ]
    assert _incident_age_bounds(records, now) == (10.0, 300.0)


def test_age_bounds_clamps_future_stamp_to_zero():
    from ogle.cli import _incident_age_bounds

    # a last_seen in the future (clock skew) reads as age 0, never negative
    assert _incident_age_bounds([{"last_seen": 1050.0}], now=1000.0) == (0.0, 0.0)


def test_render_emits_age_gauges_when_timed():
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        0,
        "s.json",
        age_bounds=(10.0, 300.0),
    )
    samples, types, helps = _parse_prom(text)
    assert types["ogle_incidents_last_seen_min_age_seconds"] == "gauge"
    assert types["ogle_incidents_last_seen_max_age_seconds"] == "gauge"
    assert samples["ogle_incidents_last_seen_min_age_seconds"] == "10"
    assert samples["ogle_incidents_last_seen_max_age_seconds"] == "300"


def test_render_declares_age_gauges_but_no_sample_when_untimed():
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        0,
        "s.json",
        age_bounds=None,
    )
    _, types, helps = _parse_prom(text)
    # HELP/TYPE still declared (stable scrape shape) ...
    assert "ogle_incidents_last_seen_min_age_seconds" in helps
    assert "ogle_incidents_last_seen_max_age_seconds" in types
    # ... but NO value line is emitted (honest "no data", not a fake zero age)
    for ln in text.splitlines():
        assert not (
            ln
            and not ln.startswith("#")
            and ln.split()[0].startswith("ogle_incidents_last_seen_")
        )


def test_metrics_cli_emits_age_sample_for_timed_incident(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    # stamp last_seen so the incident is timed; CLI uses wall-clock now, so assert
    # presence + ordering rather than an exact (nondeterministic) age.
    s.record_incident("hi", severity="high", title="H", datasets=1, now=1.0)
    s.save()
    assert main(["metrics", "--store", str(store_path)]) == 0
    samples, _, _ = _parse_prom(capsys.readouterr().out)
    lo = float(samples["ogle_incidents_last_seen_min_age_seconds"])
    hi = float(samples["ogle_incidents_last_seen_max_age_seconds"])
    assert hi >= lo > 0  # a single ancient incident: both large, max >= min


# ---- store-age heartbeat (dead-man's-switch for the monitor itself) -----------------
def test_store_file_age_none_when_file_missing(tmp_path):
    from ogle.cli import _store_file_age

    # No store written yet (first run) → no age series, never a fabricated zero.
    assert _store_file_age(tmp_path / "nope.json", now=1000.0) is None


def test_store_file_age_is_seconds_since_mtime(tmp_path):
    import os

    from ogle.cli import _store_file_age

    p = tmp_path / "baselines.json"
    p.write_text("{}", encoding="utf-8")
    os.utime(p, (500.0, 500.0))  # mtime = 500s
    assert _store_file_age(p, now=800.0) == 300.0


def test_store_file_age_clamps_future_mtime_to_zero(tmp_path):
    import os

    from ogle.cli import _store_file_age

    # mtime ahead of now (clock skew / cloud-sync touch) reads as 0, never negative.
    p = tmp_path / "baselines.json"
    p.write_text("{}", encoding="utf-8")
    os.utime(p, (2000.0, 2000.0))
    assert _store_file_age(p, now=1000.0) == 0.0


def test_render_emits_store_age_gauge_when_known():
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        0,
        "s.json",
        age_bounds=None,
        store_age=42.0,
    )
    samples, types, helps = _parse_prom(text)
    assert types["ogle_store_age_seconds"] == "gauge"
    assert "ogle_store_age_seconds" in helps
    assert samples["ogle_store_age_seconds"] == "42"


def test_render_declares_store_age_gauge_but_no_sample_when_unknown():
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        0,
        "s.json",
        age_bounds=None,
        store_age=None,
    )
    _, types, helps = _parse_prom(text)
    # HELP/TYPE always declared (stable scrape shape) ...
    assert "ogle_store_age_seconds" in helps
    assert types["ogle_store_age_seconds"] == "gauge"
    # ... but NO value line before the first store write (honest "no data").
    for ln in text.splitlines():
        assert not (
            ln
            and not ln.startswith("#")
            and ln.split()[0] == "ogle_store_age_seconds"
        )


def test_metrics_cli_emits_store_age_for_existing_store(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_store(store_path)  # writes the file → mtime exists
    assert main(["metrics", "--store", str(store_path)]) == 0
    samples, types, _ = _parse_prom(capsys.readouterr().out)
    assert types["ogle_store_age_seconds"] == "gauge"
    # A freshly-written store: age is small but real (>= 0), and always present.
    assert float(samples["ogle_store_age_seconds"]) >= 0


def test_metrics_cli_omits_store_age_sample_for_missing_store(tmp_path, capsys):
    # An empty/never-written store still renders (all-zero), but with no store file the
    # heartbeat has no honest value — the family is declared, no sample emitted.
    store_path = tmp_path / "baselines.json"
    assert main(["metrics", "--store", str(store_path)]) == 0
    samples, types, helps = _parse_prom(capsys.readouterr().out)
    assert "ogle_store_age_seconds" in helps
    assert types["ogle_store_age_seconds"] == "gauge"
    assert "ogle_store_age_seconds" not in samples


# ---- mute breakdown: permanent (standing blind spot) + snooze countdown -------------
def _mb(store, now):
    from ogle.cli import _mute_breakdown

    return _mute_breakdown(store, now)


def test_mute_breakdown_splits_permanent_and_snoozed_disjoint():
    s = BaselineStore(path="s.json")
    s.mute(CUSTOMERS_URN)  # permanent
    s.mute(ORDERS_URN, until=1000.0)  # active snooze
    mb = _mb(s, now=500.0)
    assert mb["permanent"] == 1
    assert mb["snoozed"] == 1
    # next-expiry is the countdown to the soonest active snooze (1000 - 500).
    assert mb["next_expiry_seconds"] == 500.0
    # Invariant the metric relies on: permanent + snoozed == active muted count.
    assert mb["permanent"] + mb["snoozed"] == len(s.muted(500.0))


def test_mute_breakdown_excludes_expired_snooze():
    s = BaselineStore(path="s.json")
    s.mute(ORDERS_URN, until=100.0)  # lapses before `now`
    mb = _mb(s, now=500.0)
    assert mb["snoozed"] == 0
    # No active snooze → no countdown (honest None, not a fabricated 0 = "expiring now").
    assert mb["next_expiry_seconds"] is None


def test_mute_breakdown_next_expiry_is_the_soonest_snooze():
    s = BaselineStore(path="s.json")
    s.mute(CUSTOMERS_URN, until=900.0)
    s.mute(ORDERS_URN, until=1500.0)
    mb = _mb(s, now=500.0)
    assert mb["snoozed"] == 2
    assert mb["next_expiry_seconds"] == 400.0  # 900 - 500, the nearer expiry


def test_render_permanent_emits_honest_zero_and_countdown_absent_when_no_snooze():
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        0,
        "s.json",
        mute_breakdown={"permanent": 0, "snoozed": 0, "next_expiry_seconds": None},
    )
    samples, types, helps = _parse_prom(text)
    # Permanent is a count → honest 0 always emitted.
    assert samples["ogle_muted_permanent"] == "0"
    # Countdown family declared for a stable scrape shape, but no sample without a snooze.
    assert "ogle_muted_snooze_next_expiry_seconds" in helps
    assert types["ogle_muted_snooze_next_expiry_seconds"] == "gauge"
    assert "ogle_muted_snooze_next_expiry_seconds" not in samples


def test_render_emits_permanent_count_and_snooze_countdown():
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        3,  # muted_active
        "s.json",
        mute_breakdown={"permanent": 2, "snoozed": 1, "next_expiry_seconds": 3600.0},
    )
    samples, _, _ = _parse_prom(text)
    assert samples["ogle_muted_permanent"] == "2"
    assert samples["ogle_muted_snooze_next_expiry_seconds"] == "3600"


def test_render_omits_breakdown_defaults_to_zero_permanent():
    # Back-compat: callers that don't pass a breakdown still render a valid all-zero split.
    text = _render_prometheus(
        {"watching": 0, "fields": 0, "rows": 0, "unknown_rows": 0},
        _inc_summary([]),
        0,
        "s.json",
    )
    samples, _, helps = _parse_prom(text)
    assert samples["ogle_muted_permanent"] == "0"
    assert "ogle_muted_snooze_next_expiry_seconds" in helps


def test_metrics_cli_reports_permanent_mute_as_blind_spot(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = _seed_store(store_path)
    s.mute(CUSTOMERS_URN)  # permanent — a standing blind spot
    s.mute(ORDERS_URN, until=9_999_999_999.0)  # far-future snooze
    s.save()
    assert main(["metrics", "--store", str(store_path)]) == 0
    samples, _, _ = _parse_prom(capsys.readouterr().out)
    assert samples["ogle_muted_permanent"] == "1"
    assert samples["ogle_muted_active"] == "2"  # permanent + active snooze
    # The countdown is present and positive (snooze lapses in the far future).
    assert float(samples["ogle_muted_snooze_next_expiry_seconds"]) > 0
