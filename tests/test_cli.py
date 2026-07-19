"""Unit tests for ogle.cli — the `ogle check` entrypoint over the drift-check pipeline.

Driven entirely through the OFFLINE signatures-file path (`--signatures`), so the whole
command is exercised end-to-end with no `datahub` SDK and no Docker quickstart: load store
-> read signatures JSON -> run_drift_check -> render -> save store -> exit code. Fixtures
use the Task #2 shape where `customers` feeds the deployed `churn_predictor` (serving path).
"""

import json
import time

import pytest

import io

from ogle.cli import _emit, _fmt_age, _parse_age, build_parser, load_signatures_file, main
from ogle.signature import build_signature
from ogle.store import BaselineStore

CUSTOMERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
ORDERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"


def _sig(urn=CUSTOMERS_URN, **kw):
    kw.setdefault("schema_fields", [("id", "int"), ("email", "string")])
    kw.setdefault("row_count", 1000)
    return build_signature(urn, **kw)


def _write_sigs(path, sigs, serving=None):
    """Write a signatures file in the object shape the CLI accepts."""
    payload = {"signatures": [s.to_dict() for s in sigs]}
    if serving is not None:
        payload["serving_urns"] = list(serving)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---- load_signatures_file: shapes + errors ----------------------------------------
def test_load_object_shape(tmp_path):
    f = _write_sigs(tmp_path / "s.json", [_sig()], serving=[CUSTOMERS_URN])
    sigs, serving = load_signatures_file(f)
    assert [s.urn for s in sigs] == [CUSTOMERS_URN]
    assert serving == [CUSTOMERS_URN]


def test_load_bare_list_shape(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps([_sig().to_dict()]), encoding="utf-8")
    sigs, serving = load_signatures_file(f)
    assert [s.urn for s in sigs] == [CUSTOMERS_URN]
    assert serving == []


def test_load_roundtrips_signature_fields(tmp_path):
    original = _sig(row_count=42, field_null_fractions={"email": 0.1})
    f = _write_sigs(tmp_path / "s.json", [original])
    sigs, _ = load_signatures_file(f)
    assert sigs[0].row_count == 42
    assert sigs[0].field_null_fractions == {"email": 0.1}
    assert sigs[0].schema_hash == original.schema_hash


def test_load_missing_file_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_signatures_file(tmp_path / "nope.json")


def test_load_bad_json_raises_valueerror(tmp_path):
    f = tmp_path / "s.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_signatures_file(f)


def test_load_signature_missing_urn_raises(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"signatures": [{"row_count": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing a urn"):
        load_signatures_file(f)


def test_load_wrong_toplevel_type_raises(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps("just a string"), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON list or object"):
        load_signatures_file(f)


# ---- exit codes: the contract a cron wrapper branches on --------------------------
def test_first_run_seeds_and_exits_zero(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs)])
    assert rc == 0  # new dataset seeded, nothing to alert
    assert store.exists()  # baseline persisted for next run
    assert "seeded 1 new dataset" in capsys.readouterr().out


def test_unchanged_exits_zero(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig()}).save()
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs)])
    assert rc == 0
    assert "No drift" in capsys.readouterr().out


def test_check_warns_and_recovers_on_corrupt_store(tmp_path, capsys):
    # A corrupt store must not crash the scheduled run; it re-baselines and warns on stderr.
    store = tmp_path / "baselines.json"
    store.write_text("}{ not json", encoding="utf-8")
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs)])
    assert rc == 0  # first run against a fresh store: seed, don't page
    err = capsys.readouterr().err
    assert "WARNING" in err and "quarantined" in err
    assert store.with_name(store.name + ".corrupt").exists()  # forensic copy kept
    assert store.exists()  # clean store re-written in place


def test_new_incident_exits_one(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig(row_count=1000)}).save()
    # volume collapse -> a fresh incident -> should_alert -> exit 1
    sigs = _write_sigs(tmp_path / "s.json", [_sig(row_count=0)])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs)])
    assert rc == 1


def test_repeat_incident_debounced_to_zero(tmp_path):
    """A second identical run must NOT re-alert — the scheduled-loop debounce."""
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig(row_count=1000)}).save()
    drift = _write_sigs(tmp_path / "s.json", [_sig(row_count=0)])

    first = main(["check", "--store", str(store), "--signatures", str(drift)])
    assert first == 1  # new incident
    # Baseline advanced to row_count=0; re-present the SAME collapsed signature. The
    # incident fingerprint is already in the store -> not new -> no alert.
    same = _write_sigs(tmp_path / "s2.json", [_sig(row_count=0)])
    second = main(["check", "--store", str(store), "--signatures", str(same)])
    assert second == 0


def test_input_error_exits_two(tmp_path):
    rc = main(["check", "--store", str(tmp_path / "b.json"),
               "--signatures", str(tmp_path / "missing.json")])
    assert rc == 2


# ---- sensitivity flags: per-deployment threshold tuning ---------------------------
def test_bad_volume_threshold_exits_two_before_any_io(tmp_path, capsys):
    """An invalid threshold must fail fast (exit 2) and never seed a store."""
    store = tmp_path / "baselines.json"
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs),
               "--volume-threshold", "0"])
    assert rc == 2
    assert "volume threshold must be > 0" in capsys.readouterr().err
    assert not store.exists()  # bailed before touching disk


def test_bad_null_threshold_exits_two(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs),
               "--null-threshold", "2"])
    assert rc == 2
    assert "null threshold must be in (0, 1]" in capsys.readouterr().err


def test_loose_volume_threshold_suppresses_incident(tmp_path):
    """The same drift that fires at the default band must go quiet under a looser one."""
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={ORDERS_URN: _sig(ORDERS_URN, row_count=10_000)}).save()
    drift = _write_sigs(tmp_path / "s.json", [_sig(ORDERS_URN, row_count=6_000)])  # -40%

    # Default ±30% band: -40% is a new incident -> exit 1.
    assert main(["check", "--store", str(store), "--signatures", str(drift),
                 "--no-update"]) == 1
    # ±50% band: -40% is within tolerance -> no drift -> exit 0.
    assert main(["check", "--store", str(store), "--signatures", str(drift),
                 "--no-update", "--volume-threshold", "0.5"]) == 0


def test_sensitivity_flags_registered_in_help():
    parser = build_parser()
    ns = parser.parse_args(
        ["check", "--signatures", "x", "--volume-threshold", "0.4",
         "--null-threshold", "0.1", "--no-serving-escalation"]
    )
    assert ns.volume_threshold == 0.4
    assert ns.null_threshold == 0.1
    assert ns.no_serving_escalation is True


# ---- --fail-on: CI severity gate on the exit code ---------------------------------
# Severity bands (see scorer._severity_from_ratio, default volume band 0.30):
#   collapse 1000 -> 0        = HIGH
#   1000 -> 500  (-50%, x1.67)= MEDIUM
#   1000 -> 650  (-35%, x1.17)= LOW
def _drift_run(tmp_path, cur_rows, extra_args, base_rows=1000, capsys=None):
    """Seed a baseline, present a drifted signature, return the exit code."""
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig(row_count=base_rows)}).save()
    sigs = _write_sigs(tmp_path / "s.json", [_sig(row_count=cur_rows)])
    return main(["check", "--store", str(store), "--signatures", str(sigs),
                 "--no-update", *extra_args])


def test_fail_on_high_passes_a_low_incident(tmp_path, capsys):
    # -35% is a real new incident (exit 1 by default) but only LOW severity.
    assert _drift_run(tmp_path, 650, []) == 1                       # control: default gate
    assert _drift_run(tmp_path, 650, ["--fail-on", "high"]) == 0    # gated below the floor
    out = capsys.readouterr().out
    assert "below --fail-on high" in out  # never a silent pass


def test_fail_on_high_still_fails_a_high_incident(tmp_path):
    assert _drift_run(tmp_path, 0, ["--fail-on", "high"]) == 1  # collapse = HIGH >= high


def test_fail_on_medium_gates_low_but_not_medium(tmp_path):
    assert _drift_run(tmp_path, 650, ["--fail-on", "medium"]) == 0  # LOW  < medium
    assert _drift_run(tmp_path, 500, ["--fail-on", "medium"]) == 1  # MEDIUM >= medium


def test_fail_on_low_matches_default_for_any_incident(tmp_path):
    # A floor of "low" is the loosest gate — it should behave like no --fail-on.
    assert _drift_run(tmp_path, 650, ["--fail-on", "low"]) == 1


def test_fail_on_never_fails_when_healthy(tmp_path, capsys):
    # No new incident -> exit 0 regardless of the floor, and no gate note printed.
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig()}).save()
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])  # unchanged
    assert main(["check", "--store", str(store), "--signatures", str(sigs),
                 "--no-update", "--fail-on", "low"]) == 0
    assert "below --fail-on" not in capsys.readouterr().out


def test_fail_on_registered_and_validated():
    parser = build_parser()
    assert parser.parse_args(["check", "--signatures", "x", "--fail-on", "high"]).fail_on == "high"
    with pytest.raises(SystemExit):  # not a valid severity
        parser.parse_args(["check", "--signatures", "x", "--fail-on", "critical"])


def test_gate_should_fail_pure_helper():
    """Direct unit of the gate, independent of argparse/I-O (fault-injection anchor)."""
    from ogle.cli import gate_should_fail
    from ogle.narrative import build_incident
    from ogle.pipeline import DriftReport
    from ogle.scorer import DriftFinding, DriftKind, Severity

    def _report(sev, is_new=True):
        f = DriftFinding(urn=CUSTOMERS_URN, kind=DriftKind.VOLUME, severity=sev, message="x")
        inc = build_incident([f])
        return DriftReport(findings=[f], incident=inc, narrative="",
                           is_new_incident=is_new, incident_count=1)

    med = _report(Severity.MEDIUM)
    assert gate_should_fail(med, None) is True              # default: any new incident
    assert gate_should_fail(med, "low") is True             # MEDIUM >= low
    assert gate_should_fail(med, "medium") is True          # MEDIUM >= medium
    assert gate_should_fail(med, "high") is False           # MEDIUM < high
    # A non-new (debounced) incident never fails, even at the loosest floor.
    assert gate_should_fail(_report(Severity.HIGH, is_new=False), "low") is False


# ---- persistence + --no-update ----------------------------------------------------
def test_baselines_persist_across_runs(tmp_path):
    store = tmp_path / "baselines.json"
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    main(["check", "--store", str(store), "--signatures", str(sigs)])
    reloaded = BaselineStore.load(store)
    assert reloaded.get_baseline(CUSTOMERS_URN) is not None


def test_no_update_does_not_write_store(tmp_path):
    store = tmp_path / "baselines.json"
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    main(["check", "--store", str(store), "--signatures", str(sigs), "--no-update"])
    assert not store.exists()  # read-only probe never persists


def test_no_update_still_alerts_on_new_incident(tmp_path):
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig(row_count=1000)}).save()
    sigs = _write_sigs(tmp_path / "s.json", [_sig(row_count=0)])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs), "--no-update"])
    assert rc == 1


# ---- serving escalation via --serving ---------------------------------------------
def test_cli_serving_flag_augments_file(tmp_path):
    """--serving on the command line unions with the file's serving_urns."""
    store = tmp_path / "baselines.json"
    BaselineStore(
        path=store,
        baselines={CUSTOMERS_URN: _sig(row_count=1000), ORDERS_URN: _sig(urn=ORDERS_URN, row_count=1000)},
    ).save()
    # orders collapses; mark it serving only via the CLI flag (not in the file).
    sigs = _write_sigs(tmp_path / "s.json", [_sig(urn=ORDERS_URN, row_count=0)])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs),
               "--serving", ORDERS_URN, "--json"])
    assert rc == 1


# ---- JSON output ------------------------------------------------------------------
def test_json_output_is_parseable(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig(row_count=1000)}).save()
    sigs = _write_sigs(tmp_path / "s.json", [_sig(row_count=0)])
    main(["check", "--store", str(store), "--signatures", str(sigs), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["should_alert"] is True
    assert payload["incident"] is not None


# ---- parser + no-command behavior -------------------------------------------------
def test_no_command_prints_help_exits_zero(capsys):
    rc = main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "ogle" in capsys.readouterr().out.lower()


def test_emit_survives_legacy_console_encoding():
    """Emoji in the narrative must not crash a cp1252 (Windows) console."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252", errors="strict", newline="")
    # A naive stream.write would raise UnicodeEncodeError on the check-mark.
    _emit("✅ drift detected \U0001F534", stream=stream)
    stream.flush()
    text = buf.getvalue().decode("cp1252")
    assert "drift detected" in text  # ran to completion; unencodable chars replaced


def test_parser_check_defaults():
    args = build_parser().parse_args(["check"])
    assert args.store == "ogle-baselines.json"
    assert args.gms == "http://localhost:8080"
    assert args.no_update is False
    assert args.json is False


# ---- `ogle demo` — one-command keyless judge repro ---------------------------------
def test_demo_seeds_then_alerts_exit_one(capsys):
    """`ogle demo` runs the bundled fixtures end-to-end: healthy seed -> HIGH alert."""
    rc = main(["demo"])
    out = capsys.readouterr().out
    assert rc == 1  # second pass fires the serving-path incident
    assert "Seed baselines" in out
    assert "HIGH drift" in out
    # Reproduces the captured alert's incident fingerprint, not a coincidental one.
    assert "fd6f829c77ff9fb4" in out


def test_demo_never_writes_to_cwd(tmp_path, monkeypatch):
    """The in-memory demo store must not leave a baseline file behind."""
    monkeypatch.chdir(tmp_path)
    main(["demo"])
    assert list(tmp_path.iterdir()) == []


def test_parser_registers_demo():
    args = build_parser().parse_args(["demo"])
    assert args.func.__name__ == "cmd_demo"


def test_demo_default_omits_llm_summary(capsys):
    """Plain `demo` stays keyless — no feature-#2 section unless --narrate is asked for."""
    main(["demo"])
    out = capsys.readouterr().out
    assert "LLM root-cause summary" not in out


def test_demo_narrate_adds_llm_summary(capsys, monkeypatch):
    """`demo --narrate` surfaces feature #2, using an injected narrator (no live Ollama)."""
    monkeypatch.setattr(
        "ogle.cli.build_narrator", lambda spec: (lambda prompt: "INJECTED-LLM-SUMMARY")
    )
    rc = main(["demo", "--narrate"])
    out = capsys.readouterr().out
    assert rc == 1  # the alert still governs the exit code
    assert "LLM root-cause summary" in out
    assert "INJECTED-LLM-SUMMARY" in out


def test_demo_narrate_falls_back_when_model_unreachable(capsys, monkeypatch):
    """An unreachable model must not break the demo — narrate falls back, exit code holds."""
    def _broken(_prompt):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("ogle.cli.build_narrator", lambda spec: _broken)
    rc = main(["demo", "--narrate"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "LLM root-cause summary" in out  # section renders via the deterministic fallback
    assert "HIGH drift" in out


def test_demo_narrate_bad_spec_exits_two(capsys, monkeypatch):
    """A malformed --narrate SPEC is a usage error (exit 2), matching `ogle check`."""
    def _reject(spec):
        raise ValueError("unknown narrator spec")

    monkeypatch.setattr("ogle.cli.build_narrator", _reject)
    rc = main(["demo", "--narrate", "bogus"])
    assert rc == 2


def test_parser_demo_narrate_defaults_to_ollama():
    args = build_parser().parse_args(["demo", "--narrate"])
    assert args.narrate == "ollama"


# ---- mute / unmute / muted (known false positives) --------------------------------
def test_mute_persists_and_reports(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(["mute", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 0
    assert "muted" in capsys.readouterr().out
    assert BaselineStore.load(store).is_muted(CUSTOMERS_URN) is True


def test_mute_is_idempotent(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    capsys.readouterr()
    rc = main(["mute", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 0
    assert "already muted" in capsys.readouterr().out


def test_unmute_reverses(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    capsys.readouterr()
    rc = main(["unmute", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 0
    assert "unmuted" in capsys.readouterr().out
    assert BaselineStore.load(store).is_muted(CUSTOMERS_URN) is False


def test_unmute_not_muted_reports(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(["unmute", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 0
    assert "not muted" in capsys.readouterr().out


def test_muted_lists_urns(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    main(["mute", ORDERS_URN, "--store", str(store)])
    capsys.readouterr()
    rc = main(["muted", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert CUSTOMERS_URN in out and ORDERS_URN in out
    assert "2 muted" in out


def test_muted_json_shape(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    capsys.readouterr()
    rc = main(["muted", "--store", str(store), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # A permanent mute carries a null expiry; snoozes carry an epoch `until`.
    assert payload == {"muted": [{"urn": CUSTOMERS_URN, "until": None}]}


def test_muted_empty_reports_none(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(["muted", "--store", str(store)])
    assert rc == 0
    assert "no muted datasets" in capsys.readouterr().out.lower()


def test_muted_urns_plain_selector(tmp_path, capsys):
    """`muted --urns` emits one URN per line (the pipe-side selector) — no prose header."""
    store = tmp_path / "baselines.json"
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    main(["mute", ORDERS_URN, "--store", str(store)])
    capsys.readouterr()
    rc = main(["muted", "--store", str(store), "--urns"])
    assert rc == 0
    out = capsys.readouterr().out
    # Exactly the two URNs, one per line, sorted — nothing else (no "N muted" header).
    assert out.splitlines() == sorted([CUSTOMERS_URN, ORDERS_URN])
    assert "muted" not in out


def test_muted_urns_overrides_json(tmp_path, capsys):
    """--urns wins over --json (it IS the scriptable form), mirroring `baselines --urns`."""
    store = tmp_path / "baselines.json"
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    capsys.readouterr()
    rc = main(["muted", "--store", str(store), "--urns", "--json"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == CUSTOMERS_URN  # plain line, not JSON


def test_muted_urns_silent_on_empty(tmp_path, capsys):
    """Empty set → SILENT (clean pipe), not the human 'no muted datasets' line."""
    store = tmp_path / "baselines.json"
    rc = main(["muted", "--store", str(store), "--urns"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_check_on_muted_dataset_stays_quiet_exit_zero(tmp_path, capsys):
    """End-to-end: a muted dataset that collapses must NOT page (exit 0, 'silenced' tail)."""
    store = tmp_path / "baselines.json"
    BaselineStore(path=store, baselines={CUSTOMERS_URN: _sig(row_count=1000)}).save()
    main(["mute", CUSTOMERS_URN, "--store", str(store)])
    capsys.readouterr()
    drift = _write_sigs(tmp_path / "s.json", [_sig(row_count=0)])
    rc = main(["check", "--store", str(store), "--signatures", str(drift)])
    assert rc == 0  # muted -> no alert despite a real collapse
    assert "silenced 1 muted dataset" in capsys.readouterr().out


# ---- timed mutes / snooze (CLI) ---------------------------------------------------
def test_mute_for_days_reports_snooze_and_persists_future_expiry(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(["mute", CUSTOMERS_URN, "--for", "7", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "snoozed" in out and "until" in out
    exp = BaselineStore.load(store).mute_expiry(CUSTOMERS_URN)
    assert exp is not None and exp > time.time()  # roughly a week out


def test_mute_for_hours_snoozes(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(["mute", CUSTOMERS_URN, "--for-hours", "2", "--store", str(store)])
    assert rc == 0
    exp = BaselineStore.load(store).mute_expiry(CUSTOMERS_URN)
    assert exp is not None and exp > time.time()


def test_mute_for_and_for_hours_together_is_usage_error(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(
        ["mute", CUSTOMERS_URN, "--for", "1", "--for-hours", "1", "--store", str(store)]
    )
    assert rc == 2
    assert "not both" in capsys.readouterr().err


def test_mute_nonpositive_snooze_is_usage_error(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    rc = main(["mute", CUSTOMERS_URN, "--for", "0", "--store", str(store)])
    assert rc == 2
    assert "positive" in capsys.readouterr().err
    # nothing was written
    assert not store.exists() or BaselineStore.load(store).muted() == []


def test_muted_json_reports_snooze_expiry(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    main(["mute", ORDERS_URN, "--for", "5", "--store", str(store)])
    capsys.readouterr()
    rc = main(["muted", "--store", str(store), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["muted"]) == 1
    entry = payload["muted"][0]
    assert entry["urn"] == ORDERS_URN
    assert entry["until"] is not None and entry["until"] > time.time()


def test_check_purges_lapsed_snooze_and_pages(tmp_path, capsys):
    """A snooze in the past must not silence a real collapse — check pages AND self-cleans it."""
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path, baselines={CUSTOMERS_URN: _sig(row_count=1000)})
    s.mute(CUSTOMERS_URN, until=time.time() - 3600)  # expired an hour ago
    s.save()
    drift = _write_sigs(tmp_path / "s.json", [_sig(row_count=0)])
    rc = main(["check", "--store", str(store_path), "--signatures", str(drift)])
    assert rc == 1  # lapsed snooze -> the collapse pages
    # And the dead snooze is gone from the store afterward.
    assert BaselineStore.load(store_path).muted() == []


# ---- `ogle baselines`: the watch-list view (the other half of the store) ----------
def _seed_baselines(store_path):
    """A store with two tracked datasets (customers: 2 fields/1000 rows, orders: 1/0)."""
    s = BaselineStore(
        path=store_path,
        baselines={
            CUSTOMERS_URN: _sig(row_count=1000),  # 2 fields (id, email)
            ORDERS_URN: _sig(
                urn=ORDERS_URN, schema_fields=[("oid", "int")], row_count=0
            ),
        },
    )
    s.save()
    return s


def test_baselines_lists_tracked_datasets(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["baselines", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 tracked dataset(s)" in out
    assert CUSTOMERS_URN in out and ORDERS_URN in out
    assert "2 field(s)" in out and "1000 rows" in out


def test_baselines_empty_reports_none(tmp_path, capsys):
    store = tmp_path / "baselines.json"  # never created
    rc = main(["baselines", "--store", str(store)])
    assert rc == 0
    assert "no baselines yet" in capsys.readouterr().out.lower()


def test_baselines_json_shape(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    seeded = _seed_baselines(store)
    rc = main(["baselines", "--store", str(store), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    by_urn = {e["urn"]: e for e in payload["baselines"]}
    assert by_urn[CUSTOMERS_URN]["fields"] == 2
    assert by_urn[CUSTOMERS_URN]["row_count"] == 1000
    # schema_hash is the real denormalized hash, not a placeholder.
    assert by_urn[CUSTOMERS_URN]["schema_hash"] == seeded.get_baseline(
        CUSTOMERS_URN
    ).schema_hash
    assert by_urn[ORDERS_URN]["row_count"] == 0


def test_baselines_grep_filters_by_urn(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["baselines", "--store", str(store), "--grep", "orders"])
    assert rc == 0
    out = capsys.readouterr().out
    assert ORDERS_URN in out and CUSTOMERS_URN not in out


def test_baselines_grep_is_case_insensitive(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["baselines", "--store", str(store), "--grep", "ORDERS", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [e["urn"] for e in payload["baselines"]] == [ORDERS_URN]


def test_baselines_grep_no_match_distinguishes_from_empty(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["baselines", "--store", str(store), "--grep", "nonesuch"])
    assert rc == 0
    assert "no baselines match the filter" in capsys.readouterr().out.lower()


def test_baselines_grep_all_whitespace_matches_nothing(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    # A fat-fingered `--grep "   "` is a slip, not a wildcard — matches nothing.
    rc = main(["baselines", "--store", str(store), "--grep", "   ", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["baselines"] == []


def test_baselines_urns_prints_plain_list_for_piping(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["baselines", "--store", str(store), "--urns"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # Plain URNs only (sorted), no markdown/prose — a clean selector stream.
    assert lines == sorted([CUSTOMERS_URN, ORDERS_URN])


def test_baselines_urns_honors_grep_and_overrides_json(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["baselines", "--store", str(store), "--grep", "orders", "--urns", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() == ORDERS_URN  # --urns wins over --json; --grep applied
    assert "{" not in out  # not JSON


def test_baselines_urns_empty_is_silent(tmp_path, capsys):
    store = tmp_path / "baselines.json"  # never created
    rc = main(["baselines", "--store", str(store), "--urns"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""  # a pipe gets a clean empty stream


# ---- `ogle baselines --sort`: order the watch-list by blast radius --------------
# Three datasets whose alphabetical, field-count, and row-count orders all differ, so a
# sort that no-ops (or sorts on the wrong axis) can't accidentally pass.
_A_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.a_alpha,PROD)"    # 1 field, 5000 rows
_M_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.m_mid,PROD)"      # 3 fields, 10 rows
_Z_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.z_zed,PROD)"      # 2 fields, 999 rows


def _seed_sortable(store_path):
    """A store whose URN/fields/rows orderings are all distinct (see per-axis expectations)."""
    BaselineStore(
        path=store_path,
        baselines={
            _A_URN: _sig(urn=_A_URN, schema_fields=[("id", "int")], row_count=5000),
            _M_URN: _sig(
                urn=_M_URN,
                schema_fields=[("a", "int"), ("b", "int"), ("c", "int")],
                row_count=10,
            ),
            _Z_URN: _sig(
                urn=_Z_URN, schema_fields=[("x", "int"), ("y", "int")], row_count=999
            ),
        },
    ).save()


def test_baselines_sort_default_is_alphabetical(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_sortable(store)
    rc = main(["baselines", "--store", str(store), "--urns"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [_A_URN, _M_URN, _Z_URN]  # a < m < z


def test_baselines_sort_fields_widest_first(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_sortable(store)
    rc = main(["baselines", "--store", str(store), "--sort", "fields", "--urns"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [_M_URN, _Z_URN, _A_URN]  # 3 > 2 > 1 fields


def test_baselines_sort_rows_highest_first(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_sortable(store)
    rc = main(["baselines", "--store", str(store), "--sort", "rows", "--urns"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [_A_URN, _Z_URN, _M_URN]  # 5000 > 999 > 10 rows


def test_baselines_sort_composes_with_grep(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_sortable(store)
    # --grep keeps a_alpha + m_mid (both contain "_"), --sort fields reorders the survivors.
    rc = main(
        ["baselines", "--store", str(store), "--grep", "b2fd91", "--sort", "rows", "--urns"]
    )
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [_A_URN, _Z_URN, _M_URN]  # all match; ordered by rows desc


def test_baselines_sort_tiebreak_is_urn_ascending(tmp_path, capsys):
    """Equal metric → URN ascending, so the order is deterministic run to run."""
    store = tmp_path / "baselines.json"
    # Two datasets, identical field count (1), different URNs.
    BaselineStore(
        path=store,
        baselines={
            _Z_URN: _sig(urn=_Z_URN, schema_fields=[("x", "int")], row_count=1),
            _A_URN: _sig(urn=_A_URN, schema_fields=[("id", "int")], row_count=1),
        },
    ).save()
    rc = main(["baselines", "--store", str(store), "--sort", "fields", "--urns"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [_A_URN, _Z_URN]  # tie on fields -> URN ascending


def test_baselines_sort_rows_sinks_unknown_rowcount_last(tmp_path, capsys):
    """A baseline with no row_count (None) sorts LAST under --sort rows, not first."""
    store = tmp_path / "baselines.json"
    BaselineStore(
        path=store,
        baselines={
            _A_URN: _sig(urn=_A_URN, schema_fields=[("id", "int")], row_count=None),
            _Z_URN: _sig(urn=_Z_URN, schema_fields=[("x", "int")], row_count=5),
        },
    ).save()
    rc = main(["baselines", "--store", str(store), "--sort", "rows", "--urns"])
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == [_Z_URN, _A_URN]  # 5 rows first, unknown (None) last


# ---- `ogle incidents`: the cross-run incident-memory view -------------------------
def test_incidents_empty_store_reports_none(tmp_path, capsys):
    store = tmp_path / "baselines.json"  # never created
    rc = main(["incidents", "--store", str(store)])
    assert rc == 0
    assert "no incidents remembered" in capsys.readouterr().out


def test_incidents_lists_what_check_recorded(tmp_path, capsys):
    # Drive a real drift through `ogle check`, then confirm `ogle incidents` surfaces it
    # with severity, recurrence and serving provenance — not just an opaque fingerprint.
    store_path = tmp_path / "baselines.json"
    seed = _write_sigs(tmp_path / "seed.json", [_sig(row_count=1000)], serving=[CUSTOMERS_URN])
    assert main(["check", "--store", str(store_path), "--signatures", str(seed)]) == 0
    drift = _write_sigs(tmp_path / "drift.json", [_sig(row_count=0)], serving=[CUSTOMERS_URN])
    assert main(["check", "--store", str(store_path), "--signatures", str(drift)]) == 1
    capsys.readouterr()  # drop check output

    rc = main(["incidents", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 remembered incident" in out
    assert "high" in out
    assert "seen 1×" in out
    assert "serving" in out
    # the store's fingerprint is shown for later cross-reference
    fp = BaselineStore.load(store_path).incidents()[0]["fingerprint"]
    assert fp in out


def test_incidents_json_shape(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp1", severity="high", title="HIGH drift", datasets=2, serving=True)
    s.save()
    rc = main(["incidents", "--store", str(store_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["incidents"]) == 1
    entry = payload["incidents"][0]
    assert entry["fingerprint"] == "fp1"
    assert entry["severity"] == "high"
    assert entry["count"] == 1
    assert entry["serving"] is True


def test_incidents_sorted_worst_severity_first(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("low_fp", severity="low", title="LOW drift", datasets=1)
    s.record_incident("high_fp", severity="high", title="HIGH drift", datasets=3)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--json"]) == 0
    order = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert order == ["high_fp", "low_fp"]  # HIGH ranked above LOW


def test_incidents_recurrence_count_surfaces(tmp_path, capsys):
    # A recurring incident's climbing count is the "still happening" signal an operator
    # wants — surface it verbatim.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("x", severity="high", title="HIGH drift", datasets=1)
    s.record_incident("x", severity="high", title="HIGH drift", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path)]) == 0
    assert "seen 2×" in capsys.readouterr().out


def test_incidents_legacy_bare_record_renders(tmp_path, capsys):
    # An incident memory from an older Ogle (count only, no severity/title) must render
    # with a safe fallback rather than crash on the missing provenance.
    store_path = tmp_path / "baselines.json"
    store_path.write_text(
        json.dumps({"version": 1, "seen_incidents": {"old": {"count": 4}}}),
        encoding="utf-8",
    )
    rc = main(["incidents", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "unknown" in out and "seen 4×" in out and "old" in out


def test_incidents_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--json"])
    assert ns.func.__name__ == "cmd_incidents"
    assert ns.json is True


# ---- `ogle incidents` triage filters: --min-severity / --serving-only --------------
def _seed_mixed_incidents(store_path):
    """A store with one incident at each severity plus a serving/non-serving mix."""
    s = BaselineStore(path=store_path)
    s.record_incident("high_serv", severity="high", title="HIGH", datasets=3, serving=True)
    s.record_incident("med_only", severity="medium", title="MED", datasets=2, serving=False)
    s.record_incident("low_serv", severity="low", title="LOW", datasets=1, serving=True)
    s.save()
    return s


def test_incidents_min_severity_filters_below_floor(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--min-severity", "high", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["high_serv"]  # medium + low dropped by the floor


def test_incidents_min_severity_medium_keeps_medium_and_up(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--min-severity", "medium", "--json"]) == 0
    fps = {e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]}
    assert fps == {"high_serv", "med_only"}  # low dropped, medium+high kept


def test_incidents_serving_only_filters_non_serving(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--serving-only", "--json"]) == 0
    fps = {e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]}
    assert fps == {"high_serv", "low_serv"}  # the medium non-serving one is hidden


def test_incidents_filters_compose_and(tmp_path, capsys):
    # --min-severity AND --serving-only together: only high-and-serving survives.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    rc = main(["incidents", "--store", str(store_path),
               "--min-severity", "high", "--serving-only", "--json"])
    assert rc == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["high_serv"]


def test_incidents_min_severity_drops_unknown_legacy(tmp_path, capsys):
    # A legacy bare-count record (no severity) must NOT survive a severity floor —
    # asking for a floor is asking to hide the un-triageable.
    store_path = tmp_path / "baselines.json"
    store_path.write_text(
        json.dumps({"version": 1, "seen_incidents": {"old": {"count": 4}}}),
        encoding="utf-8",
    )
    assert main(["incidents", "--store", str(store_path), "--min-severity", "low", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["incidents"] == []


def test_incidents_no_filter_shows_everything(tmp_path, capsys):
    # Sanity: without a filter all three remembered incidents are listed (no regression).
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["incidents"]) == 3


def test_incidents_filter_empty_result_message(tmp_path, capsys):
    # A floor above every remembered incident yields the "no match" message + count,
    # NOT the "none remembered" message (memory is non-empty; the filter hid it).
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("l1", severity="low", title="LOW", datasets=1)
    s.record_incident("l2", severity="low", title="LOW", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--min-severity", "high"]) == 0
    out = capsys.readouterr().out
    assert "no incidents match the filter" in out and "2 remembered" in out
    assert "no incidents remembered yet" not in out


def _seed_recurring_incidents(store_path):
    """A store with the same three incidents seen a different number of times each."""
    s = BaselineStore(path=store_path)
    for _ in range(5):
        s.record_incident("chronic", severity="high", title="CHRONIC", datasets=2, serving=True)
    for _ in range(2):
        s.record_incident("twice", severity="medium", title="TWICE", datasets=1)
    s.record_incident("once", severity="low", title="ONCE", datasets=1)
    s.save()
    return s


def test_incidents_min_count_filters_below_floor(tmp_path, capsys):
    # --min-count keeps only chronic/flapping incidents seen at least N times.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--min-count", "3", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["chronic"]  # twice (2×) and once (1×) drop below the floor of 3


def test_incidents_min_count_boundary_is_inclusive(tmp_path, capsys):
    # "at least N" — an incident seen exactly N times survives.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--min-count", "2", "--json"]) == 0
    fps = {e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]}
    assert fps == {"chronic", "twice"}  # 5× and 2× kept, the 1× drops


def test_incidents_min_count_composes_with_serving_only(tmp_path, capsys):
    # --min-count AND --serving-only: chronic is both flapping (5×) and serving; twice is
    # neither serving nor frequent-enough → only chronic survives the AND.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    rc = main(["incidents", "--store", str(store_path),
               "--min-count", "2", "--serving-only", "--json"])
    assert rc == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["chronic"]


def test_incidents_min_count_empty_result_message(tmp_path, capsys):
    # A count floor above every remembered incident yields the "no match" message + count,
    # NOT the "none remembered" message (memory is non-empty; the filter hid it).
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--min-count", "99"]) == 0
    out = capsys.readouterr().out
    assert "no incidents match the filter" in out and "3 remembered" in out
    assert "no incidents remembered yet" not in out


def test_incidents_filter_flags_registered_in_help():
    ns = build_parser().parse_args(
        ["incidents", "--min-severity", "high", "--serving-only", "--min-count", "4"]
    )
    assert ns.min_severity == "high"
    assert ns.serving_only is True
    assert ns.min_count == 4


# ---- `ogle incidents --grep`: find specific drift by title/fingerprint text ---------
def test_incidents_grep_matches_title(tmp_path, capsys):
    # --grep keeps only incidents whose title contains the needle (find drift by keyword).
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp_a", severity="high", title="customers row-count drop", datasets=1)
    s.record_incident("fp_b", severity="high", title="orders schema change", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--grep", "customers", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["fp_a"]  # "orders schema change" has no "customers"


def test_incidents_grep_is_case_insensitive(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp_a", severity="high", title="Customers row-count drop", datasets=1)
    s.save()
    # Upper-case needle still matches a mixed-case title.
    assert main(["incidents", "--store", str(store_path), "--grep", "CUSTOMERS", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["fp_a"]


def test_incidents_grep_matches_fingerprint_prefix(tmp_path, capsys):
    # The needle also matches the fingerprint, so a prefix works like `ogle resolve`.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abc123def", severity="high", title="drift one", datasets=1)
    s.record_incident("999feed", severity="high", title="drift two", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--grep", "abc1", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["abc123def"]  # matched on fingerprint, not title


def test_incidents_grep_composes_with_min_severity(tmp_path, capsys):
    # --grep AND --min-severity: both a text match and the severity floor must hold.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("hi", severity="high", title="serving drift", datasets=1)
    s.record_incident("lo", severity="low", title="serving drift", datasets=1)
    s.save()
    rc = main(["incidents", "--store", str(store_path),
               "--grep", "serving", "--min-severity", "high", "--json"])
    assert rc == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["hi"]  # both titled "serving drift"; the low one is below the floor


def test_incidents_grep_no_match_message(tmp_path, capsys):
    # A needle nothing contains yields the "no match" message + remembered count,
    # NOT the "none remembered" message (memory is non-empty; the filter hid it).
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--grep", "nonesuch"]) == 0
    out = capsys.readouterr().out
    assert "no incidents match the filter" in out and "3 remembered" in out
    assert "no incidents remembered yet" not in out


def test_incidents_grep_empty_needle_matches_nothing(tmp_path, capsys):
    # An all-whitespace needle is a user slip, not a wildcard — it matches nothing rather
    # than everything, so a fat-fingered `--grep ""` never masquerades as "all incidents".
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--grep", "   ", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["incidents"] == []


def test_incidents_grep_untitled_legacy_matches_fingerprint(tmp_path, capsys):
    # A legacy bare-count record (no title) still greps on its fingerprint.
    store_path = tmp_path / "baselines.json"
    store_path.write_text(
        json.dumps({"version": 1, "seen_incidents": {"orders_urn_xyz": {"count": 4}}}),
        encoding="utf-8",
    )
    assert main(["incidents", "--store", str(store_path), "--grep", "orders", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["orders_urn_xyz"]


def test_incidents_grep_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--grep", "customers"])
    assert ns.grep == "customers"
    # Default is None (no text filter) when the flag is omitted.
    assert build_parser().parse_args(["incidents"]).grep is None


# ---- `ogle incidents --summary`: aggregate rollup of the memory ---------------------
def test_incidents_summary_json_shape(tmp_path, capsys):
    # The rollup counts by severity, serving-path, recurrence, and total sightings.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)  # chronic(high,serv,5×) twice(med,2×) once(low,1×)
    assert main(["incidents", "--store", str(store_path), "--summary", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["total"] == 3
    assert summary["by_severity"] == {"high": 1, "medium": 1, "low": 1, "unknown": 0}
    assert summary["serving"] == 1          # only chronic touches a serving path
    assert summary["recurring"] == 2        # chronic(5×) + twice(2×); once(1×) is not recurring
    assert summary["total_sightings"] == 8  # 5 + 2 + 1


def test_incidents_summary_respects_filters(tmp_path, capsys):
    # --summary describes the FILTERED set, so it composes with the triage floors.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    rc = main(["incidents", "--store", str(store_path), "--summary", "--min-count", "3", "--json"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["total"] == 1  # only chronic (5×) clears the count floor of 3
    assert summary["by_severity"] == {"high": 1, "medium": 0, "low": 0, "unknown": 0}
    assert summary["recurring"] == 1
    assert summary["total_sightings"] == 5


def test_incidents_summary_text_render(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--summary"]) == 0
    out = capsys.readouterr().out
    assert "Incident memory summary — 3 remembered" in out
    assert "recurring (seen ≥2×): 2" in out
    assert "total sightings: 8" in out


def test_incidents_summary_unknown_severity_bucket(tmp_path, capsys):
    # A legacy bare-count record (no severity) lands in the `unknown` bucket, not dropped.
    store_path = tmp_path / "baselines.json"
    store_path.write_text(
        json.dumps({"version": 1, "seen_incidents": {"old": {"count": 3}}}),
        encoding="utf-8",
    )
    assert main(["incidents", "--store", str(store_path), "--summary", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["total"] == 1
    assert summary["by_severity"] == {"high": 0, "medium": 0, "low": 0, "unknown": 1}
    assert summary["total_sightings"] == 3


def test_incidents_summary_empty_store_message(tmp_path, capsys):
    # Summarizing an empty store reuses the "none remembered" message, not a zeroed rollup.
    store_path = tmp_path / "baselines.json"
    assert main(["incidents", "--store", str(store_path), "--summary"]) == 0
    assert "no incidents remembered yet" in capsys.readouterr().out


def test_incidents_summary_filtered_empty_message(tmp_path, capsys):
    # A filter that hides everything gets the "no match" message, not the empty-store one.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--summary", "--min-count", "99"]) == 0
    out = capsys.readouterr().out
    assert "no incidents match the filter" in out and "3 remembered" in out


# ---- `ogle incidents --limit N`: triage cap to the top-N worst -----------------------
def test_incidents_limit_caps_to_top_n(tmp_path, capsys):
    # --limit keeps only the top N after the worst-first sort: high before medium before low.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)  # high_serv, med_only, low_serv
    assert main(["incidents", "--store", str(store_path), "--limit", "2", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["high_serv", "med_only"]  # low_serv is past the cap


def test_incidents_limit_larger_than_set_shows_all(tmp_path, capsys):
    # A cap above the remembered count is a no-op — every incident still lists.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--limit", "10", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["incidents"]) == 3


def test_incidents_limit_composes_with_filters(tmp_path, capsys):
    # --limit applies AFTER the triage filters: serving-only keeps just chronic (the only
    # serving incident), then the cap of 1 keeps it — order of filter-then-cap is what's pinned.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)  # chronic(high,serv,5×) twice(med,2×) once(low,1×)
    rc = main(["incidents", "--store", str(store_path),
               "--serving-only", "--limit", "1", "--json"])
    assert rc == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["chronic"]  # only serving incident, and the cap keeps it


def test_incidents_limit_text_header_says_top_n_of_m(tmp_path, capsys):
    # The list header must announce the cap so a truncated view never reads as the whole set.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--limit", "1"]) == 0
    assert "Top 1 of 3 remembered incident(s)" in capsys.readouterr().out


def test_incidents_limit_at_full_count_keeps_plain_header(tmp_path, capsys):
    # When the cap doesn't actually hide anything, keep the plain header (no misleading "Top").
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "3 remembered incident(s)" in out and "Top" not in out


def test_incidents_limit_ignored_by_summary(tmp_path, capsys):
    # --summary describes the WHOLE filtered set; --limit must not shrink its totals.
    store_path = tmp_path / "baselines.json"
    _seed_recurring_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--summary", "--limit", "1", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["total"] == 3  # all three, not capped to 1
    assert summary["total_sightings"] == 8


def test_incidents_limit_zero_is_rejected(tmp_path, capsys):
    # A non-positive cap is a usage error, not a silent empty list that reads as "none tracked".
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--limit", "0"]) == 2
    assert "--limit must be a positive integer" in capsys.readouterr().out


def test_incidents_limit_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--limit", "5"])
    assert ns.limit == 5


# ---- `ogle incidents --fail-on` : drift-memory health gate ---------------------------
# Unlike `check --fail-on` (gates on NEW drift this run), this gates on whether remembered
# memory still holds open drift at/above a floor — a nightly job stays red until it's fixed.

def test_incidents_fail_on_high_trips_on_remembered_high(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)  # has a HIGH
    assert main(["incidents", "--store", str(store_path), "--fail-on", "high"]) == 1
    assert "exit 1" in capsys.readouterr().out


def test_incidents_fail_on_high_passes_when_only_below_floor(tmp_path, capsys):
    # A store whose worst remembered drift is MEDIUM must NOT trip a HIGH gate.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("med", severity="medium", title="MED", datasets=1)
    s.record_incident("low", severity="low", title="LOW", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--fail-on", "high"]) == 0
    assert "exit 1" not in capsys.readouterr().out


def test_incidents_fail_on_boundary_is_inclusive(tmp_path, capsys):
    # A MEDIUM incident meets a `--fail-on medium` floor (>=, not >).
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("med", severity="medium", title="MED", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--fail-on", "medium"]) == 1


def test_incidents_fail_on_empty_store_passes(tmp_path, capsys):
    # No memory => nothing to gate on => exit 0 even at the lowest floor.
    store_path = tmp_path / "baselines.json"
    BaselineStore(path=store_path).save()
    assert main(["incidents", "--store", str(store_path), "--fail-on", "low"]) == 0


def test_incidents_fail_on_default_never_gates(tmp_path, capsys):
    # Without --fail-on the command is read-only: always exit 0 even with a HIGH remembered.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path)]) == 0


def test_incidents_fail_on_composes_with_serving_only(tmp_path, capsys):
    # The gate sees the FILTERED set: a non-serving HIGH must not trip `--serving-only`.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("high_noserv", severity="high", title="HIGH", datasets=1, serving=False)
    s.record_incident("low_serv", severity="low", title="LOW", datasets=1, serving=True)
    s.save()
    # Serving path holds only a LOW -> HIGH gate passes despite the (hidden) non-serving HIGH.
    assert main(
        ["incidents", "--store", str(store_path), "--serving-only", "--fail-on", "high"]
    ) == 0


def test_incidents_fail_on_independent_of_limit(tmp_path, capsys):
    # --limit is a DISPLAY cap; it must never hide a failing incident from the gate.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    # HIGH sorts first; a lower recurrence still keeps it worst-first. Add filler so
    # --limit 1 shows only the HIGH — then a MEDIUM gate must still see... the whole set.
    s.record_incident("high", severity="high", title="HIGH", datasets=1)
    s.record_incident("low", severity="low", title="LOW", datasets=1)
    s.save()
    # Show only the top 1 (HIGH), but gate on medium: HIGH alone already trips it, and the
    # verdict is computed on the full set regardless of the cap.
    assert main(
        ["incidents", "--store", str(store_path), "--limit", "1", "--fail-on", "medium"]
    ) == 1


def test_incidents_fail_on_gates_json_without_corrupting_payload(tmp_path, capsys):
    # --json returns the failing exit code but emits NO prose note into the payload.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(
        ["incidents", "--store", str(store_path), "--fail-on", "high", "--json"]
    ) == 1
    out = capsys.readouterr().out
    json.loads(out)  # still valid JSON
    assert "exit 1" not in out


def test_incidents_fail_on_gates_summary(tmp_path, capsys):
    # The rollup view honours the gate too (whole filtered set, not the --limit cap).
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(
        ["incidents", "--store", str(store_path), "--summary", "--fail-on", "high"]
    ) == 1
    assert "exit 1" in capsys.readouterr().out


def test_incidents_fail_on_unknown_severity_never_trips(tmp_path, capsys):
    # A legacy/unknown-severity record ranks -1 and must not trip even the lowest floor.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("legacy")  # no severity
    s.save()
    assert main(["incidents", "--store", str(store_path), "--fail-on", "low"]) == 0


def test_incidents_fail_on_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--fail-on", "high"])
    assert ns.fail_on == "high"


# ---- `ogle incidents --sort` : pick the triage ordering axis --------------------------
# Default stays worst-severity-first; --sort count/datasets re-orders the SAME set (and
# redefines what --limit calls the "top N").

def _seed_sortable_incidents(store_path):
    # Deliberately anti-correlated axes so each --sort produces a DIFFERENT order:
    #   low_recur   — low severity, seen 5×, 1 dataset   (wins on count)
    #   high_narrow — high severity, seen 1×, 2 datasets (wins on severity)
    #   med_broad   — medium severity, seen 2×, 9 datasets (wins on datasets)
    s = BaselineStore(path=store_path)
    for _ in range(5):
        s.record_incident("low_recur", severity="low", title="LOW", datasets=1)
    s.record_incident("high_narrow", severity="high", title="HIGH", datasets=2)
    for _ in range(2):
        s.record_incident("med_broad", severity="medium", title="MED", datasets=9)
    s.save()


def _incident_order(capsys):
    return [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]


def test_incidents_sort_default_is_severity(tmp_path, capsys):
    # No --sort == worst-severity-first (unchanged behavior).
    store_path = tmp_path / "baselines.json"
    _seed_sortable_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--json"]) == 0
    assert _incident_order(capsys) == ["high_narrow", "med_broad", "low_recur"]


def test_incidents_sort_count_orders_most_recurring_first(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_sortable_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--sort", "count", "--json"]) == 0
    # 5× beats 2× beats 1× regardless of severity.
    assert _incident_order(capsys) == ["low_recur", "med_broad", "high_narrow"]


def test_incidents_sort_datasets_orders_broadest_first(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    _seed_sortable_incidents(store_path)
    assert (
        main(["incidents", "--store", str(store_path), "--sort", "datasets", "--json"]) == 0
    )
    # 9 > 2 > 1 datasets regardless of severity/recurrence.
    assert _incident_order(capsys) == ["med_broad", "high_narrow", "low_recur"]


def test_incidents_sort_redefines_limit_top_n(tmp_path, capsys):
    # --limit is "top N by the chosen sort": with --sort count the top 1 is the chronic
    # one, NOT the highest-severity one.
    store_path = tmp_path / "baselines.json"
    _seed_sortable_incidents(store_path)
    assert (
        main(
            ["incidents", "--store", str(store_path), "--sort", "count", "--limit", "1", "--json"]
        )
        == 0
    )
    assert _incident_order(capsys) == ["low_recur"]


def test_incidents_sort_composes_with_filters(tmp_path, capsys):
    # --sort orders only what the filters keep; a min-severity floor still drops the low one.
    store_path = tmp_path / "baselines.json"
    _seed_sortable_incidents(store_path)
    assert (
        main(
            [
                "incidents",
                "--store",
                str(store_path),
                "--min-severity",
                "medium",
                "--sort",
                "datasets",
                "--json",
            ]
        )
        == 0
    )
    # low_recur filtered out; remaining two ordered by datasets.
    assert _incident_order(capsys) == ["med_broad", "high_narrow"]


def test_incidents_sort_tiebreak_is_stable_by_fingerprint(tmp_path, capsys):
    # Two incidents identical on every sort axis fall back to fingerprint order (reverse=True
    # ⇒ descending) so the list never reshuffles between runs.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("aaa", severity="high", title="A", datasets=1)
    s.record_incident("bbb", severity="high", title="B", datasets=1)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--sort", "count", "--json"]) == 0
    assert _incident_order(capsys) == ["bbb", "aaa"]


def test_incidents_sort_rejects_unknown_axis(tmp_path):
    # argparse choices guard: an unknown axis is a usage error (exit 2), not a silent default.
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["incidents", "--sort", "bogus"])
    assert exc.value.code == 2


def test_incidents_sort_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--sort", "count"])
    assert ns.sort == "count"
    # default when omitted
    assert build_parser().parse_args(["incidents"]).sort == "severity"


# ---- `ogle incidents --fingerprints` : plain fingerprint list for piping into resolve --
# The read side becomes a selector for the write side: emit just the surviving fingerprints
# (one per line, honoring every filter + --sort + --limit) so a triager can
# `ogle incidents --serving-only --fingerprints | xargs ogle resolve` a whole batch at once.

def _fp_lines(capsys):
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]


def test_incidents_fingerprints_emits_plain_lines_in_sort_order(tmp_path, capsys):
    # Just the fingerprints, worst-severity-first, no markdown/marks/titles.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)  # high_serv, med_only, low_serv
    assert main(["incidents", "--store", str(store_path), "--fingerprints"]) == 0
    lines = _fp_lines(capsys)
    assert lines == ["high_serv", "med_only", "low_serv"]  # severity order, one per line
    # No decoration leaked in — these are raw tokens `ogle resolve` can consume.
    assert all("*" not in ln and "`" not in ln and " " not in ln for ln in lines)


def test_incidents_fingerprints_honors_filters(tmp_path, capsys):
    # Composes with the triage filters: only serving-path incidents' fingerprints print.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path), "--serving-only", "--fingerprints"]) == 0
    assert set(_fp_lines(capsys)) == {"high_serv", "low_serv"}  # med_only (non-serving) hidden


def test_incidents_fingerprints_honors_sort_and_limit(tmp_path, capsys):
    # --sort redefines the top N, so `--sort count --limit 1 --fingerprints` emits the single
    # most-recurring fingerprint — the selector picks exactly what the same sort/limit would list.
    store_path = tmp_path / "baselines.json"
    _seed_sortable_incidents(store_path)  # low_recur 5×, high_narrow 1×, med_broad 2×
    rc = main(["incidents", "--store", str(store_path),
               "--sort", "count", "--limit", "1", "--fingerprints"])
    assert rc == 0
    assert _fp_lines(capsys) == ["low_recur"]


def test_incidents_fingerprints_empty_set_is_silent(tmp_path, capsys):
    # A filter that hides everything prints NOTHING (no prose "no incidents" line) so a pipe
    # into `xargs ogle resolve` gets a clean empty stream, not a bogus token.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path),
                 "--grep", "nonesuch", "--fingerprints"]) == 0
    assert _fp_lines(capsys) == []


def test_incidents_fingerprints_empty_store_is_silent(tmp_path, capsys):
    # Same for a genuinely empty store — silent, exit 0, nothing to resolve.
    store_path = tmp_path / "baselines.json"
    assert main(["incidents", "--store", str(store_path), "--fingerprints"]) == 0
    assert _fp_lines(capsys) == []


def test_incidents_fingerprints_overrides_summary_and_json(tmp_path, capsys):
    # --fingerprints is the dominant machine mode: passing --summary/--json too still yields
    # the plain line list, never a JSON object or the rollup prose.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    assert main(["incidents", "--store", str(store_path),
                 "--fingerprints", "--summary", "--json"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["high_serv", "med_only", "low_serv"]
    assert "{" not in out and "summary" not in out  # neither JSON nor the rollup leaked


def test_incidents_fingerprints_composes_with_fail_on(tmp_path, capsys):
    # Still returns the health-gate code: --fail-on high trips (exit 1) while the fingerprints
    # print, so one command can both list the batch AND signal a failing gate.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)  # contains a high incident
    rc = main(["incidents", "--store", str(store_path),
               "--fingerprints", "--fail-on", "high"])
    assert rc == 1
    assert "high_serv" in _fp_lines(capsys)


def test_incidents_fingerprints_feed_resolve_end_to_end(tmp_path):
    # The whole point: capture the fingerprints, hand them to `ogle resolve`, and the
    # matching incidents leave cross-run memory — a scripted batch triage.
    store_path = tmp_path / "baselines.json"
    _seed_mixed_incidents(store_path)
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["incidents", "--store", str(store_path),
                     "--serving-only", "--fingerprints"]) == 0
    fps = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert set(fps) == {"high_serv", "low_serv"}
    assert main(["resolve", "--store", str(store_path), *fps]) == 0
    # Only the non-serving incident remains remembered.
    remaining = set(BaselineStore.load(store_path).seen_incidents)
    assert remaining == {"med_only"}


def test_incidents_fingerprints_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--fingerprints"])
    assert ns.fingerprints is True
    # Default is off (the human list view) when the flag is omitted.
    assert build_parser().parse_args(["incidents"]).fingerprints is False


# ---- `ogle resolve` ------------------------------------------------------------------
# Feature #3 memory operator control: once a drift is fixed in prod, the operator drops it
# from cross-run memory so `ogle incidents` no longer lists it AND a recurrence pages fresh.

def test_resolve_drops_incident_from_memory_and_persists(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abcdef1234567890", severity="high", title="HIGH drift", datasets=1)
    s.record_incident("11112222", severity="low", title="LOW drift", datasets=1)
    s.save()

    rc = main(["resolve", "abcdef1234567890", "--store", str(store_path)])
    assert rc == 0
    assert "resolved" in capsys.readouterr().out

    reloaded = BaselineStore.load(store_path)
    assert not reloaded.has_seen("abcdef1234567890")  # gone from memory
    assert reloaded.has_seen("11112222")              # untouched


def test_resolve_short_prefix_matches_like_a_git_sha(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abcdef1234567890", severity="high", title="HIGH drift", datasets=1)
    s.record_incident("beefcafe00000000", severity="low", title="LOW drift", datasets=1)
    s.save()

    rc = main(["resolve", "abcd", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "abcdef1234567890" in out  # renders the full fingerprint it landed on
    assert not BaselineStore.load(store_path).has_seen("abcdef1234567890")


def test_resolve_ambiguous_prefix_refuses_and_lists_candidates(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abcdef1234567890", severity="high", title="HIGH", datasets=1)
    s.record_incident("abcdef9999999999", severity="low", title="LOW", datasets=1)
    s.save()

    rc = main(["resolve", "abcd", "--store", str(store_path)])
    assert rc == 2  # ambiguous prefix is a usage error, not a silent guess
    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "abcdef1234567890" in err and "abcdef9999999999" in err
    # nothing was resolved when we refused
    reloaded = BaselineStore.load(store_path)
    assert reloaded.has_seen("abcdef1234567890")
    assert reloaded.has_seen("abcdef9999999999")


def test_resolve_unknown_fingerprint_is_a_reportable_miss_not_an_error(tmp_path, capsys):
    # A batch replay may include already-forgotten fingerprints. Miss ≠ error: exit 0, print
    # a "not remembered" line so the operator sees what didn't land.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("kept_fp", severity="low", title="LOW", datasets=1)
    s.save()

    rc = main(["resolve", "not_a_real_fp", "--store", str(store_path)])
    assert rc == 0
    assert "not remembered" in capsys.readouterr().out
    assert BaselineStore.load(store_path).has_seen("kept_fp")


def test_resolve_batch_partial_success_persists_only_the_hits(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("real_fp_1", severity="high", title="HIGH", datasets=1)
    s.record_incident("real_fp_2", severity="low", title="LOW", datasets=1)
    s.save()

    rc = main(
        ["resolve", "real_fp_1", "missing_fp", "real_fp_2", "--store", str(store_path)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("resolved") == 2       # two hits reported
    assert "not remembered" in out          # miss reported
    reloaded = BaselineStore.load(store_path)
    assert not reloaded.has_seen("real_fp_1")
    assert not reloaded.has_seen("real_fp_2")


def test_resolve_clears_fingerprint_so_a_recurrence_can_page_fresh(tmp_path, capsys):
    # Contract: resolve drops the fingerprint from `has_seen()` — that's what lets the
    # existing new-incident code path page again if the same drift shape reappears.
    store_path = tmp_path / "baselines.json"
    seed = _write_sigs(tmp_path / "seed.json", [_sig(row_count=1000)], serving=[CUSTOMERS_URN])
    assert main(["check", "--store", str(store_path), "--signatures", str(seed)]) == 0
    drift = _write_sigs(tmp_path / "drift.json", [_sig(row_count=0)], serving=[CUSTOMERS_URN])
    assert main(["check", "--store", str(store_path), "--signatures", str(drift)]) == 1
    capsys.readouterr()

    fp = BaselineStore.load(store_path).incidents()[0]["fingerprint"]
    assert BaselineStore.load(store_path).has_seen(fp)  # debounce is armed
    assert main(["resolve", fp, "--store", str(store_path)]) == 0
    capsys.readouterr()

    # Debounce is cleared → the pipeline's new-incident branch would fire again on a
    # recurring fingerprint. That path is covered by `test_new_incident_exits_one`.
    assert not BaselineStore.load(store_path).has_seen(fp)


def test_resolve_empty_prefix_is_a_reportable_miss_not_a_mass_wipe(tmp_path, capsys):
    # Guard: an empty token prefix-matches every incident. Refuse to resolve on that — this
    # command is targeted, not a nuke, so an empty arg must be a no-op miss.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("keep_a", severity="high", title="HIGH", datasets=1)
    s.record_incident("keep_b", severity="low", title="LOW", datasets=1)
    s.save()

    rc = main(["resolve", "", "--store", str(store_path)])
    assert rc == 0
    reloaded = BaselineStore.load(store_path)
    assert reloaded.has_seen("keep_a") and reloaded.has_seen("keep_b")


def test_resolve_strips_trailing_whitespace_from_piped_tokens(tmp_path, capsys):
    # The `incidents --fingerprints | xargs ogle resolve` pipe carries a trailing CR on
    # Windows; resolve must trim it so the token still matches (a fingerprint never has
    # surrounding whitespace). Without the strip, "keep_a\r" would be a spurious miss.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("keep_a", severity="high", title="HIGH", datasets=1)
    s.save()
    assert main(["resolve", "keep_a\r", "--store", str(store_path)]) == 0
    assert "resolved" in capsys.readouterr().out
    assert not BaselineStore.load(store_path).has_seen("keep_a")


# ---- `ogle resolve --dry-run` : preview without mutating memory -----------------------


def test_resolve_dry_run_previews_without_dropping_or_saving(tmp_path, capsys):
    # Contract: --dry-run resolves the token to its fingerprint and reports it, but the
    # incident stays in memory and the store file is never rewritten.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abcdef1234567890", severity="high", title="HIGH drift", datasets=1)
    s.save()
    mtime_before = store_path.stat().st_mtime_ns

    rc = main(["resolve", "abcdef1234567890", "--dry-run", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would resolve" in out
    assert "abcdef1234567890" in out
    assert "✅ resolved" not in out  # never claims it actually dropped anything

    reloaded = BaselineStore.load(store_path)
    assert reloaded.has_seen("abcdef1234567890")          # still remembered
    assert store_path.stat().st_mtime_ns == mtime_before  # file untouched


def test_resolve_dry_run_resolves_prefix_to_full_fingerprint(tmp_path, capsys):
    # The preview must be EXACT: a short prefix reports the full fingerprint it would drop,
    # so an operator sees precisely what a real resolve would remove.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abcdef1234567890", severity="high", title="HIGH", datasets=1)
    s.save()

    rc = main(["resolve", "abcd", "--dry-run", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would resolve" in out
    assert "abcdef1234567890" in out                      # full fp, not the typed prefix
    assert BaselineStore.load(store_path).has_seen("abcdef1234567890")


def test_resolve_dry_run_still_reports_misses_and_refuses_ambiguity(tmp_path, capsys):
    # Misses stay reportable (not an error) and an ambiguous prefix is still a usage error
    # (exit 2) even in preview mode — dry-run changes only whether we WRITE, not the guards.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("abcd1111", severity="high", title="A", datasets=1)
    s.record_incident("abcd2222", severity="low", title="B", datasets=1)
    s.save()

    # A clean miss: reportable, exit 0.
    rc = main(["resolve", "no_such_fp", "--dry-run", "--store", str(store_path)])
    assert rc == 0
    assert "not remembered" in capsys.readouterr().out

    # An ambiguous prefix: refuse and exit 2, same as a real resolve.
    rc = main(["resolve", "abcd", "--dry-run", "--store", str(store_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ambiguous" in err
    # Nothing dropped by the refusal.
    reloaded = BaselineStore.load(store_path)
    assert reloaded.has_seen("abcd1111") and reloaded.has_seen("abcd2222")


def test_resolve_dry_run_batch_previews_every_hit_and_keeps_all(tmp_path, capsys):
    # A batch dry-run previews each hit and leaves the WHOLE set intact — the safety check
    # before `... --fingerprints | xargs ogle resolve --dry-run` then a real resolve.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("real_fp_1", severity="high", title="A", datasets=1)
    s.record_incident("real_fp_2", severity="low", title="B", datasets=1)
    s.save()

    rc = main(
        ["resolve", "real_fp_1", "real_fp_2", "--dry-run", "--store", str(store_path)]
    )
    assert rc == 0
    assert capsys.readouterr().out.count("would resolve") == 2

    reloaded = BaselineStore.load(store_path)
    assert reloaded.has_seen("real_fp_1") and reloaded.has_seen("real_fp_2")


# ---- `ogle resolve -` : read fingerprints from stdin (xargs-free pipe) ---------------


def test_resolve_reads_fingerprints_from_stdin_dash(tmp_path, capsys, monkeypatch):
    # The point of `-`: `ogle incidents --fingerprints | ogle resolve -` works with no
    # xargs (native on Windows). Two fingerprints piped in, both dropped and persisted.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("real_fp_1", severity="high", title="A", datasets=1)
    s.record_incident("real_fp_2", severity="low", title="B", datasets=1)
    s.save()

    monkeypatch.setattr("sys.stdin", io.StringIO("real_fp_1\nreal_fp_2\n"))
    rc = main(["resolve", "-", "--store", str(store_path)])
    assert rc == 0
    assert capsys.readouterr().out.count("resolved") == 2
    reloaded = BaselineStore.load(store_path)
    assert not reloaded.has_seen("real_fp_1") and not reloaded.has_seen("real_fp_2")


def test_resolve_stdin_dash_ignores_blank_lines_and_crs(tmp_path, capsys, monkeypatch):
    # `sys.stdin.read().split()` drops empties and trailing CRs, so a Windows pipe with
    # blank lines never produces a bogus empty token (which would be a reportable miss).
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("keep_a", severity="high", title="A", datasets=1)
    s.save()

    monkeypatch.setattr("sys.stdin", io.StringIO("\r\nkeep_a\r\n\r\n"))
    rc = main(["resolve", "-", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved" in out
    assert "not remembered" not in out  # no phantom empty token from the blank lines
    assert not BaselineStore.load(store_path).has_seen("keep_a")


def test_resolve_stdin_dash_mixes_with_positional_tokens(tmp_path, capsys, monkeypatch):
    # `-` splices piped tokens in at its position; literal tokens on the same line still
    # resolve. Order/placement is preserved so a mixed invocation composes cleanly.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("cli_fp", severity="high", title="A", datasets=1)
    s.record_incident("piped_fp", severity="low", title="B", datasets=1)
    s.save()

    monkeypatch.setattr("sys.stdin", io.StringIO("piped_fp\n"))
    rc = main(["resolve", "cli_fp", "-", "--store", str(store_path)])
    assert rc == 0
    assert capsys.readouterr().out.count("resolved") == 2
    reloaded = BaselineStore.load(store_path)
    assert not reloaded.has_seen("cli_fp") and not reloaded.has_seen("piped_fp")


def test_resolve_stdin_dash_empty_is_a_noop_not_a_wipe(tmp_path, capsys, monkeypatch):
    # Guard: an EMPTY stdin behind `-` must resolve nothing (not prefix-match everything).
    # `read().split()` yields no tokens, so the loop never runs and the store is untouched.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("keep_a", severity="high", title="A", datasets=1)
    s.record_incident("keep_b", severity="low", title="B", datasets=1)
    s.save()
    mtime_before = store_path.stat().st_mtime_ns

    monkeypatch.setattr("sys.stdin", io.StringIO("   \n\r\n"))
    rc = main(["resolve", "-", "--store", str(store_path)])
    assert rc == 0
    reloaded = BaselineStore.load(store_path)
    assert reloaded.has_seen("keep_a") and reloaded.has_seen("keep_b")
    assert store_path.stat().st_mtime_ns == mtime_before  # nothing written


def test_resolve_stdin_dash_honors_dry_run(tmp_path, capsys, monkeypatch):
    # The stdin path composes with --dry-run: preview the piped fingerprints, drop nothing.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("real_fp_1", severity="high", title="A", datasets=1)
    s.save()

    monkeypatch.setattr("sys.stdin", io.StringIO("real_fp_1\n"))
    rc = main(["resolve", "-", "--dry-run", "--store", str(store_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would resolve" in out and "✅ resolved" not in out
    assert BaselineStore.load(store_path).has_seen("real_fp_1")  # still remembered


def test_incidents_fingerprints_pipe_into_resolve_stdin_end_to_end(tmp_path, capsys, monkeypatch):
    # The whole xargs-free story: capture `--fingerprints` output, feed it to `resolve -`
    # via stdin, and every remembered incident is gone.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp_alpha", severity="high", title="A", datasets=1)
    s.record_incident("fp_beta", severity="low", title="B", datasets=1)
    s.save()

    assert main(["incidents", "--store", str(store_path), "--fingerprints"]) == 0
    fps_blob = capsys.readouterr().out  # newline-separated fingerprints, as a pipe delivers

    monkeypatch.setattr("sys.stdin", io.StringIO(fps_blob))
    assert main(["resolve", "-", "--store", str(store_path)]) == 0
    assert capsys.readouterr().out.count("resolved") == 2
    assert BaselineStore.load(store_path).incidents() == []


def test_resolve_registered_in_help():
    ns = build_parser().parse_args(["resolve", "abcd"])
    assert ns.func.__name__ == "cmd_resolve"
    assert ns.fingerprint == ["abcd"]
    assert ns.dry_run is False  # defaults off — resolve mutates unless asked to preview


# ---- `ogle forget` -------------------------------------------------------------------
# The write-side counterpart to `ogle baselines`: prune a decommissioned dataset from the
# watch-list (drops its baseline signature + any mute state) so `baselines`/`check` stay honest.

def test_forget_drops_baseline_from_watchlist_and_persists(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)  # customers + orders
    rc = main(["forget", ORDERS_URN, "--store", str(store)])
    assert rc == 0
    assert "forgot" in capsys.readouterr().out
    reloaded = BaselineStore.load(store)
    assert ORDERS_URN not in reloaded          # gone from the watch-list
    assert CUSTOMERS_URN in reloaded           # the other dataset is untouched


def test_forget_also_clears_mute_state_for_the_dataset(tmp_path, capsys):
    # A mute pointing at a gone dataset is dead weight — forget clears it alongside the baseline.
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)
    s.mute(ORDERS_URN)  # permanent mute
    s.save()
    assert ORDERS_URN in BaselineStore.load(store).muted()

    rc = main(["forget", ORDERS_URN, "--store", str(store)])
    assert rc == 0
    reloaded = BaselineStore.load(store)
    assert ORDERS_URN not in reloaded
    assert ORDERS_URN not in reloaded.muted()  # orphan mute cleared too


def test_forget_unknown_urn_is_a_reportable_miss_not_an_error(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    mtime_before = store.stat().st_mtime_ns
    rc = main(["forget", "urn:li:dataset:(nope)", "--store", str(store)])
    assert rc == 0  # a replayed/absent URN is not a failure
    assert "not watched" in capsys.readouterr().out
    assert store.stat().st_mtime_ns == mtime_before  # nothing to save → untouched


def test_forget_batch_partial_success_persists_only_the_hits(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    rc = main(["forget", ORDERS_URN, "urn:li:dataset:(gone)", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "forgot" in out and "not watched" in out
    reloaded = BaselineStore.load(store)
    assert ORDERS_URN not in reloaded and CUSTOMERS_URN in reloaded


def test_forget_dry_run_previews_without_dropping_or_saving(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    mtime_before = store.stat().st_mtime_ns
    rc = main(["forget", ORDERS_URN, "--dry-run", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would forget" in out and "✅ forgot" not in out
    assert ORDERS_URN in BaselineStore.load(store)          # still watched
    assert store.stat().st_mtime_ns == mtime_before          # store never written


def test_forget_empty_token_is_a_reportable_miss_not_a_mass_wipe(tmp_path, capsys):
    # Guard: a fat-fingered empty/all-whitespace token forgets NOTHING (never wipes the store).
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    mtime_before = store.stat().st_mtime_ns
    rc = main(["forget", "   ", "--store", str(store)])
    assert rc == 0
    assert "not watched" in capsys.readouterr().out
    assert len(BaselineStore.load(store)) == 2
    assert store.stat().st_mtime_ns == mtime_before


def test_forget_reads_urns_from_stdin_dash(tmp_path, capsys, monkeypatch):
    # `ogle baselines --urns | ogle forget -` works with no xargs (native on Windows).
    store = tmp_path / "baselines.json"
    _seed_baselines(store)
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{CUSTOMERS_URN}\n{ORDERS_URN}\n"))
    rc = main(["forget", "-", "--store", str(store)])
    assert rc == 0
    assert capsys.readouterr().out.count("forgot") == 2
    assert len(BaselineStore.load(store)) == 0  # both pruned


def test_baselines_urns_pipe_into_forget_stdin_end_to_end(tmp_path, capsys, monkeypatch):
    # The whole xargs-free story: `baselines --grep --urns` selects the decommissioned
    # datasets, `forget -` prunes exactly those, and the rest of the watch-list survives.
    store = tmp_path / "baselines.json"
    _seed_baselines(store)  # customers + orders
    assert main(["baselines", "--store", str(store), "--grep", "orders", "--urns"]) == 0
    urns_blob = capsys.readouterr().out  # newline-separated URNs, as a pipe delivers

    monkeypatch.setattr("sys.stdin", io.StringIO(urns_blob))
    assert main(["forget", "-", "--store", str(store)]) == 0
    assert capsys.readouterr().out.count("forgot") == 1
    reloaded = BaselineStore.load(store)
    assert ORDERS_URN not in reloaded and CUSTOMERS_URN in reloaded


def test_forget_registered_in_help():
    ns = build_parser().parse_args(["forget", ORDERS_URN])
    assert ns.func.__name__ == "cmd_forget"
    assert ns.urn == [ORDERS_URN]
    assert ns.dry_run is False  # defaults off — forget mutates unless asked to preview


# ---- status: whole-store rollup ---------------------------------------------------
def test_status_rollup_human(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)  # 2 datasets, 3 fields total, 1000 rows
    s.record_incident("high_serv", severity="high", title="H", datasets=2, serving=True)
    s.record_incident("high_serv", severity="high", title="H", datasets=2, serving=True)
    s.record_incident("low_only", severity="low", title="L", datasets=1)
    s.mute(CUSTOMERS_URN)
    s.save()
    rc = main(["status", "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "watching: 2 dataset(s)" in out
    assert "3 field(s)" in out and "1000 row(s)" in out
    assert "incidents remembered: 2" in out  # two distinct fingerprints
    assert "serving-path: 1" in out and "recurring: 1" in out  # high_serv seen 2x
    assert "muted: 1 active" in out


def test_status_empty_store_reports_empty(tmp_path, capsys):
    store = tmp_path / "baselines.json"  # never created
    rc = main(["status", "--store", str(store)])
    assert rc == 0
    assert "is empty" in capsys.readouterr().out.lower()


def test_status_json_shape(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)
    s.record_incident("fp1", severity="medium", title="M", datasets=1)
    s.save()
    rc = main(["status", "--store", str(store), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)["status"]
    assert payload["baselines"]["watching"] == 2
    assert payload["baselines"]["fields"] == 3
    assert payload["baselines"]["rows"] == 1000
    assert payload["incidents"]["total"] == 1
    assert payload["incidents"]["by_severity"]["medium"] == 1
    assert payload["muted"] == 0
    assert payload["store"] == str(store)


def test_status_json_counts_unknown_rows(tmp_path, capsys):
    store = tmp_path / "baselines.json"
    s = BaselineStore(
        path=store,
        baselines={CUSTOMERS_URN: _sig(row_count=None)},  # row_count unknown
    )
    s.save()
    rc = main(["status", "--store", str(store), "--json"])
    assert rc == 0
    b = json.loads(capsys.readouterr().out)["status"]["baselines"]
    assert b["watching"] == 1 and b["rows"] == 0 and b["unknown_rows"] == 1


def test_status_fail_on_trips_exit_1_at_or_above_floor(tmp_path, capsys):
    # A remembered incident at/above the --fail-on floor turns the rollup into a failing
    # health gate (exit 1) AND still prints the snapshot + a reason line.
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)
    s.record_incident("hi", severity="high", title="H", datasets=1, serving=True)
    s.save()
    rc = main(["status", "--store", str(store), "--fail-on", "high"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "incidents remembered: 1" in out  # snapshot still rendered
    assert "--fail-on high" in out and "exit 1" in out


def test_status_fail_on_below_floor_stays_exit_0(tmp_path, capsys):
    # Only drift AT/ABOVE the floor should fail; a lone low incident under a high floor is 0.
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)
    s.record_incident("lo", severity="low", title="L", datasets=1)
    s.save()
    rc = main(["status", "--store", str(store), "--fail-on", "high"])
    assert rc == 0
    assert "exit 1" not in capsys.readouterr().out


def test_status_fail_on_gates_json_without_printing_note(tmp_path, capsys):
    # --json returns the SAME exit code (so a scheduled JSON consumer gets the verdict) but
    # stays pure JSON — the human reason line must not leak into the payload.
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)
    s.record_incident("hi", severity="high", title="H", datasets=1)
    s.save()
    rc = main(["status", "--store", str(store), "--fail-on", "medium", "--json"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "exit 1" not in out  # no prose in the JSON path
    assert json.loads(out)["status"]["incidents"]["total"] == 1


def test_status_without_fail_on_stays_exit_0_with_high_drift(tmp_path, capsys):
    # Default (no --fail-on): even a HIGH remembered incident leaves status at exit 0 — the
    # snapshot must never page unless a floor is explicitly requested.
    store = tmp_path / "baselines.json"
    s = _seed_baselines(store)
    s.record_incident("hi", severity="high", title="H", datasets=1, serving=True)
    s.save()
    rc = main(["status", "--store", str(store)])
    assert rc == 0
    assert "exit 1" not in capsys.readouterr().out


def test_status_registered_in_help():
    ns = build_parser().parse_args(["status"])
    assert ns.func.__name__ == "cmd_status"
    assert ns.json is False
    assert ns.fail_on is None


# ---- ogle show: single-dataset drill-down ----------------------------------------
def _seed_show(store_path, **mute):
    """A store with customers (2 fields, one carrying a null fraction) + orders. Optional
    mute kwargs: permanent=True, or until=<epoch> for a snooze."""
    s = BaselineStore(
        path=store_path,
        baselines={
            CUSTOMERS_URN: _sig(
                row_count=1000, field_null_fractions={"email": 0.25}
            ),
            ORDERS_URN: _sig(urn=ORDERS_URN, schema_fields=[("oid", "int")], row_count=0),
        },
    )
    if mute.get("permanent"):
        s.mute(CUSTOMERS_URN)
    elif mute.get("until") is not None:
        s.mute(CUSTOMERS_URN, until=mute["until"])
    s.save()
    return s


def test_show_human_renders_signature_and_fields(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_show(store)
    rc = main(["show", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert CUSTOMERS_URN in out
    assert "2 field(s)" in out and "1000 row(s)" in out
    # The field list — no other command shows path:type — plus the quality signal.
    assert "`email` : string" in out and "`id` : int" in out
    assert "25% null" in out
    assert "muted: no" in out


def test_show_prints_full_schema_hash(tmp_path, capsys):
    store = tmp_path / "s.json"
    seeded = _seed_show(store)
    main(["show", CUSTOMERS_URN, "--store", str(store)])
    full = seeded.get_baseline(CUSTOMERS_URN).schema_hash
    assert len(full) == 64  # full sha256, not the 12-char baselines preview
    assert full in capsys.readouterr().out


def test_show_json_shape(tmp_path, capsys):
    store = tmp_path / "s.json"
    seeded = _seed_show(store)
    rc = main(["show", CUSTOMERS_URN, "--store", str(store), "--json"])
    assert rc == 0
    ds = json.loads(capsys.readouterr().out)["dataset"]
    assert ds["urn"] == CUSTOMERS_URN
    assert ds["field_count"] == 2
    assert ds["row_count"] == 1000
    assert ds["schema_hash"] == seeded.get_baseline(CUSTOMERS_URN).schema_hash
    assert ds["muted"] is False and ds["muted_until"] is None
    by_path = {f["path"]: f for f in ds["fields"]}
    assert by_path["email"]["native_type"] == "string"
    assert by_path["email"]["null_fraction"] == 0.25
    # A field with no recorded null fraction omits the key rather than guessing 0.
    assert "null_fraction" not in by_path["id"]


def test_show_null_fraction_omitted_when_absent_human(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_show(store)
    main(["show", CUSTOMERS_URN, "--store", str(store)])
    out = capsys.readouterr().out
    # `id` has no null fraction → its line carries no "% null" suffix.
    id_line = [ln for ln in out.splitlines() if "`id` :" in ln][0]
    assert "null" not in id_line


def test_show_not_watched_exits_one(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_show(store)
    rc = main(["show", "urn:li:dataset:(nope)", "--store", str(store)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "not watched" in out.lower()
    assert "2 dataset(s) tracked" in out  # miss vs empty store


def test_show_empty_store_exits_one_and_says_so(tmp_path, capsys):
    store = tmp_path / "s.json"  # never created
    rc = main(["show", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 1
    assert "store is empty" in capsys.readouterr().out.lower()


def test_show_reports_permanent_mute(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_show(store, permanent=True)
    rc = main(["show", CUSTOMERS_URN, "--store", str(store)])
    assert rc == 0
    assert "muted: permanent" in capsys.readouterr().out


def test_show_reports_snooze_with_expiry(tmp_path, capsys):
    store = tmp_path / "s.json"
    until = time.time() + 3600
    _seed_show(store, until=until)
    rc = main(["show", CUSTOMERS_URN, "--store", str(store), "--json"])
    assert rc == 0
    ds = json.loads(capsys.readouterr().out)["dataset"]
    assert ds["muted"] is True
    assert ds["muted_until"] == pytest.approx(until)


def test_show_unknown_row_count_reads_plainly(tmp_path, capsys):
    store = tmp_path / "s.json"
    BaselineStore(
        path=store,
        baselines={CUSTOMERS_URN: _sig(row_count=None)},
    ).save()
    main(["show", CUSTOMERS_URN, "--store", str(store)])
    assert "rows unknown" in capsys.readouterr().out


def test_show_fields_sorted_by_path_for_stable_output(tmp_path, capsys):
    store = tmp_path / "s.json"
    BaselineStore(
        path=store,
        baselines={
            CUSTOMERS_URN: _sig(
                schema_fields=[("zeta", "int"), ("alpha", "int"), ("mid", "int")],
                row_count=1,
            )
        },
    ).save()
    main(["show", CUSTOMERS_URN, "--store", str(store)])
    out = capsys.readouterr().out
    assert out.index("`alpha`") < out.index("`mid`") < out.index("`zeta`")


def test_show_registered_in_help():
    ns = build_parser().parse_args(["show", CUSTOMERS_URN])
    assert ns.func.__name__ == "cmd_show"
    assert ns.urn == CUSTOMERS_URN
    assert ns.json is False


def test_show_composes_with_baselines_urns_selector(tmp_path, capsys):
    """`ogle baselines --grep customers --urns` emits the exact URN `show` consumes."""
    store = tmp_path / "s.json"
    _seed_show(store)
    main(["baselines", "--store", str(store), "--grep", "customers", "--urns"])
    urn = capsys.readouterr().out.strip()
    assert urn == CUSTOMERS_URN
    rc = main(["show", urn, "--store", str(store)])
    assert rc == 0
    assert "2 field(s)" in capsys.readouterr().out


# ---- diff: read-only baseline vs candidate-signatures-file --------------------------
def _seed_diff(store_path):
    """One watched dataset: id:int, email:string (25% null), 1000 rows."""
    BaselineStore(
        path=store_path,
        baselines={
            CUSTOMERS_URN: _sig(
                schema_fields=[("id", "int"), ("email", "string")],
                row_count=1000,
                field_null_fractions={"email": 0.25},
            )
        },
    ).save()
    return store_path


def test_diff_identical_exits_zero(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1000,
              field_null_fractions={"email": 0.25})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 0
    assert "no drift" in capsys.readouterr().out.lower()


def test_diff_field_added_exits_one(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string"), ("phone", "string")],
              row_count=1000, field_null_fractions={"email": 0.25})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "➕ `phone` : string" in out
    assert "schema hash: `" in out and "→" in out  # hash flipped with the field set


def test_diff_field_removed_exits_one(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int")], row_count=1000)],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 1
    assert "➖ `email` : string" in capsys.readouterr().out


def test_diff_type_changed_exits_one(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "bigint"), ("email", "string")], row_count=1000,
              field_null_fractions={"email": 0.25})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 1
    assert "🔀 `id` : int → bigint" in capsys.readouterr().out


def test_diff_null_fraction_changed_exits_one(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1000,
              field_null_fractions={"email": 0.60})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "`email` : 25.0% → 60.0% null" in out
    # A pure null-fraction move leaves the schema hash unchanged (hash is path+type only).
    assert "schema hash: unchanged" in out


def test_diff_null_fraction_below_rounding_gate_not_reported(tmp_path, capsys):
    """A sub-0.1pp re-profiling jitter must not surface as drift."""
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1000,
              field_null_fractions={"email": 0.25 + 0.0004})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 0
    assert "no drift" in capsys.readouterr().out.lower()


def test_diff_null_fraction_appearing_is_reported(tmp_path, capsys):
    """A field gaining a null profile it never had (unknown→known) is a real change."""
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1000,
              field_null_fractions={"email": 0.25, "id": 0.10})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 1
    assert "`id` : unknown → 10.0% null" in capsys.readouterr().out


def test_diff_row_count_delta_shown(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1200,
              field_null_fractions={"email": 0.25})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 1
    assert "rows: 1000 → 1200 (+200)" in capsys.readouterr().out


def test_diff_not_watched_exits_two(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(tmp_path / "c.json", [_sig(urn=ORDERS_URN, row_count=1)])
    rc = main(["diff", ORDERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 2
    assert "not watched" in capsys.readouterr().err.lower()


def test_diff_empty_store_exits_two(tmp_path, capsys):
    store = tmp_path / "s.json"  # never created
    cand = _write_sigs(tmp_path / "c.json", [_sig()])
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 2
    assert "store is empty" in capsys.readouterr().err.lower()


def test_diff_urn_absent_from_signatures_exits_two(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    # File is valid but has a different dataset — nothing to compare for this URN.
    cand = _write_sigs(tmp_path / "c.json", [_sig(urn=ORDERS_URN, row_count=1)])
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert rc == 2
    assert "not present" in capsys.readouterr().err.lower()


def test_diff_malformed_signatures_exits_two(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    bad = tmp_path / "c.json"
    bad.write_text("{ not json", encoding="utf-8")
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(bad)])
    assert rc == 2
    assert "valid json" in capsys.readouterr().err.lower()


def test_diff_is_read_only_store_untouched(tmp_path):
    """diff must never advance a baseline — the store bytes are identical after a drift."""
    store = tmp_path / "s.json"
    _seed_diff(store)
    before = store.read_bytes()
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string"), ("phone", "string")],
              row_count=2000, field_null_fractions={"email": 0.9})],
    )
    main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand)])
    assert store.read_bytes() == before


def test_diff_json_shape(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "bigint"), ("email", "string"), ("phone", "string")],
              row_count=1200, field_null_fractions={"email": 0.60})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand),
               "--json"])
    assert rc == 1
    d = json.loads(capsys.readouterr().out)["diff"]
    assert d["urn"] == CUSTOMERS_URN
    assert d["identical"] is False
    assert [f["path"] for f in d["fields_added"]] == ["phone"]
    assert d["fields_removed"] == []
    assert d["fields_type_changed"][0] == {"path": "id", "old_type": "int",
                                           "new_type": "bigint"}
    assert d["null_fraction_changed"][0]["path"] == "email"
    assert d["row_count"] == {"old": 1000, "new": 1200, "delta": 200, "changed": True}
    assert d["schema_hash"]["changed"] is True


def test_diff_json_identical_exits_zero(tmp_path, capsys):
    store = tmp_path / "s.json"
    _seed_diff(store)
    cand = _write_sigs(
        tmp_path / "c.json",
        [_sig(schema_fields=[("id", "int"), ("email", "string")], row_count=1000,
              field_null_fractions={"email": 0.25})],
    )
    rc = main(["diff", CUSTOMERS_URN, "--store", str(store), "--signatures", str(cand),
               "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["diff"]["identical"] is True


def test_diff_registered_in_help_signatures_required():
    ns = build_parser().parse_args(
        ["diff", CUSTOMERS_URN, "--signatures", "x.json"]
    )
    assert ns.func.__name__ == "cmd_diff"
    assert ns.urn == CUSTOMERS_URN
    assert ns.signatures == "x.json"
    # --signatures is mandatory: omitting it is a parse error (exit 2 via SystemExit).
    with pytest.raises(SystemExit):
        build_parser().parse_args(["diff", CUSTOMERS_URN])


# ---- incidents temporal axis: age display, --stale hunt, --sort recent -------------------

# An epoch so far in the past (2001-09-09) that it's stale under any real --stale threshold,
# while a sighting stamped at time.time() in-test is effectively age ~0.
_ANCIENT = 1_000_000_000.0


def test_incidents_shows_age_when_timestamped(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    # A day-old sighting renders a concrete "… ago" age.
    s.record_incident("fp", severity="high", title="HIGH drift", now=time.time() - 90000)
    s.save()
    assert main(["incidents", "--store", str(store_path)]) == 0
    out = capsys.readouterr().out
    assert "last seen 1d ago" in out


def test_incidents_fresh_sighting_reads_just_now_without_ago(tmp_path, capsys):
    # A sub-minute age reads "just now" on its own — never the awkward "just now ago".
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp", severity="high", title="HIGH drift", now=time.time())
    s.save()
    assert main(["incidents", "--store", str(store_path)]) == 0
    out = capsys.readouterr().out
    assert "last seen just now" in out
    assert "just now ago" not in out


def test_incidents_untimed_record_shows_no_age(tmp_path, capsys):
    # A legacy/untimed incident renders normally but never fakes an age.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp", severity="high", title="HIGH drift")  # no now
    s.save()
    assert main(["incidents", "--store", str(store_path)]) == 0
    assert "last seen" not in capsys.readouterr().out


def test_incidents_stale_filters_out_recent(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("old_fp", severity="high", title="OLD", now=_ANCIENT)
    s.record_incident("fresh_fp", severity="high", title="FRESH", now=time.time())
    s.save()
    assert main(["incidents", "--store", str(store_path), "--stale", "1h", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["old_fp"]  # only the drift not seen recently survives


def test_incidents_stale_skips_untimed_records(tmp_path, capsys):
    # A record with no last_seen can't be proven stale, so --stale drops it rather than
    # guessing — only the demonstrably-old one comes through.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("untimed_fp", severity="high", title="UNTIMED")  # no now
    s.record_incident("old_fp", severity="high", title="OLD", now=_ANCIENT)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--stale", "1h", "--json"]) == 0
    fps = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert fps == ["old_fp"]


def test_incidents_stale_bad_duration_is_error(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("fp", severity="high", now=_ANCIENT)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--stale", "soon"]) == 2
    assert "duration" in capsys.readouterr().out


def test_incidents_stale_fingerprints_pipe_selector(tmp_path, capsys):
    # --stale composes with --fingerprints so `incidents --stale 30d --fingerprints |
    # ogle resolve -` clears drift that stopped recurring.
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("old_fp", severity="high", now=_ANCIENT)
    s.record_incident("fresh_fp", severity="high", now=time.time())
    s.save()
    assert main(
        ["incidents", "--store", str(store_path), "--stale", "1h", "--fingerprints"]
    ) == 0
    assert capsys.readouterr().out.strip() == "old_fp"


def test_incidents_sort_recent_freshest_first(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("old_fp", severity="low", now=1000.0)
    s.record_incident("fresh_fp", severity="low", now=5000.0)
    s.save()
    assert main(["incidents", "--store", str(store_path), "--sort", "recent", "--json"]) == 0
    order = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert order == ["fresh_fp", "old_fp"]


def test_incidents_sort_recent_sinks_untimed_last(tmp_path, capsys):
    store_path = tmp_path / "baselines.json"
    s = BaselineStore(path=store_path)
    s.record_incident("timed_fp", severity="low", now=5000.0)
    s.record_incident("untimed_fp", severity="low")  # no now → sinks last
    s.save()
    assert main(["incidents", "--store", str(store_path), "--sort", "recent", "--json"]) == 0
    order = [e["fingerprint"] for e in json.loads(capsys.readouterr().out)["incidents"]]
    assert order == ["timed_fp", "untimed_fp"]


def test_incidents_stale_and_recent_registered_in_help():
    ns = build_parser().parse_args(["incidents", "--stale", "7d", "--sort", "recent"])
    assert ns.func.__name__ == "cmd_incidents"
    assert ns.stale == "7d"
    assert ns.sort == "recent"


def test_parse_age_accepts_units():
    assert _parse_age("45s") == 45
    assert _parse_age("30m") == 1800
    assert _parse_age("3h") == 10800
    assert _parse_age("2d") == 172800
    assert _parse_age("1w") == 604800
    assert _parse_age(" 12H ") == 43200  # whitespace + case tolerant


def test_parse_age_rejects_bad_input():
    assert _parse_age("3") is None       # bare number: ambiguous unit
    assert _parse_age("d") is None       # no amount
    assert _parse_age("0d") is None      # zero matches everything — rejected
    assert _parse_age("-5h") is None     # negative
    assert _parse_age("3y") is None      # unsupported unit
    assert _parse_age("") is None


def test_fmt_age_picks_largest_whole_unit():
    assert _fmt_age(10) == "just now"
    assert _fmt_age(-5) == "just now"    # clock skew never reads negative
    assert _fmt_age(300) == "5m"
    assert _fmt_age(7200) == "2h"
    assert _fmt_age(172800) == "2d"
    assert _fmt_age(1209600) == "2w"


# ---- --retract-cleared: guard + live wiring ---------------------------------------
def test_retract_cleared_offline_is_input_error(tmp_path, capsys):
    """Retraction needs a live graph; the offline signatures path can't reach DataHub."""
    store = tmp_path / "b.json"
    sigs = _write_sigs(tmp_path / "s.json", [_sig()])
    rc = main(["check", "--store", str(store), "--signatures", str(sigs), "--retract-cleared"])
    assert rc == 2
    assert "requires a live walk" in capsys.readouterr().err


def test_retract_cleared_passes_recovered_datasets_to_retract(tmp_path, capsys, monkeypatch):
    """Live run: healthy scored datasets (not drifting) are the recovered set handed to retract."""
    from ogle.walker import WalkResult

    store = tmp_path / "b.json"
    # Seed baselines so a second identical run scores both datasets healthy (no drift).
    seed = [_sig(CUSTOMERS_URN, row_count=1000), _sig(ORDERS_URN, row_count=2000)]
    wr = WalkResult(
        signatures=seed,
        serving_dataset_urns={CUSTOMERS_URN},
        dataset_to_models={CUSTOMERS_URN: ["urn:li:mlModel:(x,churn,PROD)"]},
    )
    monkeypatch.setattr("ogle.cli._walk_live", lambda gms, models, discover: (seed, sorted(wr.serving_dataset_urns), wr))

    captured = {}

    def _fake_retract(recovered, active, walk_result, gms):
        captured["recovered"] = list(recovered)
        captured["active"] = list(active)
        from ogle.writeback import WritebackPlan, WritebackResult
        return WritebackPlan(), WritebackResult()

    monkeypatch.setattr("ogle.cli._do_retract", _fake_retract)

    # First run seeds (healthy). Second run: still healthy -> both datasets recovered candidates.
    main(["check", "--store", str(store), "--gms", "http://x", "--discover"])
    capsys.readouterr()
    rc = main(["check", "--store", str(store), "--gms", "http://x", "--discover", "--retract-cleared"])

    assert rc == 0
    assert set(captured["recovered"]) == {CUSTOMERS_URN, ORDERS_URN}
    assert captured["active"] == []  # no drift this run
    assert "nothing to clear" in capsys.readouterr().out
