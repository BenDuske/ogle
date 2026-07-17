"""Unit tests for ogle.cli — the `ogle check` entrypoint over the drift-check pipeline.

Driven entirely through the OFFLINE signatures-file path (`--signatures`), so the whole
command is exercised end-to-end with no `datahub` SDK and no Docker quickstart: load store
-> read signatures JSON -> run_drift_check -> render -> save store -> exit code. Fixtures
use the Task #2 shape where `customers` feeds the deployed `churn_predictor` (serving path).
"""

import json

import pytest

import io

from ogle.cli import _emit, build_parser, load_signatures_file, main
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
