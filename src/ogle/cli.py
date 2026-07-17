"""Ogle CLI — the operator-facing entrypoint over the drift-check pipeline.

`ogle check` is the whole loop in one command:

    load baselines  ->  walk (live DataHub | offline signatures file)  ->  run_drift_check
                                                                                |
                                    save baselines  <-  render/alert  <---------+

Two input modes so the command is useful with OR without a live DataHub quickstart:

  * **Live walk** (`--gms` + `--models`/`--discover`) — pulls aspects through
    `ogle.walker.DataHubBackend`. Needs the `datahub` extra + a reachable GMS.
  * **Offline signatures** (`--signatures FILE`) — feeds pre-computed `DatasetSignature`s
    from JSON. No SDK, no Docker. This is how a scheduled job can hand Ogle signatures it
    pulled elsewhere, and how the CLI is unit-tested end-to-end.

Exit codes are chosen so a cron/Task wrapper can branch on them:
    0  healthy — no *new* incident to alert on (may include seeded/first-run datasets)
    1  a NEW incident fired (`DriftReport.should_alert`) — page Ben
    2  usage / input error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .llm import build_narrator
from .narrative import narrate
from .pipeline import DriftReport, run_drift_check
from .signature import DatasetSignature
from .store import BaselineStore

DEFAULT_STORE = "ogle-baselines.json"


def _emit(text: str, *, stream=None) -> None:
    """Print `text` without crashing on a legacy console encoding.

    The narrative carries emoji severity markers; a Windows cp1252 terminal raises
    UnicodeEncodeError on those. Encode to the stream's own encoding with errors="replace"
    so the command still runs (and its exit code stays trustworthy) on any console.
    """
    stream = stream or sys.stdout
    enc = getattr(stream, "encoding", None) or "utf-8"
    stream.write(text.encode(enc, errors="replace").decode(enc) + "\n")


# ---------------------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------------------
def load_signatures_file(path: Path) -> Tuple[List[DatasetSignature], List[str]]:
    """Read signatures (and optional serving URNs) from a JSON file.

    Accepts either shape:
      * a bare list  ``[ <sig.to_dict()>, ... ]``  (no serving URNs), or
      * an object    ``{"signatures": [...], "serving_urns": [...]}``.

    Returns (signatures, serving_urns). Raises ``ValueError`` with an operator-readable
    message on any malformed input — the CLI turns that into exit code 2.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"signatures file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"signatures file is not valid JSON ({path}): {exc}")

    if isinstance(data, list):
        raw_sigs, serving = data, []
    elif isinstance(data, dict):
        raw_sigs = data.get("signatures", [])
        serving = list(data.get("serving_urns", []))
    else:
        raise ValueError("signatures file must be a JSON list or object")

    if not isinstance(raw_sigs, list):
        raise ValueError('"signatures" must be a JSON list')

    signatures: List[DatasetSignature] = []
    for i, item in enumerate(raw_sigs):
        if not isinstance(item, dict) or "urn" not in item:
            raise ValueError(f"signature #{i} is missing a urn / is not an object")
        signatures.append(DatasetSignature.from_dict(item))
    return signatures, serving


def _walk_live(gms: str, models: Sequence[str], discover: bool):
    """Run a live DataHub walk. Imported lazily so the SDK stays an optional extra.

    Returns (signatures, sorted_serving_urns, walk_result) — the full `WalkResult` rides
    along so the caller can hand it to `writeback.plan_writeback` without re-walking.
    """
    from .walker import DataHubBackend, walk_models

    backend = DataHubBackend(gms_server=gms)
    model_urns: List[str] = list(models)
    if discover:
        model_urns.extend(backend.discover_deployed_models())
    if not model_urns:
        raise ValueError(
            "no models to walk — pass --models URN [URN ...] or --discover"
        )
    # Dedup while preserving order so diagnostics list each model once.
    seen = set()
    ordered = [u for u in model_urns if not (u in seen or seen.add(u))]
    result = walk_models(backend, ordered)
    return result.signatures, sorted(result.serving_dataset_urns), result


def _do_writeback(findings, walk_result, gms: str):
    """Live tag write-back. Imported lazily like the walker."""
    from .writeback import DataHubWritebackBackend, apply, plan_writeback

    backend = DataHubWritebackBackend(gms_server=gms)
    plan = plan_writeback(findings, walk_result)
    return plan, apply(plan, backend)


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------
def render_report(report: DriftReport, *, as_json: bool) -> str:
    """Human markdown by default; machine JSON with --json."""
    if as_json:
        return json.dumps(report.to_dict(), indent=2, sort_keys=True)

    lines = [report.narrative.rstrip()]
    tail = []
    if report.new_urns:
        tail.append(f"seeded {len(report.new_urns)} new dataset(s) (first sighting)")
    if report.scored_urns:
        tail.append(f"checked {len(report.scored_urns)} dataset(s)")
    if tail:
        lines.append("")
        lines.append("_" + "; ".join(tail) + "._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    store = BaselineStore.load(store_path)

    # Gather the current signatures + which of them feed a serving model.
    walk_result = None  # None in offline mode; a real WalkResult in live mode.
    try:
        if args.signatures:
            signatures, serving = load_signatures_file(Path(args.signatures))
        else:
            signatures, serving, walk_result = _walk_live(
                args.gms, args.models or [], args.discover
            )
        # An explicit --serving on the command line augments whatever the source reported.
        serving = sorted(set(serving) | set(args.serving or []))
    except ValueError as exc:
        print(f"ogle check: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # live-walk failure (SDK/network) — report, don't traceback
        print(f"ogle check: live walk failed: {exc}", file=sys.stderr)
        return 2

    report = run_drift_check(
        store,
        signatures,
        serving_urns=serving,
        update_baselines=not args.no_update,
    )

    _emit(render_report(report, as_json=args.json))

    # Optional LLM-phrased incident summary. Only when there is something to narrate and
    # not in JSON mode (markdown prose would corrupt the JSON payload). `narrate` itself
    # falls back to the deterministic report if the model is unreachable, so a build/parse
    # error is the only thing we surface here.
    if getattr(args, "narrate", None) and not args.json and report.findings:
        try:
            narrator = build_narrator(args.narrate)
        except ValueError as exc:
            print(f"ogle check: {exc}", file=sys.stderr)
            return 2
        _emit("\n---\n\n**Incident summary**\n")
        _emit(narrate(report.findings, llm=narrator))

    # Persist advanced baselines/incident memory unless asked for a read-only probe.
    if not args.no_update:
        try:
            store.save(store_path)
        except Exception as exc:
            print(f"ogle check: warning — could not save store: {exc}", file=sys.stderr)

    # Optional outbound tag write-back — only when THIS run fired a new incident, so a
    # scheduler doesn't reapply the same tag on every tick.
    if args.write_back:
        if not report.should_alert:
            _emit("_write-back skipped: no new incident this run._")
        elif walk_result is None:
            print(
                "ogle check: --write-back requires a live walk (--gms/--models/--discover);"
                " offline mode has no way to reach DataHub.",
                file=sys.stderr,
            )
            return 2
        else:
            try:
                plan, wb_result = _do_writeback(report.findings, walk_result, args.gms)
            except Exception as exc:
                print(f"ogle check: write-back failed: {exc}", file=sys.stderr)
                # Alert still fires — the check itself succeeded; the write-back is an
                # optional side-effect. Return 1 (new incident), not 2.
                return 1
            _render_writeback(plan, wb_result, as_json=args.json)

    return 1 if report.should_alert else 0


def _render_writeback(plan, result, *, as_json: bool) -> None:
    if as_json:
        _emit(
            json.dumps(
                {"plan": plan.to_dict(), "result": result.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not plan.actions:
        _emit("_write-back: nothing to tag._")
        return
    lines = ["", f"**Tagged {len(result.tagged_entities)} entity(ies) in DataHub:**"]
    for urn in result.tagged_entities:
        lines.append(f"- `{urn}`")
    if result.unchanged:
        lines.append(f"_({len(result.unchanged)} already tagged, skipped)_")
    if result.failed:
        lines.append(f"_({len(result.failed)} failed — see logs)_")
    _emit("\n".join(lines))


# Repo-root examples/ (works from a clone: src/ogle/cli.py -> parents[2] == repo root).
_DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "demo"


def cmd_demo(args: argparse.Namespace) -> int:
    """Zero-setup, keyless proof: seed healthy baselines, then re-check drifted fixtures.

    Runs the *same* `run_drift_check` code path the live DataHub walk feeds — no SDK, no
    Docker, no API key — against the bundled `examples/demo/*.json` fixtures. First pass
    seeds and stays healthy (exit 0); second pass fires the HIGH serving-path incident that
    `examples/alerts/churn-orders-drift.md` captured. Exit 1 on that alert, matching a real
    `ogle check`, so a judge sees the whole loop in one command.
    """
    healthy = _DEMO_DIR / "healthy-signatures.json"
    drifted = _DEMO_DIR / "drifted-signatures.json"
    for f in (healthy, drifted):
        if not f.exists():
            print(f"ogle demo: bundled fixture not found: {f}", file=sys.stderr)
            return 2

    # In-memory store — the demo never touches the operator's cwd or a real baseline file.
    store = BaselineStore.load(_DEMO_DIR / "__demo_never_written__.json")

    try:
        h_sigs, h_serving = load_signatures_file(healthy)
        d_sigs, d_serving = load_signatures_file(drifted)
    except ValueError as exc:  # a corrupted bundled fixture — treat as input error
        print(f"ogle demo: {exc}", file=sys.stderr)
        return 2

    _emit("# Ogle offline demo — churn serving-path drift\n")
    _emit("_Keyless, no DataHub required; same drift-check code path as a live walk._\n")

    _emit("## 1. Seed baselines (healthy fixture)\n")
    seed = run_drift_check(store, h_sigs, serving_urns=h_serving, update_baselines=True)
    _emit(render_report(seed, as_json=False))

    _emit("\n## 2. Re-check the drifted fixture\n")
    drift = run_drift_check(store, d_sigs, serving_urns=d_serving, update_baselines=True)
    _emit(render_report(drift, as_json=False))

    # Optional feature-#2 showcase: the same LLM root-cause narrator `ogle check --narrate`
    # exposes, so a judge sees BOTH flagship features from the one keyless command. `narrate`
    # falls back to the deterministic summary when the model is unreachable, so a laptop with
    # no local Ollama still gets a clean section instead of an error — the demo stays keyless.
    if getattr(args, "narrate", None) and drift.findings:
        try:
            narrator = build_narrator(args.narrate)
        except ValueError as exc:
            print(f"ogle demo: {exc}", file=sys.stderr)
            return 2
        _emit("\n## 3. LLM root-cause summary\n")
        _emit(narrate(drift.findings, llm=narrator))

    _emit(
        "\n---\n_Reproduces `examples/alerts/churn-orders-drift.md`. "
        "Run against DataHub: `ogle check --gms http://localhost:8080 --discover`._"
    )
    return 1 if drift.should_alert else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogle",
        description="Ogle — ML-lineage drift agent for DataHub.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"ogle {__version__}")
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser(
        "check",
        help="Run a drift check: walk lineage, diff against baselines, alert on new drift.",
    )
    check.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help=f"Baseline/incident store JSON (default: {DEFAULT_STORE}).",
    )
    src = check.add_argument_group("input (choose live walk OR an offline signatures file)")
    src.add_argument(
        "--signatures",
        metavar="FILE",
        help="Offline: read DatasetSignatures from JSON instead of a live DataHub walk.",
    )
    src.add_argument(
        "--gms",
        default="http://localhost:8080",
        help="Live: DataHub GMS server URL (default: http://localhost:8080).",
    )
    src.add_argument(
        "--models",
        nargs="*",
        metavar="URN",
        help="Live: explicit mlModel URNs to walk.",
    )
    src.add_argument(
        "--discover",
        action="store_true",
        help="Live: also auto-discover every IN_SERVICE model and walk it.",
    )
    check.add_argument(
        "--serving",
        nargs="*",
        metavar="URN",
        help="Extra dataset URNs to treat as serving (severity-escalated).",
    )
    check.add_argument(
        "--no-update",
        action="store_true",
        help="Read-only probe: do not advance baselines or incident memory.",
    )
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit the DriftReport as JSON instead of markdown.",
    )
    check.add_argument(
        "--write-back",
        action="store_true",
        help=(
            "On a new incident (exit 1), stamp affected datasets and their downstream "
            "mlModels with `urn:li:tag:ogle-drift-flagged` in DataHub. Requires --gms."
        ),
    )
    check.add_argument(
        "--narrate",
        nargs="?",
        const="ollama",
        metavar="SPEC",
        help=(
            "After the report, add an LLM-phrased incident summary grounded in the "
            "computed facts. Optional SPEC picks the model (default: 'ollama' = local "
            "qwen3:latest); also 'ollama:<model>' or '<...>@http://host:11434'. Falls "
            "back to the deterministic report if the model is unreachable. Ignored with "
            "--json."
        ),
    )
    check.set_defaults(func=cmd_check)

    # `ogle demo` — one-command, keyless judge repro over the bundled fixtures.
    demo = sub.add_parser(
        "demo",
        help="Zero-setup offline drift demo (no DataHub/API key) — seeds, then alerts.",
    )
    demo.add_argument(
        "--narrate",
        nargs="?",
        const="ollama",
        metavar="SPEC",
        help=(
            "Also show the LLM root-cause summary (feature #2) after the alert. Optional "
            "SPEC picks the model (default: 'ollama' = local qwen3:latest); also "
            "'ollama:<model>' or '<...>@http://host:11434'. Falls back to the deterministic "
            "summary if the model is unreachable, so the demo stays keyless."
        ),
    )
    demo.set_defaults(func=cmd_demo)

    # `ogle watch` — one scheduler tick that pages once on a new incident.
    from .watch import build_watch_args

    build_watch_args(sub)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
