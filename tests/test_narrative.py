"""Unit tests for ogle.narrative — findings -> human-readable incident.

Fixtures reuse the Task #2 ML layer: `customers` feeds `customer_purchase_features`, which
feeds the deployed `churn_predictor`. So a serving-path incident is the realistic case.
Everything here is pure — no live DataHub, no real LLM (the model seam is exercised with a
fake callable).
"""

import pytest

from ogle.narrative import (
    Incident,
    build_incident,
    build_llm_prompt,
    incident_fingerprint,
    narrate,
    render_markdown,
    short_name,
)
from ogle.scorer import DriftFinding, DriftKind, Severity, score_dataset
from ogle.signature import build_signature

CUSTOMERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.customers,PROD)"
ORDERS_URN = "urn:li:dataset:(urn:li:dataPlatform:dbt,b2fd91.orders,PROD)"


def _sig(urn=CUSTOMERS_URN, **kw):
    kw.setdefault("schema_fields", [("id", "int"), ("email", "string"), ("region", "string")])
    kw.setdefault("row_count", 10_000)
    return build_signature(urn=urn, **kw)


def _finding(urn=CUSTOMERS_URN, kind=DriftKind.SCHEMA, sev=Severity.HIGH, msg="x", **details):
    return DriftFinding(urn=urn, kind=kind, severity=sev, message=msg, details=details)


# ---- short_name ----

def test_short_name_extracts_dataset_name():
    assert short_name(CUSTOMERS_URN) == "b2fd91.customers"


def test_short_name_falls_back_on_unexpected_shape():
    assert short_name("not-a-real-urn") == "not-a-real-urn"


# ---- build_incident ----

def test_no_findings_yields_no_incident():
    assert build_incident([]) is None


def test_overall_severity_is_the_worst_finding():
    inc = build_incident([_finding(sev=Severity.LOW), _finding(sev=Severity.HIGH, kind=DriftKind.VOLUME)])
    assert inc.overall_severity == Severity.HIGH


def test_serving_flag_set_when_any_finding_is_serving():
    inc = build_incident([_finding(sev=Severity.MEDIUM, serving=True)])
    assert inc.serving_impacted is True
    assert "serving path" in inc.title


def test_urns_deduped_and_order_preserved():
    inc = build_incident([
        _finding(urn=CUSTOMERS_URN, kind=DriftKind.SCHEMA),
        _finding(urn=ORDERS_URN, kind=DriftKind.VOLUME),
        _finding(urn=CUSTOMERS_URN, kind=DriftKind.QUALITY),
    ])
    assert inc.urns == [CUSTOMERS_URN, ORDERS_URN]


def test_findings_ranked_most_severe_first():
    inc = build_incident([_finding(sev=Severity.LOW), _finding(sev=Severity.HIGH, kind=DriftKind.VOLUME)])
    ranks = [f.severity.rank for f in inc.findings]
    assert ranks == sorted(ranks, reverse=True)


# ---- fingerprint (Aegis dedup) ----

def test_fingerprint_is_order_independent():
    a = [_finding(kind=DriftKind.SCHEMA), _finding(kind=DriftKind.VOLUME)]
    b = list(reversed(a))
    assert incident_fingerprint(a) == incident_fingerprint(b)


def test_fingerprint_changes_when_severity_changes():
    base = incident_fingerprint([_finding(sev=Severity.MEDIUM)])
    worse = incident_fingerprint([_finding(sev=Severity.HIGH)])
    assert base != worse


def test_fingerprint_changes_when_a_new_dataset_joins():
    one = incident_fingerprint([_finding(urn=CUSTOMERS_URN)])
    two = incident_fingerprint([_finding(urn=CUSTOMERS_URN), _finding(urn=ORDERS_URN)])
    assert one != two


# ---- render_markdown ----

def test_markdown_contains_title_datasets_and_fingerprint():
    inc = build_incident([_finding(msg="schema changed: removed ['region']")])
    md = render_markdown(inc)
    assert inc.title in md
    assert "b2fd91.customers" in md
    assert inc.fingerprint in md
    assert "What to check" in md


def test_markdown_is_deterministic():
    findings = [
        _finding(kind=DriftKind.SCHEMA, sev=Severity.HIGH, msg="a"),
        _finding(kind=DriftKind.QUALITY, sev=Severity.MEDIUM, msg="b"),
    ]
    assert render_markdown(build_incident(findings)) == render_markdown(build_incident(findings))


def test_actions_deduped_one_per_kind():
    # two SCHEMA findings -> a single schema action line
    inc = build_incident([
        _finding(kind=DriftKind.SCHEMA, sev=Severity.HIGH, msg="a"),
        _finding(kind=DriftKind.SCHEMA, sev=Severity.LOW, msg="b", urn=ORDERS_URN),
    ])
    md = render_markdown(inc)
    assert md.count("check the upstream transform") == 1


# ---- build_llm_prompt ----

def test_llm_prompt_grounds_on_facts_and_forbids_invention():
    inc = build_incident([_finding(msg="row count collapsed 10000 -> 0")])
    prompt = build_llm_prompt(inc)
    assert "do not invent" in prompt.lower()
    assert "row count collapsed 10000 -> 0" in prompt


# ---- narrate (the seam) ----

def test_narrate_no_findings_is_clean_heartbeat():
    assert "No drift" in narrate([])


def test_narrate_without_llm_returns_deterministic_markdown():
    findings = [_finding(msg="schema changed: removed ['region']")]
    assert narrate(findings) == render_markdown(build_incident(findings))


def test_narrate_uses_llm_when_provided():
    calls = {}

    def fake_llm(prompt):
        calls["prompt"] = prompt
        return "Churn predictor's customers source lost the region column. Retrain blocked."

    out = narrate([_finding(msg="schema changed: removed ['region']")], llm=fake_llm)
    assert "region column" in out
    assert "FACTS:" in calls["prompt"]  # got the grounded prompt


def test_narrate_falls_back_when_llm_raises():
    def broken_llm(_):
        raise RuntimeError("model down")

    findings = [_finding(msg="x")]
    assert narrate(findings, llm=broken_llm) == render_markdown(build_incident(findings))


def test_narrate_falls_back_when_llm_returns_empty():
    findings = [_finding(msg="x")]
    assert narrate(findings, llm=lambda _: "   ") == render_markdown(build_incident(findings))


# ---- end-to-end from the scorer ----

def test_end_to_end_from_score_dataset():
    base = _sig(row_count=10_000)
    cur = _sig(row_count=0)  # collapse
    findings = score_dataset(base, cur, serving=True)
    out = narrate(findings)
    assert "HIGH" in out
    assert "serving path" in out
    assert "b2fd91.customers" in out
