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


# ---- summary_line (at-a-glance blast radius) ----

def test_summary_line_counts_findings_and_datasets():
    inc = build_incident([
        _finding(urn=CUSTOMERS_URN, kind=DriftKind.SCHEMA, sev=Severity.HIGH),
        _finding(urn=CUSTOMERS_URN, kind=DriftKind.QUALITY, sev=Severity.MEDIUM),
        _finding(urn=ORDERS_URN, kind=DriftKind.VOLUME, sev=Severity.MEDIUM),
    ])
    line = inc.summary_line
    assert "3 findings" in line
    assert "2 datasets" in line
    assert "1" in line and "high" in line
    assert "2" in line and "medium" in line


def test_summary_line_singular_grammar_and_no_serving_tag():
    inc = build_incident([_finding(sev=Severity.LOW, kind=DriftKind.VOLUME, msg="x")])
    line = inc.summary_line
    assert "1 finding**" in line          # singular, not "findings"
    assert "across 1 dataset —" in line   # singular, not "datasets"
    assert "serving path impacted" not in line


def test_summary_line_flags_serving_and_omits_absent_severities():
    inc = build_incident([_finding(sev=Severity.HIGH, serving=True)])
    line = inc.summary_line
    assert "serving path impacted" in line
    assert "medium" not in line and "low" not in line  # only the present band shows


def test_summary_line_appears_in_markdown_under_title():
    inc = build_incident([_finding(msg="schema changed: removed ['region']")])
    md = render_markdown(inc)
    assert inc.summary_line in md
    # ordering: title line precedes the summary line
    assert md.index(inc.title) < md.index("finding")


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


# ---- owner attribution (who to page) ----

def test_owner_line_rendered_under_its_dataset():
    inc = build_incident(
        [_finding(msg="schema changed: removed ['region']")],
        owners={CUSTOMERS_URN: ["data-platform-team"]},
    )
    md = render_markdown(inc)
    assert "\U0001f464 owner: data-platform-team" in md


def test_owner_line_plural_for_multiple_owners():
    inc = build_incident(
        [_finding(msg="x")],
        owners={CUSTOMERS_URN: ["alice", "bob"]},
    )
    md = render_markdown(inc)
    assert "\U0001f464 owners: alice, bob" in md


def test_no_owner_line_when_owners_absent():
    md = render_markdown(build_incident([_finding(msg="x")]))
    assert "\U0001f464" not in md


def test_owners_normalized_strip_dedup_and_drop_empties():
    inc = build_incident(
        [_finding(msg="x")],
        owners={CUSTOMERS_URN: ["  alice  ", "alice", "", "  ", "bob"]},
    )
    assert inc.owners[CUSTOMERS_URN] == ["alice", "bob"]


def test_owners_restricted_to_incident_urns():
    # An owner for a dataset that isn't in this incident must never leak into the alert.
    inc = build_incident(
        [_finding(urn=CUSTOMERS_URN, msg="x")],
        owners={ORDERS_URN: ["someone-else"]},
    )
    assert inc.owners == {}
    assert "someone-else" not in render_markdown(inc)


def test_urn_with_only_blank_owners_is_omitted():
    inc = build_incident([_finding(msg="x")], owners={CUSTOMERS_URN: ["", "   "]})
    assert CUSTOMERS_URN not in inc.owners


def test_ownership_does_not_change_fingerprint():
    # Re-assigning an owner is not drift; a still-open incident must not re-page.
    findings = [_finding(msg="x")]
    a = build_incident(findings, owners={CUSTOMERS_URN: ["alice"]})
    b = build_incident(findings, owners={CUSTOMERS_URN: ["bob"]})
    c = build_incident(findings)  # no owners at all
    assert a.fingerprint == b.fingerprint == c.fingerprint


def test_to_dict_carries_owners():
    inc = build_incident([_finding(msg="x")], owners={CUSTOMERS_URN: ["alice"]})
    assert inc.to_dict()["owners"] == {CUSTOMERS_URN: ["alice"]}


def test_narrate_surfaces_owner_without_llm():
    out = narrate([_finding(msg="x")], owners={CUSTOMERS_URN: ["on-call-ml"]})
    assert "on-call-ml" in out


def test_llm_prompt_grounds_on_owner_facts():
    inc = build_incident([_finding(msg="x")], owners={CUSTOMERS_URN: ["on-call-ml"]})
    prompt = build_llm_prompt(inc)
    assert "on-call-ml" in prompt      # the fact is handed to the model
    assert "do not invent" in prompt and "owners" in prompt  # and it's forbidden to fabricate


# ---- distribution drift renders end-to-end ----

def test_distribution_finding_renders_with_action_line():
    """A DISTRIBUTION finding must render (no KeyError) and carry its own What-to-check."""
    inc = build_incident([
        _finding(kind=DriftKind.DISTRIBUTION, sev=Severity.HIGH,
                 msg="distinct-value fraction dropped on region (80%->5%)")
    ])
    md = render_markdown(inc)
    assert "distribution" in md
    assert "region (80%->5%)" in md
    assert "**What to check**" in md
    # the distribution-specific advice, not another kind's line
    assert "fan-out join" in md


def test_distribution_finding_from_real_scorer_path():
    """End-to-end: a real cardinality-collapse finding flows scorer -> incident -> markdown."""
    base = _sig(field_unique_fractions={"region": 0.90})
    cur = _sig(field_unique_fractions={"region": 0.05})
    findings = score_dataset(base, cur)
    inc = build_incident(findings)
    assert inc is not None
    md = render_markdown(inc)
    assert "**distribution**" in md
