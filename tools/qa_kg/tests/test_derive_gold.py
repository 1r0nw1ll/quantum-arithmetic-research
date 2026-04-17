"""Unit tests for tools.qa_kg.analysis.derive_gold.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Uses an in-memory SQLite DB with a small synthetic fixture. Does not touch
the live qa_kg.db.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sqlite3

import pytest

from tools.qa_kg.analysis import derive_gold as dg
from tools.qa_kg.schema import init_db
from tools.qa_kg.kg import Edge, KG, Node


@pytest.fixture
def kg() -> KG:
    conn = init_db(":memory:")
    return KG(conn)


def _seed_cert_chain(kg: KG) -> None:
    """[225] supersedes chain v1 → v4 with lifecycle states."""
    for v, life in (("v1", "superseded"), ("v2", "superseded"),
                    ("v3", "superseded"), ("v4", "current")):
        kg.upsert_node(Node(
            id=f"cert:fs:qa_test_chain_{v}",
            node_type="Cert",
            title=f"Test chain cert {v}",
            body=f"consistency invariants {v}",
            authority="derived",
            epistemic_status="certified",
            lifecycle_state=life,
        ))
    for newer, older in (("v4", "v3"), ("v3", "v2"), ("v2", "v1")):
        kg.upsert_edge(Edge(
            src_id=f"cert:fs:qa_test_chain_{newer}",
            dst_id=f"cert:fs:qa_test_chain_{older}",
            edge_type="supersedes",
        ))


def test_tokenize_nfkd_superscripts():
    """NFKD decomposes ² → 2 so math expressions match across orthographies."""
    assert "a2" in dg._tokenize("a²")
    assert "b2" in dg._tokenize("b²")


def test_tokenize_single_letter_variables_kept():
    """Single-letter variable names are content, not stopwords."""
    assert "c" in dg._tokenize("C identity")
    assert "b" in dg._tokenize("b and h")
    assert "h" in dg._tokenize("b and h")


def test_tokenize_stopwords_filtered():
    """Common connectives and query-generic tokens removed."""
    tokens = dg._tokenize("What primary source grounds the cert?")
    assert "what" not in tokens
    assert "primary" not in tokens
    assert "source" not in tokens


def test_derive_provenance_strips_axioms(kg: KG) -> None:
    """BFS output omits node_type='Axiom' even when reachable."""
    kg.upsert_node(Node(
        id="axiom:T_TEST",
        node_type="Axiom",
        title="Test axiom",
        body="test",
        authority="primary",
        epistemic_status="axiom",
    ))
    kg.upsert_node(Node(
        id="sc:test_primary",
        node_type="Claim",
        title="Test primary claim",
        body="test",
        authority="primary",
        epistemic_status="source_claim",
    ))
    kg.upsert_node(Node(
        id="cert:fs:test_cert",
        node_type="Cert",
        title="Test cert",
        body="test",
        authority="derived",
        epistemic_status="certified",
    ))
    kg.upsert_edge(Edge(
        src_id="cert:fs:test_cert", dst_id="axiom:T_TEST",
        edge_type="derived-from", method="structural",
    ))
    kg.upsert_edge(Edge(
        src_id="cert:fs:test_cert", dst_id="sc:test_primary",
        edge_type="derived-from", method="structural",
    ))
    result = dg._derive_provenance(kg.conn, "cert:fs:test_cert")
    assert result["gold_ids"] == ["sc:test_primary"]
    assert result["axioms_stripped"] == ["axiom:T_TEST"]


def test_derive_provenance_skips_keyword_method(kg: KG) -> None:
    """Keyword edges are excluded (mirrors kg.why() semantics)."""
    kg.upsert_node(Node(
        id="cert:fs:kw_cert", node_type="Cert", title="KW", body="",
        authority="derived", epistemic_status="certified",
    ))
    kg.upsert_node(Node(
        id="sc:kw_target", node_type="Claim", title="target", body="",
        authority="primary", epistemic_status="source_claim",
    ))
    kg.upsert_edge(Edge(
        src_id="cert:fs:kw_cert", dst_id="sc:kw_target",
        edge_type="derived-from", method="keyword",
    ))
    result = dg._derive_provenance(kg.conn, "cert:fs:kw_cert")
    assert result["gold_ids"] == []


def test_derive_contradiction_lexical_audit_both_pass(kg: KG) -> None:
    """When query shares ≥2 tokens with both endpoints, audit passes."""
    kg.upsert_node(Node(
        id="sc:lex_sc", node_type="Claim",
        title="Alpha beta gamma",
        body="delta epsilon zeta",
        authority="primary", epistemic_status="source_claim",
    ))
    kg.upsert_node(Node(
        id="obs:lex_obs", node_type="Claim",
        title="Delta epsilon observation",
        body="beta alpha verification record",
        authority="internal", epistemic_status="observation",
    ))
    kg.upsert_edge(Edge(
        src_id="sc:lex_sc", dst_id="obs:lex_obs",
        edge_type="contradicts",
        provenance='{"reason": "typo"}',
    ))
    result = dg._derive_contradiction(
        kg.conn, ["sc:lex_sc", "obs:lex_obs"],
        "alpha beta delta epsilon",
    )
    assert result["gold_ids"] == ["obs:lex_obs", "sc:lex_sc"]
    assert result["lexical_audit"]["both_pass"] is True
    assert result["reason"] == "typo"


def test_derive_contradiction_lexical_audit_one_fails(kg: KG) -> None:
    """Endpoint with <2 shared tokens is flagged."""
    kg.upsert_node(Node(
        id="sc:thin", node_type="Claim",
        title="Thin claim", body="unique_sc_token",
        authority="primary", epistemic_status="source_claim",
    ))
    kg.upsert_node(Node(
        id="obs:thick", node_type="Claim",
        title="Thick observation matching the query well",
        body="alpha beta gamma delta verbose",
        authority="internal", epistemic_status="observation",
    ))
    kg.upsert_edge(Edge(
        src_id="sc:thin", dst_id="obs:thick",
        edge_type="contradicts", provenance='{"reason": "dispute"}',
    ))
    result = dg._derive_contradiction(
        kg.conn, ["sc:thin", "obs:thick"],
        "alpha beta gamma delta",
    )
    assert result["lexical_audit"]["both_pass"] is False


def test_derive_domain_filters_to_source_claim(kg: KG) -> None:
    """Domain derivation returns only source_claim nodes, not certs/works."""
    kg.upsert_node(Node(
        id="sc:d_sc", node_type="Claim", title="SC", body="",
        authority="primary", epistemic_status="source_claim",
        domain="geometry",
    ))
    kg.upsert_node(Node(
        id="work:d_work", node_type="Work", title="Work", body="",
        authority="primary", epistemic_status="source_work",
        domain="geometry",
    ))
    kg.upsert_node(Node(
        id="sc:other_domain", node_type="Claim", title="Other", body="",
        authority="primary", epistemic_status="source_claim",
        domain="biology",
    ))
    result = dg._derive_domain(kg.conn, "geometry")
    assert result["gold_ids"] == ["sc:d_sc"]
    assert result["count"] == 1


def test_derive_authority_validates_primary(kg: KG) -> None:
    """Curator-specified ID must exist AND be primary+source_claim."""
    kg.upsert_node(Node(
        id="sc:auth_ok", node_type="Claim", title="OK", body="",
        authority="primary", epistemic_status="source_claim",
        domain="physics",
    ))
    result = dg._derive_authority(kg.conn, "sc:auth_ok")
    assert result["gold_ids"] == ["sc:auth_ok"]
    assert result["validation"]["is_primary_source_claim"] is True


def test_derive_authority_rejects_non_primary(kg: KG) -> None:
    """Derived or agent node is not accepted as curator anchor."""
    kg.upsert_node(Node(
        id="cert:fs:wrong_authority", node_type="Cert", title="Wrong", body="",
        authority="derived", epistemic_status="certified",
    ))
    result = dg._derive_authority(kg.conn, "cert:fs:wrong_authority")
    assert result["gold_ids"] == []
    assert result["validation"]["is_primary_source_claim"] is False


def test_derive_authority_missing_node(kg: KG) -> None:
    """Nonexistent node ID surfaces a validation error."""
    result = dg._derive_authority(kg.conn, "sc:does_not_exist")
    assert result["gold_ids"] == []
    assert "error" in result["validation"]


def test_derive_lifecycle_chain(kg: KG) -> None:
    _seed_cert_chain(kg)
    result = dg._derive_lifecycle(kg.conn, "qa_test_chain")
    assert result["gold_ids"] == ["cert:fs:qa_test_chain_v4"]
    assert set(result["anti_gold_ids"]) == {
        "cert:fs:qa_test_chain_v1",
        "cert:fs:qa_test_chain_v2",
        "cert:fs:qa_test_chain_v3",
    }


def test_derive_edge_case_shapes():
    q_empty = {"expected_result": "empty"}
    assert dg._derive_edge_case(q_empty)["gold_ids"] == []
    q_primary = {"expected_result": "no_derived_in_top_k"}
    assert "primary" in dg._derive_edge_case(q_primary)["expected_predicate"]
    q_valid = {"expected_result": "valid_at_passthrough"}
    assert "passes" in dg._derive_edge_case(q_valid)["expected_predicate"]
    q_unknown = {"expected_result": "mystery"}
    assert "unknown" in dg._derive_edge_case(q_unknown)["expected_predicate"]
