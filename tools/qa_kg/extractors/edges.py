"""Deterministic edge extractors.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Three passes:
1. cert → axiom 'derived-from' — every cert depends on at least {A1, A2, T2, NT}.
   Additional axioms added by keyword match in cert body.
2. rule → axiom 'instantiates' — Hard Rules enforce specific axioms, by keyword match.
3. cert → cert 'extends' — parse [N] references in cert body (pass_desc).

No LLM. All reversible (idempotent upsert).
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import re

from tools.qa_kg.kg import KG, Edge


# ---- axiom linkage rules ------------------------------------------------------

# Every cert transitively depends on these (state rep + firewall).
BASELINE_AXIOMS = ("A1", "A2", "T2", "NT")

# Keyword → extra axiom id.
AXIOM_KEYWORDS = {
    "S1": [r"\bb\*b\b", r"square", r"quadrat"],
    "S2": [r"float state", r"no float", r"np\.zeros", r"np\.random"],
    "T1": [r"path time", r"path length", r"integer time", r"discrete time"],
    "T2": [r"observer projection", r"float cast", r"modulus", r"firewall"],
    "A2": [r"derived coord", r"d = b\+e", r"a = b\+2e", r"raw d"],
    "NT": [r"theorem nt", r"observer projection", r"continuous", r"firewall"],
}

# Keyword → Hard Rule.  rule titles are lower-cased, underscored.
RULE_AXIOM_KEYWORDS = {
    # rule title substring → axiom codes
    "stochastic": ["T2", "NT", "S2"],
    "continuous": ["T2", "NT"],
    "raw": ["A2"],
    "mod_reduced": ["A2"],
    "qa_always_applies": ["A1", "A2"],
    "theorem_nt": ["NT", "T2"],
    "observer": ["NT", "T2"],
    "primary_sources": [],  # epistemic, not axiomatic
    "no_arxiv": [],
    "no_secrets": [],
    "adversarial": [],
}

_CERT_REF_RE = re.compile(r"\[(\d+)\]")


def _all_cert_ids(kg: KG) -> list[str]:
    rows = kg.conn.execute(
        "SELECT id FROM nodes WHERE node_type = 'Cert'"
    ).fetchall()
    return [r["id"] for r in rows]


def _all_rule_ids(kg: KG) -> list[tuple[str, str]]:
    rows = kg.conn.execute(
        "SELECT id, title FROM nodes WHERE node_type = 'Rule'"
    ).fetchall()
    return [(r["id"], r["title"]) for r in rows]


def _cert_body(kg: KG, cert_id: str) -> str:
    row = kg.conn.execute("SELECT body, title FROM nodes WHERE id=?", (cert_id,)).fetchone()
    if not row:
        return ""
    return (row["title"] or "") + "\n" + (row["body"] or "")


def extract_cert_axiom_edges(kg: KG) -> int:
    """Emit cert→axiom 'derived-from' edges. Returns edge count."""
    count = 0
    axiom_ids = {code: f"axiom:{code}" for code in ("A1", "A2", "T2", "S1", "S2", "T1", "NT")}
    for cert_id in _all_cert_ids(kg):
        body = _cert_body(kg, cert_id).lower()
        linked: set[str] = set(BASELINE_AXIOMS)
        for code, patterns in AXIOM_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, body, re.I):
                    linked.add(code)
                    break
        for code in linked:
            try:
                kg.upsert_edge(Edge(
                    src_id=cert_id, dst_id=axiom_ids[code],
                    edge_type="derived-from", confidence=0.9,
                    provenance="extractors.edges.cert_axiom_keywords",
                ))
                count += 1
            except Exception:
                continue
    return count


def extract_rule_axiom_edges(kg: KG) -> int:
    """Emit rule→axiom 'instantiates' edges."""
    count = 0
    axiom_ids = {code: f"axiom:{code}" for code in ("A1", "A2", "T2", "S1", "S2", "T1", "NT")}
    for rule_id, title in _all_rule_ids(kg):
        key = rule_id.replace("rule:", "").lower()
        title_key = (title or "").lower()
        linked: set[str] = set()
        for fragment, codes in RULE_AXIOM_KEYWORDS.items():
            if fragment in key or fragment in title_key:
                linked.update(codes)
        for code in linked:
            try:
                kg.upsert_edge(Edge(
                    src_id=rule_id, dst_id=axiom_ids[code],
                    edge_type="instantiates", confidence=0.85,
                    provenance="extractors.edges.rule_axiom_keywords",
                ))
                count += 1
            except Exception:
                continue
    return count


def extract_cert_cross_refs(kg: KG) -> int:
    """Emit cert→cert 'extends' edges where a cert body references [N] for another cert."""
    count = 0
    cert_ids = set(_all_cert_ids(kg))
    for cert_id in cert_ids:
        if not cert_id.startswith("cert:") or cert_id.startswith("cert:fs:"):
            continue
        try:
            this_num = int(cert_id.split(":", 1)[1])
        except ValueError:
            continue
        body = _cert_body(kg, cert_id)
        refs = {int(m) for m in _CERT_REF_RE.findall(body)}
        refs.discard(this_num)
        for ref in refs:
            tgt = f"cert:{ref}"
            if tgt not in cert_ids:
                continue
            try:
                kg.upsert_edge(Edge(
                    src_id=cert_id, dst_id=tgt,
                    edge_type="extends", confidence=0.7,
                    provenance="extractors.edges.cert_cross_refs",
                ))
                count += 1
            except Exception:
                continue
    return count


def populate(kg: KG) -> dict[str, int]:
    return {
        "cert_axiom": extract_cert_axiom_edges(kg),
        "rule_axiom": extract_rule_axiom_edges(kg),
        "cert_cross": extract_cert_cross_refs(kg),
    }
