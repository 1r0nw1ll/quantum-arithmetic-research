"""Keyword-based edge extraction — low confidence, clearly labeled.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Phase 0: all keyword-matched edges emit `edge_type="keyword-co-occurs"` with
`method="keyword"` and `confidence=0.3`. These edges represent BODY-TOKEN
CO-OCCURRENCE, not derivation, instantiation, or proof.

The previous implementation wrote `derived-from` at 0.9 confidence from a
baseline auto-link ({A1, A2, T2, NT} for every cert) plus keyword matches.
Both produced near-uninformative saturation (the same four axioms reached
from every cert). They are removed here. `derived-from` edges are reserved
for structural proof links, which will be populated by a Phase 3 extractor
that reads actual cert proof artifacts — NOT body text.

Extractors:
  1. cert → axiom / rule → axiom / cert → cert — via `[N]` cert-id refs
     and axiom code tokens in the source's body. All tagged keyword-co-occurs.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import re

from tools.qa_kg.kg import KG, Edge


AXIOM_CODES = ("A1", "A2", "A3", "A4", "T1", "T2", "S1", "S2", "NT")
_AXIOM_RE = re.compile(r"\b(" + "|".join(AXIOM_CODES) + r")\b")
_CERT_REF_RE = re.compile(r"\[(\d+)\]")


def _all_cert_ids(kg: KG) -> set[str]:
    return {r["id"] for r in kg.conn.execute(
        "SELECT id FROM nodes WHERE node_type = 'Cert'"
    ).fetchall()}


def _all_rule_ids(kg: KG) -> list[tuple[str, str]]:
    return [(r["id"], (r["title"] or "").lower())
            for r in kg.conn.execute(
                "SELECT id, title FROM nodes WHERE node_type = 'Rule'"
            ).fetchall()]


def _cert_body(kg: KG, cert_id: str) -> str:
    row = kg.conn.execute(
        "SELECT title, body FROM nodes WHERE id=?", (cert_id,)
    ).fetchone()
    if not row:
        return ""
    return (row["title"] or "") + "\n" + (row["body"] or "")


def extract_cert_axiom_cooccurrences(kg: KG) -> int:
    """Emit keyword-co-occurs edges cert→axiom ONLY when an axiom code
    literally appears in the cert's body. No baseline auto-link."""
    axiom_ids = {c: f"axiom:{c}" for c in AXIOM_CODES}
    count = 0
    for cert_id in _all_cert_ids(kg):
        body = _cert_body(kg, cert_id)
        for code in set(_AXIOM_RE.findall(body)):
            try:
                kg.upsert_edge(Edge(
                    src_id=cert_id, dst_id=axiom_ids[code],
                    edge_type="keyword-co-occurs",
                    confidence=0.3,
                    method="keyword",
                    provenance=f"extractors.edges.cert_axiom:{code}",
                ))
                count += 1
            except Exception:
                continue
    return count


def extract_rule_axiom_cooccurrences(kg: KG) -> int:
    axiom_ids = {c: f"axiom:{c}" for c in AXIOM_CODES}
    count = 0
    for rid, _title in _all_rule_ids(kg):
        row = kg.conn.execute("SELECT title, body FROM nodes WHERE id=?", (rid,)).fetchone()
        text = (row["title"] or "") + "\n" + (row["body"] or "")
        for code in set(_AXIOM_RE.findall(text)):
            try:
                kg.upsert_edge(Edge(
                    src_id=rid, dst_id=axiom_ids[code],
                    edge_type="keyword-co-occurs",
                    confidence=0.3,
                    method="keyword",
                    provenance=f"extractors.edges.rule_axiom:{code}",
                ))
                count += 1
            except Exception:
                continue
    return count


def extract_cert_cross_refs(kg: KG) -> int:
    """`[N]` references in a cert's body emit keyword-co-occurs cert→cert.
    This is co-occurrence only — the citation may mean extends, cites, or
    coincidence; we do not claim to know which."""
    count = 0
    cert_ids = set()
    for r in kg.conn.execute("SELECT id FROM nodes WHERE node_type='Cert'").fetchall():
        cert_ids.add(r["id"])
    for cert_id in cert_ids:
        if cert_id.startswith("cert:fs:"):
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
                    edge_type="keyword-co-occurs",
                    confidence=0.3,
                    method="keyword",
                    provenance="extractors.edges.cert_cross",
                ))
                count += 1
            except Exception:
                continue
    return count


def populate(kg: KG) -> dict[str, int]:
    return {
        "cert_axiom": extract_cert_axiom_cooccurrences(kg),
        "rule_axiom": extract_rule_axiom_cooccurrences(kg),
        "cert_cross": extract_cert_cross_refs(kg),
    }
