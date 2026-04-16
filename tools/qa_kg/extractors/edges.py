"""Keyword-based edge extraction — low confidence, clearly labeled.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Phase 0+: all keyword-matched edges emit `edge_type="keyword-co-occurs"` with
`method="keyword"` and `confidence=0.3`. These edges represent BODY-TOKEN
CO-OCCURRENCE, not derivation, instantiation, or proof.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import logging
import re

from tools.qa_kg.kg import KG, Edge, FirewallViolation
from tools.qa_kg.extractors.axioms import CANONICAL_AXIOM_CODES


_log = logging.getLogger("qa_kg.extractors.edges")

AXIOM_CODES = CANONICAL_AXIOM_CODES
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


_fw_violations = 0


def _try_edge(kg: KG, edge: Edge) -> bool:
    global _fw_violations
    try:
        kg.upsert_edge(edge)
        return True
    except FirewallViolation as exc:
        _fw_violations += 1
        _log.warning("FirewallViolation in edge extraction: %s", exc)
        return False
    except ValueError as exc:
        _log.debug("Edge skipped (unknown node): %s", exc)
        return False


def extract_cert_axiom_cooccurrences(kg: KG) -> int:
    axiom_ids = {c: f"axiom:{c}" for c in AXIOM_CODES}
    count = 0
    for cert_id in _all_cert_ids(kg):
        body = _cert_body(kg, cert_id)
        for code in set(_AXIOM_RE.findall(body)):
            if _try_edge(kg, Edge(
                src_id=cert_id, dst_id=axiom_ids[code],
                edge_type="keyword-co-occurs",
                confidence=0.3, method="keyword",
                provenance=f"extractors.edges.cert_axiom:{code}",
            )):
                count += 1
    return count


def extract_rule_axiom_cooccurrences(kg: KG) -> int:
    axiom_ids = {c: f"axiom:{c}" for c in AXIOM_CODES}
    count = 0
    for rid, _title in _all_rule_ids(kg):
        row = kg.conn.execute("SELECT title, body FROM nodes WHERE id=?", (rid,)).fetchone()
        text = (row["title"] or "") + "\n" + (row["body"] or "")
        for code in set(_AXIOM_RE.findall(text)):
            if _try_edge(kg, Edge(
                src_id=rid, dst_id=axiom_ids[code],
                edge_type="keyword-co-occurs",
                confidence=0.3, method="keyword",
                provenance=f"extractors.edges.rule_axiom:{code}",
            )):
                count += 1
    return count


def extract_cert_cross_refs(kg: KG) -> int:
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
            if _try_edge(kg, Edge(
                src_id=cert_id, dst_id=tgt,
                edge_type="keyword-co-occurs",
                confidence=0.3, method="keyword",
                provenance="extractors.edges.cert_cross",
            )):
                count += 1
    return count


def firewall_violation_count() -> int:
    return _fw_violations


def populate(kg: KG) -> dict[str, int]:
    global _fw_violations
    _fw_violations = 0
    return {
        "cert_axiom": extract_cert_axiom_cooccurrences(kg),
        "rule_axiom": extract_rule_axiom_cooccurrences(kg),
        "cert_cross": extract_cert_cross_refs(kg),
        "firewall_violations": _fw_violations,
    }
