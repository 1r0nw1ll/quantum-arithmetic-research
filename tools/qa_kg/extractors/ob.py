# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG infrastructure; ingests OB thoughts -->
"""Ingest Open Brain thoughts as Thought-typed nodes.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Phase 0: all edges extracted here use `edge_type="keyword-co-occurs"` with
method="keyword" and confidence=0.3. A mention of `[N]` in a thought body
indicates the thought TEXT names cert N; it is not derivation, not
instantiation, not authoritative provenance.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from tools.qa_kg.kg import KG, Edge, Node


_CERT_REF_RE = re.compile(r"\[(\d+)\]")
_AXIOM_RE = re.compile(r"\b(A1|A2|T2|S1|S2|T1|NT)\b")


@dataclass
class OBThought:
    id: str
    ts: str
    thought_type: str
    tags: list[str]
    body: str


def parse_markdown(text: str) -> list[OBThought]:
    boundary = re.compile(r"(?m)^-\s+\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    starts = [m.start() for m in boundary.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    thoughts: list[OBThought] = []
    for i in range(len(starts) - 1):
        chunk = text[starts[i]:starts[i + 1]].rstrip()
        head_match = re.match(
            r"^-\s+(?P<ts>\S+)\s+\((?P<type>[^)]+)\)\s+(?:\[(?P<tags>[^\]]*)\]\s+)?(?P<first>.*)",
            chunk,
        )
        if not head_match:
            continue
        ts = head_match.group("ts")
        ttype = head_match.group("type").strip()
        tags_raw = head_match.group("tags") or ""
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        first = head_match.group("first") or ""
        rest = chunk[head_match.end():].lstrip("\n")
        body = first
        if rest:
            body = (body + "\n" + rest).strip()
        h = hashlib.sha256((ts + body[:200]).encode("utf-8")).hexdigest()[:16]
        thoughts.append(OBThought(
            id=f"ob:{h}", ts=ts, thought_type=ttype, tags=tags, body=body,
        ))
    return thoughts


def ingest(kg: KG, thoughts: Iterable[OBThought]) -> dict[str, int]:
    cert_ids = {r["id"] for r in kg.conn.execute(
        "SELECT id FROM nodes WHERE node_type='Cert'"
    ).fetchall()}
    rule_idx = [(r["id"], (r["title"] or "").lower()) for r in kg.conn.execute(
        "SELECT id, title FROM nodes WHERE node_type='Rule'"
    ).fetchall()]
    axiom_ids = {c: f"axiom:{c}" for c in ("A1", "A2", "T2", "S1", "S2", "T1", "NT")}

    nodes = 0
    edges = 0
    for t in thoughts:
        title = (t.body.split("\n", 1)[0] or "")[:120]
        tags_str = ",".join(t.tags)
        body_short = t.body[:2000]
        kg.upsert_node(Node(
            id=t.id, node_type="Thought",
            title=title or f"OB {t.ts}",
            body=f"ts: {t.ts}\ntype: {t.thought_type}\ntags: {tags_str}\n---\n{body_short}",
            source=f"open_brain:{t.ts}",
        ))
        nodes += 1

        for n in _CERT_REF_RE.findall(t.body):
            tgt = f"cert:{n}"
            if tgt in cert_ids:
                try:
                    kg.upsert_edge(Edge(
                        src_id=t.id, dst_id=tgt,
                        edge_type="keyword-co-occurs",
                        confidence=0.3, method="keyword",
                        provenance="extractors.ob.cert_ref",
                    )); edges += 1
                except Exception:
                    pass
        for code in set(_AXIOM_RE.findall(t.body)):
            try:
                kg.upsert_edge(Edge(
                    src_id=t.id, dst_id=axiom_ids[code],
                    edge_type="keyword-co-occurs",
                    confidence=0.3, method="keyword",
                    provenance=f"extractors.ob.axiom_code:{code}",
                )); edges += 1
            except Exception:
                pass
        body_lower = t.body.lower()
        for rid, rtitle in rule_idx:
            if rtitle and rtitle in body_lower:
                try:
                    kg.upsert_edge(Edge(
                        src_id=t.id, dst_id=rid,
                        edge_type="keyword-co-occurs",
                        confidence=0.3, method="keyword",
                        provenance="extractors.ob.rule_title_match",
                    )); edges += 1
                except Exception:
                    pass

    return {"nodes": nodes, "edges": edges}


def ingest_markdown(kg: KG, text: str) -> dict[str, int]:
    thoughts = parse_markdown(text)
    stats = ingest(kg, thoughts)
    stats["parsed"] = len(thoughts)
    return stats
