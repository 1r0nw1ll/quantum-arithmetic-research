# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG archive bridge -->
"""A-RAG archive bridge — lazy, firewalled.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

A-RAG already stores Candidate F (b,e) on every message (per [202]). We expose
search results as-is (marked archive) without persistence. Promotion into QA-MEM
re-computes (b,e) via OUR Candidate F which uses our NODE_TYPE_RANK — same
formula family, extended role ranks for knowledge-graph node types.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sqlite3
from pathlib import Path

from tools.qa_kg.kg import KG, Node


ARAG_DB = Path("/home/player2/signal_experiments/_forensics/qa_retrieval.sqlite")


def _arag_conn() -> sqlite3.Connection:
    if not ARAG_DB.exists():
        raise FileNotFoundError(f"A-RAG DB missing at {ARAG_DB}")
    conn = sqlite3.connect(str(ARAG_DB))
    conn.row_factory = sqlite3.Row
    return conn


def search(query: str, *, k: int = 10, source: str | None = None) -> list[dict]:
    conn = _arag_conn()
    base = """
    SELECT m.msg_id, m.source, m.role, m.conv_title, m.create_time_utc,
           m.b, m.e, substr(m.raw_text, 1, 300) AS preview,
           bm25(messages_fts) AS score
    FROM messages_fts
    JOIN messages m ON m.msg_id = messages_fts.msg_id
    WHERE messages_fts MATCH ?
    """
    args: list = [query]
    if source:
        base += " AND m.source = ?"; args.append(source)
    base += " ORDER BY score LIMIT ?"
    args.append(k)
    rows = conn.execute(base, args).fetchall()
    return [{
        "tier": "archive",
        "msg_id": r["msg_id"], "source": r["source"], "role": r["role"],
        "conv_title": r["conv_title"], "ts": r["create_time_utc"],
        "arag_coord": (r["b"], r["e"]),
        "preview": r["preview"], "score": r["score"],
    } for r in rows]


def promote_to_kg(kg: KG, msg_id: str, *, reason: str = "") -> str:
    """Persist an archive chunk as a Thought-typed node.
    Candidate F recomputes (b,e) on the new title+body under Thought rank."""
    conn = _arag_conn()
    row = conn.execute(
        "SELECT msg_id, source, role, conv_title, create_time_utc, raw_text "
        "FROM messages WHERE msg_id = ?", (msg_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"A-RAG msg_id {msg_id!r} not found")
    nid = f"arag:{row['source']}:{msg_id}"
    title = row["conv_title"] or f"{row['source']} {row['role']} {row['create_time_utc']}"
    body = "\n".join([
        f"source: {row['source']}", f"role: {row['role']}",
        f"ts: {row['create_time_utc']}", f"promotion_reason: {reason}",
        "---", (row["raw_text"] or "")[:4000],
    ])
    kg.upsert_node(Node(
        id=nid, node_type="Thought", title=str(title)[:150], body=body,
        source=f"arag:{msg_id}",
        authority="internal",
        epistemic_status="source_claim",
        method="arag_message",
        source_locator=f"ob:{msg_id}",
    ))
    return nid
