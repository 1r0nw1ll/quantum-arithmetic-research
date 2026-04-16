"""QA-KG main API.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Schema v2: column names `idx_b/idx_e` reflect that these are Candidate F
retrieval-index coordinates (not QA states). QA state subjects live in
`subject_b/subject_e`, populated only from cert metadata.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tools.qa_kg.orbit import (
    Index, Tier, char_ord_sum, compute_index, edge_allowed, tier_for_index,
)
from tools.qa_kg.schema import DEFAULT_DB, init_db


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Node:
    """Knowledge graph node.

    Candidate F produces `idx_b/idx_e` (retrieval-index, not QA state).
    `subject_b/subject_e` optionally declare the QA STATE the node is ABOUT
    — populated only from cert metadata, never by Candidate F.
    """
    id: str
    node_type: str
    title: str
    body: str = ""
    source: str = ""
    vetted_by: str = ""
    vetted_ts: str = ""
    predicate_ref: str = ""
    subject_b: int | None = None
    subject_e: int | None = None

    def content(self) -> str:
        return (self.title or "") + ("\n" + self.body if self.body else "")

    def resolved_index(self) -> Index | None:
        text = self.content()
        if not text:
            return None
        return compute_index(text, self.node_type)

    def resolved_tier(self) -> Tier:
        idx = self.resolved_index()
        if idx is None:
            return Tier.UNASSIGNED
        return tier_for_index(idx.idx_b, idx.idx_e)


@dataclass
class Edge:
    src_id: str
    dst_id: str
    edge_type: str
    confidence: float = 1.0
    method: str = ""            # 'keyword' | 'cert_registry' | 'structural' | ''
    provenance: str = ""
    via_cert: str = ""


class FirewallViolation(RuntimeError):
    """Unassigned → canonical causal edge attempted without via_cert."""


class KG:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_node(self, node: Node) -> None:
        idx = node.resolved_index()
        tier = node.resolved_tier()
        cb = idx.idx_b if idx else None
        ce = idx.idx_e if idx else None
        cos = char_ord_sum(node.content()) if node.content() else None
        now = _now()
        self.conn.execute(
            """
            INSERT INTO nodes (id, node_type, title, body, tier, idx_b, idx_e,
                               char_ord_sum, subject_b, subject_e,
                               source, vetted_by, vetted_ts, predicate_ref,
                               created_ts, updated_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                node_type=excluded.node_type,
                title=excluded.title,
                body=excluded.body,
                tier=excluded.tier,
                idx_b=excluded.idx_b,
                idx_e=excluded.idx_e,
                char_ord_sum=excluded.char_ord_sum,
                subject_b=excluded.subject_b,
                subject_e=excluded.subject_e,
                source=excluded.source,
                vetted_by=excluded.vetted_by,
                vetted_ts=excluded.vetted_ts,
                predicate_ref=excluded.predicate_ref,
                updated_ts=excluded.updated_ts
            """,
            (node.id, node.node_type, node.title, node.body, tier.value,
             cb, ce, cos, node.subject_b, node.subject_e,
             node.source, node.vetted_by, node.vetted_ts, node.predicate_ref,
             now, now),
        )
        self.conn.commit()

    def upsert_edge(self, edge: Edge) -> None:
        src = self.conn.execute("SELECT tier FROM nodes WHERE id=?", (edge.src_id,)).fetchone()
        dst = self.conn.execute("SELECT tier FROM nodes WHERE id=?", (edge.dst_id,)).fetchone()
        if src is None or dst is None:
            raise ValueError(f"edge references unknown node: {edge.src_id}→{edge.dst_id}")
        src_tier = Tier(src["tier"])
        dst_tier = Tier(dst["tier"])
        if not edge_allowed(src_tier, dst_tier, edge.edge_type, bool(edge.via_cert)):
            raise FirewallViolation(
                f"{src_tier.value}→{dst_tier.value} via '{edge.edge_type}' requires via_cert"
            )
        self.conn.execute(
            """
            INSERT INTO edges (src_id, dst_id, edge_type, confidence, method,
                               provenance, via_cert, created_ts)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(src_id, dst_id, edge_type) DO UPDATE SET
                confidence=excluded.confidence,
                method=excluded.method,
                provenance=excluded.provenance,
                via_cert=excluded.via_cert
            """,
            (edge.src_id, edge.dst_id, edge.edge_type, edge.confidence,
             edge.method, edge.provenance, edge.via_cert, _now()),
        )
        self.conn.commit()

    def get(self, node_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()

    def search(self, query: str, *, tier: str | None = None, k: int = 10) -> list[sqlite3.Row]:
        base = (
            "SELECT nodes.* FROM nodes_fts "
            "JOIN nodes ON nodes.rowid = nodes_fts.rowid WHERE nodes_fts MATCH ?"
        )
        args: list = [query]
        if tier:
            base += " AND nodes.tier = ?"
            args.append(tier)
        base += " ORDER BY bm25(nodes_fts) LIMIT ?"
        args.append(k)
        return list(self.conn.execute(base, args).fetchall())

    def neighbors(self, node_id: str, *, edge_types: Iterable[str] | None = None,
                  direction: str = "both") -> list[sqlite3.Row]:
        clauses: list[str] = []
        args: list = []
        if direction in ("out", "both"):
            clauses.append("(src_id = ?)"); args.append(node_id)
        if direction in ("in", "both"):
            clauses.append("(dst_id = ?)"); args.append(node_id)
        where = " OR ".join(clauses)
        if edge_types:
            types = list(edge_types)
            where = f"({where}) AND edge_type IN ({','.join('?' * len(types))})"
            args.extend(types)
        return list(self.conn.execute(f"SELECT * FROM edges WHERE {where}", args).fetchall())

    def idx_neighborhood(self, idx: Index, tier: str | None = None) -> list[sqlite3.Row]:
        """All nodes at a given retrieval-index cell.

        NOTE: same retrieval-index does NOT imply semantic similarity.
        Candidate F is uniform by construction. Use edges for structure.
        """
        q = "SELECT * FROM nodes WHERE idx_b=? AND idx_e=?"
        args: list = [idx.idx_b, idx.idx_e]
        if tier:
            q += " AND tier=?"; args.append(tier)
        return list(self.conn.execute(q, args).fetchall())

    def idx_b_column(self, idx_b: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM nodes WHERE idx_b = ? ORDER BY idx_e, id", (idx_b,)
        ).fetchall())

    def idx_e_row(self, idx_e: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM nodes WHERE idx_e = ? ORDER BY idx_b, id", (idx_e,)
        ).fetchall())

    def why(self, node_id: str, *, max_depth: int = 5) -> list[sqlite3.Row]:
        """Provenance chain via structural edges only.

        Filters to edge_type ∈ {validates, derived-from, extends, instantiates}
        AND method ≠ 'keyword'. Under Phase 0 this means keyword-regex edges
        are excluded from provenance — chains only traverse real structural
        links. Empty results here are honest: we have no authoritative proof
        graph yet (Phase 3 work).
        """
        q = """
        WITH RECURSIVE chain(src, dst, edge_type, method, depth) AS (
            SELECT src_id, dst_id, edge_type, method, 0 FROM edges
              WHERE src_id = ?
                AND edge_type IN ('validates','derived-from','extends','instantiates')
                AND method != 'keyword'
            UNION ALL
            SELECT e.src_id, e.dst_id, e.edge_type, e.method, chain.depth + 1
              FROM edges e
              JOIN chain ON e.src_id = chain.dst
              WHERE chain.depth < ?
                AND e.edge_type IN ('validates','derived-from','extends','instantiates')
                AND e.method != 'keyword'
        )
        SELECT * FROM chain
        """
        return list(self.conn.execute(q, (node_id, max_depth)).fetchall())

    def contradictions(self, node_id: str) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM edges WHERE edge_type='contradicts' AND (src_id=? OR dst_id=?)",
            (node_id, node_id),
        ).fetchall())

    def digest(self, *, tier: str = "cosmos", limit: int = 40) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            """
            SELECT nodes.*,
                   COALESCE((SELECT COUNT(*) FROM query_log ql WHERE ql.node_id = nodes.id), 0) AS hits
            FROM nodes WHERE tier = ?
            ORDER BY hits DESC, updated_ts DESC LIMIT ?
            """, (tier, limit),
        ).fetchall())

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT tier, COUNT(*) AS n FROM nodes GROUP BY tier"
        ).fetchall()
        out = {r["tier"]: r["n"] for r in rows}
        out["edges"] = self.conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"]
        return out


def connect(db_path: Path | str | None = None) -> KG:
    return KG(init_db(db_path if db_path else DEFAULT_DB))
