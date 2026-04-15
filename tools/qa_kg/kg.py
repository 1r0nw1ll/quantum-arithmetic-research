"""QA-KG main API: upsert, search, traverse, digest.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tools.qa_kg.orbit import Coord, Tier, coord_for, edge_allowed
from tools.qa_kg.schema import DEFAULT_DB, init_db


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Node:
    id: str
    node_type: str
    title: str
    body: str = ""
    tier: Tier = Tier.SATELLITE
    coord: Coord | None = None
    source: str = ""
    vetted_by: str = ""
    predicate_ref: str = ""

    def resolved_coord(self) -> Coord:
        if self.coord is not None:
            return self.coord
        if self.tier is Tier.SINGULARITY:
            return Coord(9, 9)
        return coord_for(self.id)


@dataclass
class Edge:
    src_id: str
    dst_id: str
    edge_type: str
    confidence: float = 1.0
    provenance: str = ""
    via_cert: str = ""


class FirewallViolation(RuntimeError):
    """Raised when an edge violates Theorem NT structural firewall."""


class KG:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- writes -------------------------------------------------------------

    def upsert_node(self, node: Node) -> None:
        coord = node.resolved_coord()
        now = _now()
        self.conn.execute(
            """
            INSERT INTO nodes (id, node_type, title, body, tier, coord_b, coord_e,
                               source, vetted_by, predicate_ref, created_ts, updated_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                node_type=excluded.node_type,
                title=excluded.title,
                body=excluded.body,
                tier=excluded.tier,
                coord_b=excluded.coord_b,
                coord_e=excluded.coord_e,
                source=excluded.source,
                vetted_by=excluded.vetted_by,
                predicate_ref=excluded.predicate_ref,
                updated_ts=excluded.updated_ts
            """,
            (node.id, node.node_type, node.title, node.body, node.tier.value,
             coord.b, coord.e, node.source, node.vetted_by, node.predicate_ref,
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
                f"Theorem NT: {src_tier.value}→{dst_tier.value} via '{edge.edge_type}' "
                f"requires via_cert (archive→canonical without cert mediation forbidden)"
            )
        self.conn.execute(
            """
            INSERT INTO edges (src_id, dst_id, edge_type, confidence, provenance, via_cert, created_ts)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(src_id, dst_id, edge_type) DO UPDATE SET
                confidence=excluded.confidence,
                provenance=excluded.provenance,
                via_cert=excluded.via_cert
            """,
            (edge.src_id, edge.dst_id, edge.edge_type, edge.confidence,
             edge.provenance, edge.via_cert, _now()),
        )
        self.conn.commit()

    # ---- reads --------------------------------------------------------------

    def get(self, node_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()

    def search(self, query: str, *, tier: str | None = None, k: int = 10) -> list[sqlite3.Row]:
        base = (
            "SELECT nodes.* FROM nodes_fts "
            "JOIN nodes ON nodes.rowid = nodes_fts.rowid "
            "WHERE nodes_fts MATCH ?"
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
            c = "(src_id = ?)"
            args.append(node_id)
            clauses.append(c)
        if direction in ("in", "both"):
            c = "(dst_id = ?)"
            args.append(node_id)
            clauses.append(c)
        where = " OR ".join(clauses)
        if edge_types:
            types = list(edge_types)
            where = f"({where}) AND edge_type IN ({','.join('?' * len(types))})"
            args.extend(types)
        q = f"SELECT * FROM edges WHERE {where}"
        return list(self.conn.execute(q, args).fetchall())

    def coord_neighborhood(self, coord: Coord, tier: str | None = None) -> list[sqlite3.Row]:
        """All nodes sharing this (b,e) coord — structural connection, no search needed."""
        q = "SELECT * FROM nodes WHERE coord_b=? AND coord_e=?"
        args: list = [coord.b, coord.e]
        if tier:
            q += " AND tier=?"
            args.append(tier)
        return list(self.conn.execute(q, args).fetchall())

    def why(self, node_id: str, *, max_depth: int = 5) -> list[sqlite3.Row]:
        """Provenance chain — walk 'validates'/'derived-from'/'extends' edges toward Singularity."""
        q = """
        WITH RECURSIVE chain(src, dst, edge_type, depth) AS (
            SELECT src_id, dst_id, edge_type, 0 FROM edges
              WHERE src_id = ? AND edge_type IN ('validates','derived-from','extends','instantiates')
            UNION ALL
            SELECT e.src_id, e.dst_id, e.edge_type, chain.depth + 1 FROM edges e
              JOIN chain ON e.src_id = chain.dst
              WHERE chain.depth < ? AND e.edge_type IN ('validates','derived-from','extends','instantiates')
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
        """Session-start digest — hot nodes in a tier, ranked by recency + query telemetry."""
        q = """
        SELECT nodes.*,
               COALESCE((SELECT COUNT(*) FROM query_log ql WHERE ql.node_id = nodes.id), 0) AS hits
        FROM nodes
        WHERE tier = ?
        ORDER BY hits DESC, updated_ts DESC
        LIMIT ?
        """
        return list(self.conn.execute(q, (tier, limit)).fetchall())

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT tier, COUNT(*) AS n FROM nodes GROUP BY tier"
        ).fetchall()
        out = {r["tier"]: r["n"] for r in rows}
        out["edges"] = self.conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"]
        return out


def connect(db_path: Path | str | None = None) -> KG:
    return KG(init_db(db_path if db_path else DEFAULT_DB))
