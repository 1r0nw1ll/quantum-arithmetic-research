"""QA-KG main API.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Schema v3: epistemic fields (authority, epistemic_status, method,
source_locator, lifecycle_state) added per Phase 1. Authority drives the
Theorem NT firewall alongside tier.

Phase 2: kg.promote() + DB-backed agent firewall. Agent causal edges are
blocked at the policy level (edge_allowed returns False unconditionally for
authority=agent). The only bypass is a promoted-from edge in the DB,
queried by upsert_edge at insert time. promote() creates that edge after
validating _meta_ledger.json staleness + broadcast corroboration.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import datetime as _dt
import json
import logging
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tools.qa_kg.orbit import (
    CAUSAL_EDGE_TYPES, Index, Tier, char_ord_sum, compute_index,
    edge_allowed, tier_for_index,
)
from tools.qa_kg.schema import (
    AUTHORITIES, DEFAULT_DB, EPISTEMIC_STATUSES, LIFECYCLE_STATES, init_db,
)

_log = logging.getLogger("qa_kg")

PROMOTED_FROM_EDGE = "promoted-from"
LEDGER_STALENESS_DAYS = 14
BROADCAST_WINDOW_S = 60

_REPO = Path(__file__).resolve().parents[2]
_META_DIR = _REPO / "qa_alphageometry_ptolemy"
_LEDGER_PATH = _META_DIR / "_meta_ledger.json"

_PROMOTER_AUTHORITIES = frozenset({"primary", "derived", "internal"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_git_head_cache: str | None = None


def _current_git_head() -> str:
    """Return current git HEAD sha. Cached per process. Falls back to UNKNOWN."""
    global _git_head_cache
    if _git_head_cache is not None:
        return _git_head_cache
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_REPO),
        )
        _git_head_cache = result.stdout.strip() or "UNKNOWN"
    except (subprocess.SubprocessError, FileNotFoundError):
        _git_head_cache = "UNKNOWN"
    return _git_head_cache


def _load_ledger(path: Path | None = None) -> dict:
    """Load _meta_ledger.json. Returns {} if missing or unparseable."""
    p = path or _LEDGER_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Failed to load ledger at %s: %s", p, exc)
        return {}


@dataclass
class Node:
    """Knowledge graph node.

    Candidate F produces `idx_b/idx_e` (retrieval-index, not QA state).
    `subject_b/subject_e` optionally declare the QA STATE the node is ABOUT
    — populated only from cert metadata, never by Candidate F.

    Phase 1 epistemic fields:
      authority        — who produced this knowledge (primary/derived/internal/agent)
      epistemic_status — what kind of claim (axiom/source_claim/certified/...)
      method           — extraction pathway (cert_validator/ob_capture/...)
      source_locator   — schemed pointer (file:<path>, ob:<id>, cert:<id>)
      lifecycle_state  — current/deprecated/superseded/withdrawn
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
    authority: str | None = None
    epistemic_status: str | None = None
    method: str | None = None
    source_locator: str | None = None
    lifecycle_state: str = "current"

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
    """Theorem NT firewall blocked an unauthorized edge."""


def _validate_node_epistemic(node: Node) -> None:
    """Application-layer CHECK for authority/epistemic_status/lifecycle_state."""
    if node.authority is not None and node.authority not in AUTHORITIES:
        raise ValueError(
            f"authority={node.authority!r} not in {AUTHORITIES}"
        )
    if node.epistemic_status is not None and node.epistemic_status not in EPISTEMIC_STATUSES:
        raise ValueError(
            f"epistemic_status={node.epistemic_status!r} not in {EPISTEMIC_STATUSES}"
        )
    if node.lifecycle_state not in LIFECYCLE_STATES:
        raise ValueError(
            f"lifecycle_state={node.lifecycle_state!r} not in {LIFECYCLE_STATES}"
        )


class KG:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_node(self, node: Node) -> None:
        _validate_node_epistemic(node)
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
                               authority, epistemic_status, method,
                               source_locator, lifecycle_state,
                               created_ts, updated_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                authority=excluded.authority,
                epistemic_status=excluded.epistemic_status,
                method=excluded.method,
                source_locator=excluded.source_locator,
                lifecycle_state=excluded.lifecycle_state,
                updated_ts=excluded.updated_ts
            """,
            (node.id, node.node_type, node.title, node.body, tier.value,
             cb, ce, cos, node.subject_b, node.subject_e,
             node.source, node.vetted_by, node.vetted_ts, node.predicate_ref,
             node.authority, node.epistemic_status, node.method,
             node.source_locator, node.lifecycle_state,
             now, now),
        )
        self.conn.commit()

    def upsert_edge(self, edge: Edge) -> None:
        src = self.conn.execute("SELECT tier, authority FROM nodes WHERE id=?", (edge.src_id,)).fetchone()
        dst = self.conn.execute("SELECT tier, authority FROM nodes WHERE id=?", (edge.dst_id,)).fetchone()
        if src is None or dst is None:
            raise ValueError(f"edge references unknown node: {edge.src_id}→{edge.dst_id}")
        src_tier = Tier(src["tier"])
        dst_tier = Tier(dst["tier"])
        src_authority = src["authority"]
        if not edge_allowed(src_tier, dst_tier, edge.edge_type,
                            bool(edge.via_cert), src_authority=src_authority):
            # Phase 2: DB-backed promoted-from bypass for agent nodes.
            # edge_allowed returns False unconditionally for agent + causal.
            # The only way through is a promoted-from edge in the DB created
            # by kg.promote(). Callers cannot fake this with a via_cert string.
            if (src_authority == "agent"
                    and edge.edge_type in CAUSAL_EDGE_TYPES
                    and self._has_promoted_from(edge.src_id)):
                pass  # promoted agent — allow the causal edge
            else:
                raise FirewallViolation(
                    f"{src_tier.value}(authority={src_authority})→{dst_tier.value} "
                    f"via '{edge.edge_type}' blocked by Theorem NT firewall"
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

    def search(self, query: str, *, tier: str | None = None,
               authority: list[str] | None = None,
               k: int = 10) -> list[sqlite3.Row]:
        base = (
            "SELECT nodes.* FROM nodes_fts "
            "JOIN nodes ON nodes.rowid = nodes_fts.rowid WHERE nodes_fts MATCH ?"
        )
        args: list = [query]
        if tier:
            base += " AND nodes.tier = ?"
            args.append(tier)
        if authority:
            placeholders = ",".join("?" * len(authority))
            base += f" AND nodes.authority IN ({placeholders})"
            args.extend(authority)
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
        """All nodes at a given retrieval-index cell."""
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

        Filters to method != 'keyword'. Under Phase 1 this still returns
        empty for most nodes — real structural provenance is Phase 3 work.
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

    # --- Phase 2: promote protocol ---

    def _has_promoted_from(self, node_id: str) -> bool:
        """DB query: does this node have a promoted-from edge with via_cert?"""
        row = self.conn.execute(
            "SELECT 1 FROM edges WHERE src_id=? AND edge_type=? AND via_cert != ''",
            (node_id, PROMOTED_FROM_EDGE),
        ).fetchone()
        return row is not None

    def promote(
        self,
        agent_note_id: str,
        via_cert: str,
        promoter_node_id: str,
        broadcast_payload: dict,
        *,
        ledger_path: Path | str | None = None,
    ) -> None:
        """Promote an agent note so it can emit causal edges.

        Creates a promoted-from edge from agent_note_id → promoter_node_id
        with provenance containing the broadcast payload snapshot.

        Validates:
          1. agent_note_id exists with authority=agent
          2. promoter_node_id exists with authority ∈ {primary, derived, internal}
          3. via_cert resolves to PASS in _meta_ledger.json within staleness
          4. broadcast_payload timestamp within ±60s of now
        """
        agent_row = self.conn.execute(
            "SELECT authority FROM nodes WHERE id=?", (agent_note_id,)
        ).fetchone()
        if agent_row is None:
            raise ValueError(f"agent_note_id {agent_note_id!r} not found")
        if agent_row["authority"] != "agent":
            raise ValueError(
                f"agent_note_id {agent_note_id!r} has authority="
                f"{agent_row['authority']!r}, expected 'agent'"
            )

        promoter_row = self.conn.execute(
            "SELECT authority FROM nodes WHERE id=?", (promoter_node_id,)
        ).fetchone()
        if promoter_row is None:
            raise ValueError(f"promoter_node_id {promoter_node_id!r} not found")
        if promoter_row["authority"] not in _PROMOTER_AUTHORITIES:
            raise ValueError(
                f"promoter_node_id {promoter_node_id!r} has authority="
                f"{promoter_row['authority']!r}, expected one of "
                f"{sorted(_PROMOTER_AUTHORITIES)}"
            )

        # Ledger staleness check
        lpath = Path(ledger_path) if ledger_path else _LEDGER_PATH
        ledger = _load_ledger(lpath)
        if via_cert not in ledger:
            raise FirewallViolation(
                f"via_cert {via_cert!r} not in ledger at {lpath}"
            )
        entry = ledger[via_cert]
        if entry.get("status") != "PASS":
            raise FirewallViolation(
                f"via_cert {via_cert!r} ledger status={entry.get('status')!r}, "
                f"expected 'PASS'"
            )
        entry_ts = _dt.datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
        age = _dt.datetime.now(_dt.timezone.utc) - entry_ts
        if age.days > LEDGER_STALENESS_DAYS:
            raise FirewallViolation(
                f"via_cert {via_cert!r} ledger stale: {age.days}d > "
                f"{LEDGER_STALENESS_DAYS}d"
            )
        current_head = _current_git_head()
        if current_head != "UNKNOWN" and entry.get("git_head") != current_head:
            raise FirewallViolation(
                f"via_cert {via_cert!r} ledger git_head={entry.get('git_head')!r} "
                f"!= HEAD={current_head!r}"
            )

        # Broadcast corroboration: payload ts within ±60s of now
        bp_ts_raw = broadcast_payload.get("ts", "")
        if not bp_ts_raw:
            raise FirewallViolation(
                "broadcast_payload missing 'ts' field"
            )
        bp_ts = _dt.datetime.fromisoformat(bp_ts_raw.replace("Z", "+00:00"))
        bp_age = abs((_dt.datetime.now(_dt.timezone.utc) - bp_ts).total_seconds())
        if bp_age > BROADCAST_WINDOW_S:
            raise FirewallViolation(
                f"broadcast_payload ts {bp_ts_raw!r} is {bp_age:.0f}s from now, "
                f"exceeds ±{BROADCAST_WINDOW_S}s window"
            )

        # Write promoted-from edge (bypasses edge_allowed — it's not causal)
        provenance = json.dumps({
            "session": broadcast_payload.get("session", ""),
            "signed_ts": _now(),
            "promoter_node_id": promoter_node_id,
            "broadcast_payload_snapshot": broadcast_payload,
        })
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
            (agent_note_id, promoter_node_id, PROMOTED_FROM_EDGE,
             1.0, "promote", provenance, via_cert, _now()),
        )
        self.conn.commit()
        _log.info("promoted %s via %s (promoter=%s)",
                   agent_note_id, via_cert, promoter_node_id)

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
