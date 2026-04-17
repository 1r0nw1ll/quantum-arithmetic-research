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
from tools.qa_kg.ranker import (
    RankedHit, RankerSpec, compose_score, load_spec, normalize_bm25,
)

_log = logging.getLogger("qa_kg")

PROMOTED_FROM_EDGE = "promoted-from"
LEDGER_STALENESS_DAYS = 14
BROADCAST_WINDOW_S = 60

# Phase 5: the ledger entry for cert [228] QA_KG_DETERMINISM_CERT.v1
# carries an optional `graph_hash` field. When present, kg.promote()
# enforces it HARD — live canonical graph_hash must match. When absent
# (pre-Phase-5 ledger), the existing Phase 2 timestamp+git_head checks
# alone govern staleness. Additive, backward compatible.
DETERMINISM_CERT_LEDGER_KEY = "228"

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

    Phase 4 ranker-input fields (defaults absorb existing extractors):
      confidence  — measured signal in [0,1]; default 1.0. Set explicitly
                    by extractors that have actual measurement uncertainty
                    (e.g., source_claims with extraction_method='ocr').
                    DO NOT default by authority — that double-counts the
                    authority_weight in the ranker formula (see plan M1).
      valid_from  — ISO-8601 timestamp the claim becomes valid. Empty
                    string means "no explicit start date"; ranker falls
                    back to created_ts for time_decay anchor.
      valid_until — ISO-8601 timestamp the claim expires. Empty = no
                    expiry. Cert [254] R4 filters when populated.
      domain      — optional domain tag (Phase 4.5 will populate; ranker
                    accepts None for "no filter").
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
    confidence: float = 1.0
    valid_from: str = ""
    valid_until: str = ""
    domain: str = ""

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

    # --- Phase 3 factories: pin authority/epistemic_status combinations ---

    @classmethod
    def source_work(
        cls,
        *,
        work_id: str,
        title: str,
        source_locator: str,
        body: str = "",
        extraction_method: str = "manual",
    ) -> "Node":
        """Factory for a primary-source container (book / paper / wiki page).

        Pins authority='primary', epistemic_status='source_work',
        node_type='Work'. Extractors should prefer this over constructing
        Node directly — it guarantees the SourceWork invariants that
        [225] v4 KG11 and [253] SC2 check at cert time.
        """
        return cls(
            id=f"work:{work_id}",
            node_type="Work",
            title=title,
            body=body,
            authority="primary",
            epistemic_status="source_work",
            method=extraction_method,
            source_locator=source_locator,
        )

    @classmethod
    def source_claim(
        cls,
        *,
        claim_id: str,
        quote: str,
        source_locator: str,
        extraction_method: str,
        title: str | None = None,
    ) -> "Node":
        """Factory for a verbatim quote from a primary source.

        Pins authority='primary', epistemic_status='source_claim',
        node_type='Claim'. `body` carries the verbatim quote (this is
        what [253] SC1 checks for non-empty). extraction_method must be
        in {manual, ocr, llm, script} per [253] SC3.
        """
        if not quote:
            raise ValueError("source_claim requires non-empty quote")
        if extraction_method not in ("manual", "ocr", "llm", "script"):
            raise ValueError(
                f"extraction_method={extraction_method!r} not in "
                f"{{manual, ocr, llm, script}}"
            )
        return cls(
            id=f"sc:{claim_id}",
            node_type="Claim",
            title=title or f"SourceClaim {claim_id}",
            body=quote,
            authority="primary",
            epistemic_status="source_claim",
            method=extraction_method,
            source_locator=source_locator,
        )


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


def _validate_node_fields(node: Node) -> None:
    """Application-layer CHECK for Phase 1 + Phase 4 enum / range fields.

    SQLite cannot retroactively enforce CHECK constraints added via ALTER
    TABLE ADD COLUMN, so Phase 1 (authority/epistemic_status/lifecycle)
    and Phase 4 (confidence range) all live here as the runtime safety net
    for any DB created before the matching schema version.
    """
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
    if not (0.0 <= node.confidence <= 1.0):
        raise ValueError(
            f"confidence={node.confidence!r} not in [0.0, 1.0]"
        )


class KG:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert_node(self, node: Node) -> None:
        _validate_node_fields(node)
        # Phase 6 W5 — authority is immutable once set. Blocks both directions:
        # accidental upgrade (agent→internal/derived/primary) AND silent
        # downgrade (primary→agent from a buggy extractor re-run). Real
        # authority corrections require explicit delete+recreate so the
        # change is auditable, not a silent overwrite.
        if node.authority:
            existing = self.conn.execute(
                "SELECT authority FROM nodes WHERE id=?", (node.id,),
            ).fetchone()
            if (existing is not None
                    and existing["authority"]
                    and existing["authority"] != node.authority):
                raise FirewallViolation(
                    f"authority_immutable: {existing['authority']}→"
                    f"{node.authority} on {node.id}. Authority changes "
                    f"require explicit delete+recreate, not silent overwrite."
                )
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
                               confidence, valid_from, valid_until, domain,
                               created_ts, updated_ts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                confidence=excluded.confidence,
                valid_from=excluded.valid_from,
                valid_until=excluded.valid_until,
                domain=excluded.domain,
                updated_ts=excluded.updated_ts
            """,
            (node.id, node.node_type, node.title, node.body, tier.value,
             cb, ce, cos, node.subject_b, node.subject_e,
             node.source, node.vetted_by, node.vetted_ts, node.predicate_ref,
             node.authority, node.epistemic_status, node.method,
             node.source_locator, node.lifecycle_state,
             node.confidence, node.valid_from, node.valid_until, node.domain,
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
        """Low-level FTS5/BM25 wrapper. NO authority awareness.

        DO NOT use for agent-facing queries — agents must call
        search_authority_ranked() (Phase 4 cert [254] enforces formula
        correctness; Phase 6 cert [255] will enforce the MCP boundary).

        This entry point stays available for cert validators that want raw
        FTS5 introspection (e.g., [225]/[252]/[253] consistency checks).
        """
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

    # ---------------------------------------------------------------------
    # Phase 4: authority-tiered retrieval ranker
    # ---------------------------------------------------------------------

    _MIN_AUTHORITY_LADDER: tuple[tuple[str, frozenset[str]], ...] = (
        ("primary",  frozenset({"primary"})),
        ("derived",  frozenset({"primary", "derived"})),
        ("internal", frozenset({"primary", "derived", "internal"})),
        ("agent",    frozenset({"primary", "derived", "internal", "agent"})),
    )

    @staticmethod
    def _resolve_min_authority(min_authority: str) -> frozenset[str]:
        ladder = dict(KG._MIN_AUTHORITY_LADDER)
        if min_authority not in ladder:
            raise ValueError(
                f"min_authority={min_authority!r} not in {sorted(ladder)}"
            )
        return ladder[min_authority]

    def _candidate_pool(
        self,
        query: str,
        allowed_authorities: frozenset[str],
        domain: str | None,
        valid_at_iso: str | None,
        k_pool: int,
    ) -> list[tuple[sqlite3.Row, float]]:
        """Two-pass candidate selection (plan M2).

        Pass A — ALL FTS5 matches with authority IN {primary, derived} ∩
        allowed_authorities. No BM25 cap; primary material is never
        silently demoted by low BM25.

        Pass B — top-(k_pool - len(A)) FTS5 matches with authority IN
        allowed_authorities \\ {primary, derived}, ordered by raw BM25.
        Skipped entirely when Pass A already meets or exceeds k_pool.

        Filters applied to BOTH passes:
          - lifecycle_state != 'withdrawn'
          - domain = ? (when domain is not None and not '')
          - valid_until = '' OR valid_until >= valid_at_iso (when given)
        """
        primary_set = frozenset({"primary", "derived"}) & allowed_authorities
        secondary_set = allowed_authorities - primary_set

        def _build(authorities: frozenset[str], limit: int | None) -> list[tuple[sqlite3.Row, float]]:
            if not authorities:
                return []
            placeholders = ",".join("?" * len(authorities))
            sql = (
                "SELECT nodes.*, bm25(nodes_fts) AS _bm25_raw "
                "FROM nodes_fts "
                "JOIN nodes ON nodes.rowid = nodes_fts.rowid "
                "WHERE nodes_fts MATCH ? "
                f"  AND nodes.authority IN ({placeholders}) "
                "  AND nodes.lifecycle_state != 'withdrawn' "
            )
            args: list = [query, *sorted(authorities)]
            if domain:
                sql += " AND nodes.domain = ? "
                args.append(domain)
            if valid_at_iso:
                sql += " AND (nodes.valid_until = '' OR nodes.valid_until >= ?) "
                args.append(valid_at_iso)
            sql += " ORDER BY bm25(nodes_fts) "
            if limit is not None:
                sql += " LIMIT ? "
                args.append(limit)
            rows = list(self.conn.execute(sql, args).fetchall())
            return [(r, float(r["_bm25_raw"])) for r in rows]

        pass_a = _build(primary_set, limit=None)
        if len(pass_a) >= k_pool:
            if len(pass_a) > 200:
                _log.warning(
                    "candidate_pool Pass A returned %d primary/derived matches "
                    "for query=%r — consider tightening the query or "
                    "raising candidate_pool_k", len(pass_a), query
                )
            return pass_a
        remaining = k_pool - len(pass_a)
        pass_b = _build(secondary_set, limit=remaining)
        # Dedup by id in case a node somehow surfaces in both (shouldn't,
        # but cheap insurance against future authority-set overlaps).
        seen: set[str] = {row["id"] for row, _ in pass_a}
        merged = list(pass_a)
        for row, score in pass_b:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            merged.append((row, score))
        return merged

    def _contradiction_states(self, node_ids: list[str]) -> dict[str, str]:
        """Map each id → 'none'|'src'|'dst'|'both' based on contradicts edges."""
        if not node_ids:
            return {}
        state: dict[str, str] = {nid: "none" for nid in node_ids}
        placeholders = ",".join("?" * len(node_ids))
        rows = self.conn.execute(
            f"SELECT src_id, dst_id FROM edges WHERE edge_type='contradicts' "
            f"  AND (src_id IN ({placeholders}) OR dst_id IN ({placeholders}))",
            (*node_ids, *node_ids),
        ).fetchall()
        for r in rows:
            sid, did = r["src_id"], r["dst_id"]
            if sid in state:
                state[sid] = "both" if state[sid] == "dst" else "src"
            if did in state:
                state[did] = "both" if state[did] == "src" else "dst"
        return state

    _PROVENANCE_EDGE_TYPES = (
        "validates", "derived-from", "extends", "instantiates",
    )

    def _provenance_depth_to_axiom(
        self, node_id: str, max_depth: int = 5,
    ) -> int:
        """Min hops from node_id to a node with epistemic_status='axiom'.

        Recursive CTE over structural+non-keyword edges only (mirrors
        kg.why()). Returns -1 if no path within max_depth. Plan D2 uses
        live CTE; threshold for materialization to a column is documented
        in docs/specs/QA_MEM_SCOPE.md (nodes > 5,000 OR p95 > 200ms).
        """
        edge_types = self._PROVENANCE_EDGE_TYPES
        placeholders = ",".join("?" * len(edge_types))
        sql = f"""
        WITH RECURSIVE chain(target, depth) AS (
            SELECT dst_id, 1 FROM edges
              WHERE src_id = ?
                AND edge_type IN ({placeholders})
                AND method != 'keyword'
            UNION
            SELECT e.dst_id, c.depth + 1 FROM edges e
              JOIN chain c ON e.src_id = c.target
              WHERE c.depth < ?
                AND e.edge_type IN ({placeholders})
                AND e.method != 'keyword'
        )
        SELECT MIN(c.depth) AS d
          FROM chain c
          JOIN nodes n ON n.id = c.target
         WHERE n.epistemic_status = 'axiom'
        """
        args = [node_id, *edge_types, max_depth, *edge_types]
        # If the node IS an axiom, depth is 0 (self-rooted).
        self_row = self.conn.execute(
            "SELECT epistemic_status FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if self_row is not None and self_row["epistemic_status"] == "axiom":
            return 0
        row = self.conn.execute(sql, args).fetchone()
        if row is None or row["d"] is None:
            return -1
        return int(row["d"])

    def search_authority_ranked(
        self,
        query: str,
        *,
        min_authority: str = "internal",
        domain: str | None = None,
        valid_at: _dt.datetime | None = None,
        k: int = 10,
        spec: RankerSpec | None = None,
    ) -> list[RankedHit]:
        """Authority-tiered retrieval. Phase 4 cert [254].

        Pipeline (see plan §Architecture):
          1. Resolve min_authority to allowed set.
          2. Two-pass candidate pool (plan M2).
          3. Contradiction state lookup.
          4. Provenance depth per candidate.
          5. BM25 normalize across the union.
          6. Compose scores via ranker.compose_score.
          7. Sort + tiebreak (deterministic).
          8. Trim to k.

        Contradiction surfacing is unconditional (plan M3) — there is no
        `include_contradictions` parameter on the public API. For clean
        export use the internal `_export_clean_subset` helper.
        """
        spec = spec if spec is not None else load_spec()
        allowed = self._resolve_min_authority(min_authority)
        # Snapshot wall-clock once per call when valid_at is unspecified, so
        # all candidates in a single call see the same "now" and the
        # composed scores are internally consistent. Cross-call determinism
        # under valid_at=None depends on the wall clock; cert [254] R7
        # exercises determinism by passing an explicit valid_at.
        effective_valid_at = (
            valid_at if valid_at is not None
            else _dt.datetime.now(_dt.timezone.utc)
        )
        valid_at_iso = (
            valid_at.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if valid_at is not None else None
        )

        pool = self._candidate_pool(
            query, allowed, domain, valid_at_iso, spec.candidate_pool_k,
        )
        if not pool:
            return []

        node_ids = [row["id"] for row, _ in pool]
        contradiction_map = self._contradiction_states(node_ids)
        depth_map = {nid: self._provenance_depth_to_axiom(nid) for nid in node_ids}
        bm_norms = normalize_bm25([raw for _, raw in pool])

        hits: list[RankedHit] = []
        for (row, _raw_bm25), bm_norm in zip(pool, bm_norms):
            authority = row["authority"]
            if authority is None:
                # Defense-in-depth: this should never happen because the
                # candidate-pool query filters on authority IN allowed.
                continue
            score, breakdown = compose_score(
                authority=authority,
                bm25_norm=bm_norm,
                confidence=float(row["confidence"]),
                epistemic_status=row["epistemic_status"],
                created_ts=row["created_ts"] or "",
                valid_from=row["valid_from"] or "",
                valid_at=effective_valid_at,
                contradiction_state=contradiction_map[row["id"]],
                provenance_depth=depth_map[row["id"]],
                lifecycle_state=row["lifecycle_state"] or "current",
                spec=spec,
            )
            hits.append(RankedHit(
                node=row,
                score=score,
                authority=authority,
                contradiction_state=contradiction_map[row["id"]],
                provenance_depth=depth_map[row["id"]],
                score_breakdown=breakdown,
            ))

        # Deterministic tiebreak: score DESC, authority_weight DESC, id ASC.
        def _sort_key(h: RankedHit) -> tuple[float, float, str]:
            aw = spec.authority_weight.get(h.authority, 0.0)
            # Negate so default ascending sort = descending value.
            return (-h.score, -aw, h.node["id"])

        hits.sort(key=_sort_key)
        return hits[:k]

    def _export_clean_subset(
        self,
        query: str,
        *,
        min_authority: str = "internal",
        domain: str | None = None,
        valid_at: _dt.datetime | None = None,
        k: int = 10,
        spec: RankerSpec | None = None,
    ) -> list[RankedHit]:
        """INTERNAL — strips contradicted hits for documentation export.

        DO NOT use for agent queries. Surfacing contradictions is a Phase 4
        design intent (cert [254] R3); agent-facing callers MUST use
        search_authority_ranked.
        """
        hits = self.search_authority_ranked(
            query, min_authority=min_authority, domain=domain,
            valid_at=valid_at, k=k * 2, spec=spec,
        )
        clean = [h for h in hits if h.contradiction_state == "none"]
        return clean[:k]

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
        mcp_session: str | None = None,
        agent_writes_path: Path | str | None = None,
    ) -> None:
        """Promote an agent note so it can emit causal edges.

        Creates a promoted-from edge from agent_note_id → promoter_node_id
        with provenance containing the broadcast payload snapshot.

        Validates:
          1. agent_note_id exists with authority=agent
          2. promoter_node_id exists with authority ∈ {primary, derived, internal}
          3. via_cert resolves to PASS in _meta_ledger.json within staleness
          4. broadcast_payload timestamp within ±60s of now

        Phase 6: when ``mcp_session`` is provided (non-empty string), the
        per-session write counter in ``_agent_writes.json`` is incremented
        under flock before the promoted-from edge is written. Over-cap
        attempts raise RateLimitExceeded (subclass of RuntimeError via
        tools.qa_kg_mcp.rate_limit). ``mcp_session=None`` preserves the
        Phase 2 call shape for extractor-bus and fixture callers so the
        counter is agent-write-path-specific, not blanket.
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

        # Phase 5: graph_hash staleness check (D6). Additive — only fires
        # when the ledger entry for cert [228] carries a graph_hash. Live
        # canonical hash must match ledger's recorded hash; rebuild via
        # `python qa_alphageometry_ptolemy/qa_meta_validator.py` refreshes
        # the ledger. Pre-Phase-5 ledgers are silently compatible (entry
        # absent → fall through).
        p5_entry = ledger.get(DETERMINISM_CERT_LEDGER_KEY) or {}
        p5_expected = p5_entry.get("graph_hash")
        if p5_expected:
            from tools.qa_kg.canonicalize import graph_hash as _gh
            live_hash = _gh(self.conn)
            if live_hash != p5_expected:
                raise FirewallViolation(
                    f"graph_hash drift vs cert [228] ledger entry: "
                    f"ledger={p5_expected[:12]}… live={live_hash[:12]}…. "
                    f"Rerun `python qa_alphageometry_ptolemy/qa_meta_validator.py` "
                    f"to refresh ledger after DB-mutating extractor passes."
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

        # Phase 6: per-session rate limit. Only fires when mcp_session is set
        # (the MCP server is the caller). Increment happens AFTER staleness
        # validation so that a stale-ledger rejection does NOT burn a count.
        if mcp_session:
            from tools.qa_kg_mcp.rate_limit import increment as _rl_increment
            _rl_increment(
                mcp_session,
                ledger_path=Path(agent_writes_path) if agent_writes_path else None,
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
