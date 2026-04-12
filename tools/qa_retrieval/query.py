"""
QA Conversation Retrieval — query interface.

QA_COMPLIANCE = "observer=retrieval_query, state_alphabet=mod9"

Implements the three A-RAG tools certified by family [210]:
  keyword_search  → KEYWORD_VIEW  (FTS5 BM25, sector/role prefilter)
  semantic_search → SEMANTIC_VIEW (Personalized PageRank over typed edges)
  chunk_read      → CHUNK_STORE   (direct message lookup by ID)

All scoring happens at the observer layer (float BM25, float PPR probabilities).
Sector / role filters are integer-only (QA layer). The boundary is crossed
exactly twice per query per Theorem NT:
  1. query text → (b, e) via Candidate F              (observer → discrete)
  2. discrete sector/graph operations → ranked scores (discrete → observer)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path("/home/player2/signal_experiments")
sys.path.insert(0, str(REPO_ROOT))

from tools.qa_retrieval.schema import (  # noqa: E402
    DB_PATH,
    ROLE_RANK,
    compute_be,
    compute_sector,
    open_db,
    dr,
)

# Graph-type module [210] wrapper — optional import for typed multi-hop traversal
try:
    sys.path.insert(0, str(REPO_ROOT / "qa_lab"))
    from qa_graph.knowledge_graph import QAKnowledgeGraph, diagonal_index
    _HAS_KG = True
except ImportError:
    _HAS_KG = False


# --- Result shape ---


@dataclass
class SearchResult:
    msg_id: str
    source: str
    role: str
    conv_id: str
    conv_title: str | None
    content_type: str
    b: int
    e: int
    d_label: int
    a_label: int
    char_count: int
    score: float  # float, observer-layer measurement
    text_snippet: str
    create_time_utc: str | None

    def to_dict(self) -> dict:
        return asdict(self)


# --- Helpers ---


def _sector_where_clause(sector: tuple[int, int] | None) -> tuple[str, list]:
    """Return (sql_fragment, params) for an optional sector filter on messages.

    The tuple holds T-operator labels (d_label, a_label), not QA state.
    """
    if sector is None:
        return "", []
    d_label, a_label = sector
    return (
        "AND (1 + ((b + e - 1) % 9)) = ? AND (1 + ((b + 2*e - 1) % 9)) = ?",
        [d_label, a_label],
    )


def _row_to_result(row: tuple, score: float) -> SearchResult:
    return SearchResult(
        msg_id=row[0],
        source=row[1],
        role=row[2],
        conv_id=row[3],
        conv_title=row[4],
        content_type=row[5],
        b=row[6],
        e=row[7],
        d_label=row[8],
        a_label=row[9],
        char_count=row[10],
        score=float(score),
        text_snippet=row[11][:200] if row[11] else "",
        create_time_utc=row[12],
    )


_BASE_SELECT = """
    SELECT m.msg_id, m.source, m.role, m.conv_id, m.conv_title, m.content_type,
           m.b, m.e,
           1 + ((m.b + m.e - 1) % 9) as d_label,
           1 + ((m.b + 2*m.e - 1) % 9) as a_label,
           m.char_count, m.raw_text, m.create_time_utc
    FROM messages m
"""


# --- Query sector derivation ---


def query_sector(query_text: str, role: str = "user") -> tuple[int, int]:
    """Compute the (b, e) sector label a query text would land in under Candidate F.

    Used when you want to 'find messages in my own orbit' — compute the
    query's sector and use it as a KEYWORD_VIEW / SEMANTIC_VIEW prefilter.
    """
    b, e = compute_be(query_text, role)
    return compute_sector(b, e)


# --- KEYWORD_VIEW → keyword_search ---


def keyword_search(
    query: str,
    sector: tuple[int, int] | None = None,
    role: str | None = None,
    source: str | None = None,
    limit: int = 10,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """FTS5 keyword search, optionally narrowed by sector / role / source.

    SECTOR is an integer tuple (d_label, a_label), both in {1..9}.
    Returns results ranked by BM25 (float, observer-layer).
    """
    owned = conn is None
    if owned:
        conn = open_db(DB_PATH)
    try:
        params: list = [query]
        sector_sql, sector_params = _sector_where_clause(sector)
        role_sql = "AND m.role = ?" if role else ""
        source_sql = "AND m.source = ?" if source else ""

        sql = f"""
        SELECT m.msg_id, m.source, m.role, m.conv_id, m.conv_title, m.content_type,
               m.b, m.e,
               1 + ((m.b + m.e - 1) % 9),
               1 + ((m.b + 2*m.e - 1) % 9),
               m.char_count, m.raw_text, m.create_time_utc,
               bm25(messages_fts) as score
        FROM messages_fts
        JOIN messages m ON m.msg_id = messages_fts.msg_id
        WHERE messages_fts MATCH ?
        {sector_sql}
        {role_sql}
        {source_sql}
        ORDER BY score
        LIMIT ?
        """
        params.extend(sector_params)
        if role:
            params.append(role)
        if source:
            params.append(source)
        params.append(limit)

        results = []
        for row in conn.execute(sql, params).fetchall():
            # bm25() returns NEGATIVE scores (smaller = better match in SQLite FTS5)
            # Flip sign so "higher score = more relevant" at the API surface.
            fts_score = -float(row[13])
            results.append(_row_to_result(row[:13], fts_score))
        return results
    finally:
        if owned:
            conn.close()


# --- SEMANTIC_VIEW → semantic_search via Personalized PageRank ---


def _load_edge_graph(
    conn: sqlite3.Connection,
    edge_types: tuple[str, ...] = ("parent", "cite", "succ", "ref"),
    source_filter: str | None = None,
) -> tuple[dict[str, list[tuple[str, int]]], set[str]]:
    """Build an in-memory adjacency dict: node → list of (neighbor, weight).

    Treats all edges as undirected for PPR purposes (follow both directions).
    """
    placeholders = ",".join("?" * len(edge_types))
    where = f"WHERE edge_type IN ({placeholders})"
    params: list = list(edge_types)

    if source_filter:
        where += (
            " AND src_msg_id IN (SELECT msg_id FROM messages WHERE source = ?)"
            " AND dst_msg_id IN (SELECT msg_id FROM messages WHERE source = ?)"
        )
        params.extend([source_filter, source_filter])

    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    nodes: set[str] = set()
    for src, dst, w in conn.execute(
        f"SELECT src_msg_id, dst_msg_id, weight FROM edges {where}",
        params,
    ).fetchall():
        nodes.add(src)
        nodes.add(dst)
        adj[src].append((dst, w))
        adj[dst].append((src, w))  # undirected for PPR
    return adj, nodes


def _personalized_pagerank(
    adj: dict[str, list[tuple[str, int]]],
    nodes: set[str],
    seeds: list[str],
    alpha: float = 0.5,
    max_iterations: int = 30,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Pure-Python Personalized PageRank via power iteration.

    alpha = teleport probability (0.5 per HippoRAG tuning).
    Returns {node_id: stationary_probability}. Only seeds receive teleport mass.
    """
    if not seeds:
        return {}
    valid_seeds = [s for s in seeds if s in nodes]
    if not valid_seeds:
        return {}

    n_nodes = len(nodes)
    if n_nodes == 0:
        return {}

    # Pre-compute out-degrees (sum of weights)
    out_weight: dict[str, int] = {node: sum(w for _, w in adj.get(node, [])) for node in nodes}

    # Teleport distribution: uniform over seeds
    teleport = 1.0 / len(valid_seeds)
    teleport_dist = {s: teleport for s in valid_seeds}

    # Initialize scores with teleport distribution
    scores = {node: 0.0 for node in nodes}
    for s in valid_seeds:
        scores[s] = teleport

    for iteration in range(max_iterations):
        new_scores = {node: 0.0 for node in nodes}

        # Teleport contribution: alpha * teleport_dist
        for s, t in teleport_dist.items():
            new_scores[s] += alpha * t

        # Propagation contribution: (1 - alpha) * sum of neighbor scores / out_weight
        for node, score in scores.items():
            if score <= 0:
                continue
            neighbors = adj.get(node, [])
            if not neighbors:
                # Dangling node — redistribute via teleport
                for s, t in teleport_dist.items():
                    new_scores[s] += (1 - alpha) * score * t
                continue
            total_w = out_weight[node]
            if total_w <= 0:
                continue
            for neighbor, w in neighbors:
                new_scores[neighbor] += (1 - alpha) * score * (w / total_w)

        # Check convergence (L1 difference)
        delta = sum(abs(new_scores[node] - scores[node]) for node in nodes)
        scores = new_scores
        if delta < tolerance:
            break

    return scores


def semantic_search(
    seed_msg_ids: list[str],
    edge_types: tuple[str, ...] = ("parent", "cite", "succ", "ref"),
    alpha: float = 0.5,
    max_iterations: int = 30,
    limit: int = 10,
    sector_filter: tuple[int, int] | None = None,
    source_filter: str | None = None,
    exclude_seeds: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """Personalized PageRank over the typed edge graph, seeded at given messages.

    Borrowed from HippoRAG's PPR ranking primitive. Alpha = teleport probability.
    """
    owned = conn is None
    if owned:
        conn = open_db(DB_PATH)
    try:
        adj, nodes = _load_edge_graph(conn, edge_types=edge_types, source_filter=source_filter)
        scores = _personalized_pagerank(
            adj, nodes, seed_msg_ids,
            alpha=alpha, max_iterations=max_iterations,
        )
        if not scores:
            return []

        # Fetch message rows for top candidates, optionally filtered by sector
        sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
        results: list[SearchResult] = []

        seed_set = set(seed_msg_ids) if exclude_seeds else set()

        for msg_id, score in sorted_nodes:
            if score <= 0:
                break
            if msg_id in seed_set:
                continue
            params: list = [msg_id]
            sector_sql, sector_params = _sector_where_clause(sector_filter)
            params.extend(sector_params)
            sql = f"{_BASE_SELECT} WHERE m.msg_id = ? {sector_sql} LIMIT 1"
            row = conn.execute(sql, params).fetchone()
            if row is None:
                continue
            results.append(_row_to_result(row, score))
            if len(results) >= limit:
                break
        return results
    finally:
        if owned:
            conn.close()


# --- GRAPH_VIEW → graph_search (typed multi-hop via knowledge_graph.py [210]) ---


def graph_search(
    seed_msg_ids: list[str],
    edge_types: tuple[str, ...] = ("parent", "succ"),
    max_hops: int = 2,
    limit: int = 10,
    sector_filter: tuple[int, int] | None = None,
    exclude_seeds: bool = True,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """Typed multi-hop BFS traversal via QAKnowledgeGraph [210].

    Unlike PPR (which computes a global stationary distribution), graph_search
    performs bounded BFS restricted to the specified edge types. This is faster
    for shallow traversals (1-3 hops) and gives exact typed reachability.

    Requires qa_lab/qa_graph/knowledge_graph.py (slot 3 module).
    Falls back to semantic_search (PPR) if the KG module is unavailable.
    """
    if not _HAS_KG:
        return semantic_search(
            seed_msg_ids, edge_types=edge_types, limit=limit,
            sector_filter=sector_filter, conn=conn,
        )

    owned = conn is None
    if owned:
        conn = open_db(DB_PATH)
    try:
        kg = QAKnowledgeGraph.from_qa_retrieval_db(str(DB_PATH), limit_messages=None)

        reachable: set[str] = set()
        for seed in seed_msg_ids:
            if seed in kg.nodes:
                reached = kg.multi_hop_neighbors(seed, edge_types, max_hops=max_hops)
                reachable.update(reached)

        if exclude_seeds:
            reachable -= set(seed_msg_ids)

        if not reachable:
            return []

        results: list[SearchResult] = []
        for msg_id in reachable:
            if len(results) >= limit:
                break
            params: list = [msg_id]
            sector_sql, sector_params = _sector_where_clause(sector_filter)
            params.extend(sector_params)
            sql = f"{_BASE_SELECT} WHERE m.msg_id = ? {sector_sql} LIMIT 1"
            row = conn.execute(sql, params).fetchone()
            if row is not None:
                hop_dist = 1.0  # uniform score for BFS (not ranked by distance)
                results.append(_row_to_result(row, hop_dist))

        return results
    finally:
        if owned:
            conn.close()


def diagonal_search(
    diagonal_e: int,
    query: str | None = None,
    limit: int = 10,
    source: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """Search within a single role diagonal.

    The role-diagonal theorem ([210], [214]) guarantees that (a_label - d_label)
    mod 9 = e mod 9. So filtering by diagonal_e restricts to exactly one of the
    9 disjoint 9-sector stripes — structurally equivalent to a role filter but
    without requiring the role name.

    If query is provided, applies FTS5 keyword search within the diagonal.
    Otherwise returns the most recent messages on that diagonal.
    """
    if not (1 <= diagonal_e <= 9):
        raise ValueError(f"diagonal_e must be in {{1..9}}, got {diagonal_e}")

    owned = conn is None
    if owned:
        conn = open_db(DB_PATH)
    try:
        diag_mod = diagonal_e % 9
        diag_where = f"AND ((1 + ((m.b + 2*m.e - 1) % 9)) - (1 + ((m.b + m.e - 1) % 9)) + 9) % 9 = {diag_mod}"

        if query:
            params: list = [query]
            sql = f"""
            SELECT m.msg_id, m.source, m.role, m.conv_id, m.conv_title, m.content_type,
                   m.b, m.e,
                   1 + ((m.b + m.e - 1) % 9),
                   1 + ((m.b + 2*m.e - 1) % 9),
                   m.char_count, m.raw_text, m.create_time_utc,
                   bm25(messages_fts) as score
            FROM messages_fts
            JOIN messages m ON m.msg_id = messages_fts.msg_id
            WHERE messages_fts MATCH ?
            {diag_where}
            ORDER BY score
            LIMIT ?
            """
            source_sql = ""
            if source:
                source_sql = "AND m.source = ?"
                sql = sql.replace(diag_where, f"{diag_where} {source_sql}")
                params.append(source)
            params.append(limit)
        else:
            params = [limit]
            source_sql = f"AND m.source = '{source}'" if source else ""
            sql = f"""
            {_BASE_SELECT}
            WHERE 1=1 {diag_where} {source_sql}
            ORDER BY m.create_time_utc DESC
            LIMIT ?
            """

        results = []
        for row in conn.execute(sql, params).fetchall():
            if query:
                fts_score = -float(row[13])
                results.append(_row_to_result(row[:13], fts_score))
            else:
                results.append(_row_to_result(row, 1.0))
        return results
    finally:
        if owned:
            conn.close()


# --- CHUNK_STORE → chunk_read ---


def chunk_read(
    msg_ids: Iterable[str],
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """Direct message retrieval by ID. Observer-layer raw text passthrough."""
    owned = conn is None
    if owned:
        conn = open_db(DB_PATH)
    try:
        ids = list(msg_ids)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        sql = f"{_BASE_SELECT} WHERE m.msg_id IN ({placeholders})"
        results = []
        for row in conn.execute(sql, ids).fetchall():
            results.append(_row_to_result(row, 1.0))
        # Preserve input order
        order = {mid: i for i, mid in enumerate(ids)}
        results.sort(key=lambda r: order.get(r.msg_id, 999999))
        return results
    finally:
        if owned:
            conn.close()


# --- Convenience: full pipeline like A-RAG's composed tool calls ---


def retrieve_pipeline(
    query: str,
    limit: int = 10,
    fts_candidate_depth: int = 30,
    use_semantic: bool = True,
    sector: tuple[int, int] | None = None,
    role: str | None = None,
    source: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict:
    """Four-stage pipeline matching the A-RAG cert contract:
    (1) compute query sector → (2) FTS5 keyword narrow →
    (3) PPR graph walk seeded at keyword hits → (4) merged ranking.

    Returns a dict with stage-by-stage results for debugging + final ranking.
    """
    owned = conn is None
    if owned:
        conn = open_db(DB_PATH)
    try:
        # Stage 1: query sector
        q_sector = query_sector(query, role=role or "user")

        # Stage 2: FTS keyword narrow (over the query sector OR whole corpus)
        # Run without sector filter FIRST so the FTS can find relevant messages
        # anywhere. The query's own sector is informational.
        kw_results = keyword_search(
            query,
            sector=sector,
            role=role,
            source=source,
            limit=fts_candidate_depth,
            conn=conn,
        )

        sem_results: list[SearchResult] = []
        if use_semantic and kw_results:
            # Stage 3: PPR seeded at keyword hits
            seeds = [r.msg_id for r in kw_results[:10]]
            sem_results = semantic_search(
                seeds,
                limit=limit,
                sector_filter=sector,
                source_filter=source,
                conn=conn,
            )

        # Stage 4: merged ranking — FTS hits first, PPR expansions after
        seen: set[str] = set()
        final: list[SearchResult] = []
        for r in kw_results[:limit]:
            if r.msg_id not in seen:
                final.append(r)
                seen.add(r.msg_id)
        for r in sem_results:
            if r.msg_id not in seen and len(final) < 2 * limit:
                final.append(r)
                seen.add(r.msg_id)

        return {
            "query": query,
            "query_sector_if_user": q_sector,
            "stages": {
                "keyword": [r.to_dict() for r in kw_results],
                "semantic": [r.to_dict() for r in sem_results],
            },
            "final": [r.to_dict() for r in final[:limit]],
        }
    finally:
        if owned:
            conn.close()


# --- Stats / visualization ---


def _print_stats(source: str | None = None, role: str | None = None) -> None:
    """Pretty-print sector distribution as a mod-9 grid with role diagonals."""
    conn = open_db(DB_PATH)
    try:
        where = []
        params: list = []
        if source:
            where.append("source = ?")
            params.append(source)
        if role:
            where.append("role = ?")
            params.append(role)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # Total counts
        total = conn.execute(f"SELECT COUNT(*) FROM messages {where_sql}", params).fetchone()[0]
        print(f"Total messages: {total:,}" + (f" (filtered: {', '.join(where)})" if where else ""))

        # Role distribution
        print("\nRole distribution:")
        for r in conn.execute(
            f"""SELECT role, COUNT(*) FROM messages {where_sql}
                GROUP BY role ORDER BY COUNT(*) DESC""",
            params,
        ):
            pct = 100 * r[1] / total if total else 0
            print(f"  {r[0]:<10}: {r[1]:>7,}  ({pct:>4.1f}%)")

        # Sector grid (per role), iterate canonical roles in rank order
        print("\nSector grid by role (d_label rows, a_label columns, counts in cells):")
        for role_name in sorted(ROLE_RANK, key=lambda r: ROLE_RANK[r]):
            if role and role != role_name:
                continue
            role_where = where[:] + ["role = ?"]
            role_params = params[:] + [role_name]
            role_where_sql = "WHERE " + " AND ".join(role_where)

            grid_rows = conn.execute(f"""
                SELECT 1 + ((b+e-1) % 9) AS d_lbl,
                       1 + ((b+2*e-1) % 9) AS a_lbl,
                       COUNT(*) AS n
                FROM messages
                {role_where_sql}
                GROUP BY d_lbl, a_lbl
            """, role_params).fetchall()
            if not grid_rows:
                continue

            grid = [[0] * 10 for _ in range(10)]  # 1..9 used
            total_role = 0
            for d_lbl, a_lbl, n in grid_rows:
                grid[d_lbl][a_lbl] = n
                total_role += n

            print(f"\n  ─── role={role_name} · n={total_role:,} · diagonal=(a-d)≡{ROLE_RANK[role_name]}(mod 9) ───")
            header = "        " + " ".join(f"a={i}" for i in range(1, 10))  # noqa: A2-2
            print(header)
            for d_lbl in range(1, 10):
                cells = [f"{grid[d_lbl][a_lbl]:>4}" if grid[d_lbl][a_lbl] else "    ." for a_lbl in range(1, 10)]
                print(f"  d={d_lbl}   " + "  ".join(cells))  # noqa: A2-1

        # Edge stats
        print("\nEdge counts:")
        for r in conn.execute("SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type"):
            print(f"  {r[0]:<8}: {r[1]:>8,}")

        # Integrity check: role diagonal property
        rows = conn.execute(f"""
            SELECT role,
                   1 + ((b+e-1) % 9) AS d_lbl,
                   1 + ((b+2*e-1) % 9) AS a_lbl,
                   COUNT(*) AS n
            FROM messages
            {where_sql}
            GROUP BY role, d_lbl, a_lbl
        """, params).fetchall()
        violations = 0
        for role_name, d_lbl, a_lbl, n in rows:
            expected = ROLE_RANK[role_name] % 9
            observed = (a_lbl - d_lbl) % 9
            if observed != expected:
                violations += n
        if violations:
            print(f"\n⚠  DIAGONAL VIOLATIONS: {violations:,} rows (should be 0)")
        else:
            print(f"\n✓  Diagonal property holds: {len(rows)} sector/role combos, 0 violations")

    finally:
        conn.close()


# --- CLI ---


def _print_results(results: list[SearchResult], show_snippets: bool = True) -> None:
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        sector_str = f"(d={r.d_label},a={r.a_label})"  # noqa: A2-1,A2-2
        print(
            f"  {i:2}. [{r.source}/{r.role}] "
            f"b={r.b} e={r.e} sector={sector_str} "
            f"chars={r.char_count} score={r.score:.4f}"
        )
        title = (r.conv_title or "").strip()
        print(f"      title: {title[:80]}")
        if show_snippets:
            snippet = r.text_snippet.replace("\n", " ")[:200]
            print(f"      > {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="QA Conversation Retrieval — query CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    kw = sub.add_parser("keyword", help="KEYWORD_VIEW: FTS5 search")
    kw.add_argument("query")
    kw.add_argument("--sector", type=str, help="d,a (e.g. '6,7')")
    kw.add_argument("--role", choices=sorted(ROLE_RANK))
    kw.add_argument("--source", help="Source filter (e.g., chatgpt, claude, gemini, obsidian)")
    kw.add_argument("--limit", type=int, default=10)
    kw.add_argument("--json", action="store_true")

    sem = sub.add_parser("semantic", help="SEMANTIC_VIEW: PPR over typed edges")
    sem.add_argument("seeds", nargs="+", help="One or more seed msg_ids")
    sem.add_argument("--alpha", type=float, default=0.5)
    sem.add_argument("--limit", type=int, default=10)
    sem.add_argument("--sector", type=str)
    sem.add_argument("--source", help="Source filter (e.g., chatgpt, claude, gemini, obsidian)")
    sem.add_argument("--json", action="store_true")

    ch = sub.add_parser("chunk", help="CHUNK_STORE: direct message lookup")
    ch.add_argument("msg_ids", nargs="+")
    ch.add_argument("--json", action="store_true")

    pipe = sub.add_parser("pipeline", help="Full four-stage retrieval pipeline")
    pipe.add_argument("query")
    pipe.add_argument("--limit", type=int, default=10)
    pipe.add_argument("--sector", type=str)
    pipe.add_argument("--role", choices=sorted(ROLE_RANK))
    pipe.add_argument("--source", help="Source filter (e.g., chatgpt, claude, gemini, obsidian)")
    pipe.add_argument("--no-semantic", action="store_true")
    pipe.add_argument("--json", action="store_true")

    gs = sub.add_parser("graph", help="GRAPH_VIEW: typed multi-hop BFS via knowledge_graph.py")
    gs.add_argument("seeds", nargs="+", help="Seed msg_ids")
    gs.add_argument("--edge-types", type=str, default="parent,succ", help="Comma-separated edge types")
    gs.add_argument("--hops", type=int, default=2, help="Max hop depth (default 2)")
    gs.add_argument("--limit", type=int, default=10)
    gs.add_argument("--sector", type=str)
    gs.add_argument("--json", action="store_true")

    diag = sub.add_parser("diagonal", help="DIAGONAL_VIEW: search within a role diagonal")
    diag.add_argument("diagonal_e", type=int, help="Diagonal index (1=user, 2=assistant, 5=thought, ...)")
    diag.add_argument("--query", type=str, help="Optional FTS5 keyword query within diagonal")
    diag.add_argument("--source", help="Source filter")
    diag.add_argument("--limit", type=int, default=10)
    diag.add_argument("--json", action="store_true")

    stats = sub.add_parser("stats", help="Show sector distribution and DB stats")
    stats.add_argument("--source", help="Source filter (e.g., chatgpt, claude, gemini, obsidian)")
    stats.add_argument("--role", choices=sorted(ROLE_RANK))

    args = parser.parse_args()

    def parse_sector(s: str | None) -> tuple[int, int] | None:
        if not s:
            return None
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError(f"sector must be 'd,a', got {s!r}")
        return int(parts[0]), int(parts[1])

    if args.cmd == "keyword":
        results = keyword_search(
            args.query,
            sector=parse_sector(args.sector),
            role=args.role,
            source=args.source,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        else:
            print(f"Keyword search: {args.query!r}  →  {len(results)} results")
            _print_results(results)

    elif args.cmd == "semantic":
        results = semantic_search(
            args.seeds,
            alpha=args.alpha,
            limit=args.limit,
            sector_filter=parse_sector(args.sector),
            source_filter=args.source,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        else:
            print(f"Semantic search: {len(args.seeds)} seeds  →  {len(results)} results")
            _print_results(results)

    elif args.cmd == "chunk":
        results = chunk_read(args.msg_ids)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        else:
            print(f"Chunk read: {len(args.msg_ids)} ids  →  {len(results)} rows")
            _print_results(results)

    elif args.cmd == "graph":
        edge_types = tuple(args.edge_types.split(","))
        results = graph_search(
            args.seeds,
            edge_types=edge_types,
            max_hops=args.hops,
            limit=args.limit,
            sector_filter=parse_sector(args.sector),
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        else:
            print(f"Graph search: {len(args.seeds)} seeds × {args.hops} hops via {edge_types}  →  {len(results)} results")
            _print_results(results)

    elif args.cmd == "diagonal":
        results = diagonal_search(
            args.diagonal_e,
            query=args.query,
            limit=args.limit,
            source=args.source,
        )
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        else:
            role_name = {v: k for k, v in ROLE_RANK.items()}.get(args.diagonal_e, "?")
            print(f"Diagonal search: e={args.diagonal_e} ({role_name})  →  {len(results)} results")
            _print_results(results)

    elif args.cmd == "stats":
        _print_stats(source=args.source, role=args.role)

    elif args.cmd == "pipeline":
        result = retrieve_pipeline(
            args.query,
            limit=args.limit,
            sector=parse_sector(args.sector),
            role=args.role,
            source=args.source,
            use_semantic=not args.no_semantic,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Query: {result['query']!r}")
            print(f"Query-as-user sector: {result['query_sector_if_user']}")
            print(f"\n=== Stage: keyword (FTS5) — {len(result['stages']['keyword'])} candidates ===")
            kw_results = [SearchResult(**{k: v for k, v in r.items()}) for r in result['stages']['keyword']]
            _print_results(kw_results[:5], show_snippets=False)
            print(f"\n=== Stage: semantic (PPR) — {len(result['stages']['semantic'])} expansions ===")
            sem_results = [SearchResult(**{k: v for k, v in r.items()}) for r in result['stages']['semantic']]
            _print_results(sem_results[:5], show_snippets=False)
            print(f"\n=== Final merged top-{args.limit} ===")
            final_results = [SearchResult(**{k: v for k, v in r.items()}) for r in result['final']]
            _print_results(final_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
