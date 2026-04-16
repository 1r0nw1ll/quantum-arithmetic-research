# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG AUDIT-ONLY v1 validator; do not run against current DB -->
"""QA-KG Consistency Cert [225] v1 — FROZEN / AUDIT-ONLY.

!!! DO NOT RUN THIS VALIDATOR AGAINST A CURRENT DB !!!

This is the Phase 0 pre-rename validator. It references columns
coord_b / coord_e which no longer exist in schema v2. Attempting to
run it against a current qa_kg.db will fail with 'no such column'.

Current cert is qa_kg_consistency_cert_v2 (same repo, sibling dir).
This v1 file is preserved unmodified for audit of the Phase 0 supersession.
If a future session thinks v1 needs a fix, the correct action is a v3 bump
(per docs/specs/QA_MEM_SCOPE.md), NOT in-place edits here.

QA_COMPLIANCE = "cert_validator — AUDIT-ONLY; validates KG structural invariants, no empirical QA state machine"

Invariants:
  KG1  Cosmos provenance: every Cosmos node reaches Singularity via
       {validates, derived-from, extends, instantiates, maps-to}.
  KG2  No 'contradicts' cycles.
  KG3  Theorem NT firewall: no Unassigned→Cosmos causal edge without via_cert.
  KG4  Satellite orphan aging: flag Satellite nodes with no edges older than
       ORPHAN_AGING_DAYS (default 30).

Source: QA-MEM architecture spec in memory/project_qa_mem_architecture.md;
CLAUDE.md QA Axiom Compliance; Theorem NT (Observer Projection Firewall).

Usage:
    python qa_kg_consistency_cert_validate.py [--db PATH] [--strict]

Exit codes:
    0  all invariants PASS
    1  one or more invariants FAIL (non-strict: KG4 warnings allowed)
    2  infrastructure error (DB missing, etc.)
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG structural invariants, no empirical QA state machine"

import argparse
import datetime as _dt
import os
import sqlite3
import sys
from pathlib import Path


ORPHAN_AGING_DAYS = 30
CAUSAL_EDGE_TYPES = ("validates", "derived-from", "extends", "instantiates", "maps-to")


def _db_path_default() -> Path:
    override = os.environ.get("QA_KG_DB")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "tools" / "qa_kg" / "qa_kg.db"


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise FileNotFoundError(
            f"QA-KG DB not found at {db}. Run `python -m tools.qa_kg.cli build` first."
        )
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def check_kg1_cosmos_provenance(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    """Every Cosmos node must have provenance, either via causal chain to
    Singularity, or via an explicit non-empty vetted_by pointer (curated source)."""
    types = ",".join(f"'{t}'" for t in CAUSAL_EDGE_TYPES)
    q = f"""
    WITH RECURSIVE reach(node) AS (
        SELECT id FROM nodes WHERE tier = 'singularity'
        UNION
        SELECT e.src_id FROM edges e JOIN reach ON e.dst_id = reach.node
          WHERE e.edge_type IN ({types})
    )
    SELECT id FROM nodes
      WHERE tier = 'cosmos'
        AND id NOT IN (SELECT node FROM reach)
        AND (vetted_by IS NULL OR vetted_by = '')
    """
    orphans = [r["id"] for r in conn.execute(q).fetchall()]
    ok = len(orphans) == 0
    return ok, f"{len(orphans)} Cosmos node(s) without provenance (chain or vetted_by)", orphans


def check_kg2_no_contradicts_cycles(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    """Walk contradicts edges; flag any node reachable from itself."""
    edges = conn.execute(
        "SELECT src_id, dst_id FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["src_id"], []).append(e["dst_id"])
    cycles: list[str] = []
    for start in adj:
        stack = [(start, [start])]
        visited_local: set[str] = set()
        while stack:
            node, path = stack.pop()
            if node in visited_local:
                continue
            visited_local.add(node)
            for nxt in adj.get(node, []):
                if nxt == start:
                    cycles.append(" -> ".join(path + [nxt]))
                    break
                stack.append((nxt, path + [nxt]))
    ok = len(cycles) == 0
    return ok, f"{len(cycles)} contradicts-cycle(s) detected", cycles


def check_kg3_firewall(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    """No Unassigned→Cosmos causal edge without via_cert."""
    types = ",".join(f"'{t}'" for t in CAUSAL_EDGE_TYPES)
    q = f"""
    SELECT e.src_id, e.dst_id, e.edge_type
    FROM edges e
    JOIN nodes src ON src.id = e.src_id
    JOIN nodes dst ON dst.id = e.dst_id
    WHERE e.edge_type IN ({types})
      AND src.tier = 'unassigned'
      AND dst.tier IN ('cosmos','singularity')
      AND (e.via_cert = '' OR e.via_cert IS NULL)
    """
    viols = [f"{r['src_id']} --{r['edge_type']}--> {r['dst_id']}"
             for r in conn.execute(q).fetchall()]
    ok = len(viols) == 0
    return ok, f"{len(viols)} Theorem NT firewall violation(s)", viols


def check_kg4_orphan_aging(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    """Satellite nodes with no edges, older than ORPHAN_AGING_DAYS."""
    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=ORPHAN_AGING_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    q = """
    SELECT id FROM nodes
      WHERE tier = 'satellite'
        AND updated_ts < ?
        AND id NOT IN (SELECT src_id FROM edges)
        AND id NOT IN (SELECT dst_id FROM edges)
    """
    stale = [r["id"] for r in conn.execute(q, (cutoff,)).fetchall()]
    ok = len(stale) == 0
    return ok, f"{len(stale)} stale orphan Satellite node(s) > {ORPHAN_AGING_DAYS}d", stale


CHECKS = [
    ("KG1", "Cosmos provenance to Singularity", check_kg1_cosmos_provenance, True),
    ("KG2", "No contradicts cycles", check_kg2_no_contradicts_cycles, True),
    ("KG3", "Theorem NT firewall intact", check_kg3_firewall, True),
    ("KG4", "Satellite orphan aging", check_kg4_orphan_aging, False),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--strict", action="store_true", help="Fail on KG4 warnings")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or _db_path_default()
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 2

    hard_fail = False
    for code, desc, fn, is_hard in CHECKS:
        ok, msg, detail = fn(conn)
        flag = "PASS" if ok else ("FAIL" if is_hard or args.strict else "WARN")
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... and {len(detail) - 10} more")
        if not ok and (is_hard or args.strict):
            hard_fail = True

    if hard_fail:
        print("[FAIL] QA-KG consistency cert [225]")
        return 1
    print("[PASS] QA-KG consistency cert [225]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
