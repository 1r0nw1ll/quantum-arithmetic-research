# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG v2 infrastructure validator -->
"""QA-KG Consistency Cert [225] v2 validator.

QA_COMPLIANCE = "cert_validator — validates KG structural invariants, no empirical QA state machine"

Invariants:
  KG1  (HARD) No self-vetting — every non-Axiom classified node with non-empty
       vetted_by has vetted_by ≠ node_id.
  KG2  (HARD) No 'contradicts' cycles.
  KG3  (HARD-or-NA) Theorem NT firewall precondition guard — if the graph
       contains at least one Unassigned node, verify that no causal edges from
       Unassigned → Cosmos/Singularity lack via_cert. If the graph has zero
       Unassigned nodes, KG3 is reported N/A (precondition not met); this
       prevents trivial PASS from an empty precondition.
  KG4  (WARN) Satellite orphan aging — stale Satellite nodes with no edges > 30d.
  KG5  (HARD) tier ≡ canonical qa_orbit_rules.orbit_family(idx_b, idx_e).
  KG6  (HARD) Candidate F integrity [202] — idx_b == dr(char_ord_sum),
       idx_e == NODE_TYPE_RANK[node_type].
  KG7  (HARD) subject_b/subject_e A1 compliance — if non-NULL, in {1..9}.
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

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from qa_orbit_rules import orbit_family as _canonical_orbit_family
from tools.qa_kg.orbit import dr, NODE_TYPE_RANK


def _db_path_default() -> Path:
    override = os.environ.get("QA_KG_DB")
    if override:
        return Path(override)
    return _REPO / "tools" / "qa_kg" / "qa_kg.db"


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise FileNotFoundError(f"QA-KG DB not found at {db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def check_kg1_no_self_vetting(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols = [r["id"] for r in conn.execute(
        "SELECT id FROM nodes "
        "WHERE node_type != 'Axiom' AND vetted_by != '' AND vetted_by = id"
    ).fetchall()]
    return len(viols) == 0, f"{len(viols)} self-vetted node(s)", viols


def check_kg2_no_contradicts_cycles(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    edges = conn.execute(
        "SELECT src_id, dst_id FROM edges WHERE edge_type = 'contradicts'"
    ).fetchall()
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["src_id"], []).append(e["dst_id"])
    cycles: list[str] = []
    for start in adj:
        stack = [(start, [start])]
        visited: set[str] = set()
        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nxt in adj.get(node, []):
                if nxt == start:
                    cycles.append(" -> ".join(path + [nxt]))
                    break
                stack.append((nxt, path + [nxt]))
    return len(cycles) == 0, f"{len(cycles)} contradicts-cycle(s)", cycles


def check_kg3_firewall_with_precondition(
    conn: sqlite3.Connection,
) -> tuple[str, str, list[str]]:
    """Returns ("PASS"|"FAIL"|"N/A", message, details).

    N/A when no Unassigned nodes exist — the firewall's precondition is
    unmet, so a "pass" would be vacuous and misleading.
    """
    n_unassigned = conn.execute(
        "SELECT COUNT(*) n FROM nodes WHERE tier = 'unassigned'"
    ).fetchone()["n"]
    if n_unassigned == 0:
        return (
            "N/A",
            "0 Unassigned nodes — firewall precondition absent; cannot meaningfully verify",
            [],
        )
    types = ",".join(f"'{t}'" for t in CAUSAL_EDGE_TYPES)
    q = f"""
    SELECT e.src_id, e.dst_id, e.edge_type FROM edges e
    JOIN nodes src ON src.id = e.src_id
    JOIN nodes dst ON dst.id = e.dst_id
    WHERE e.edge_type IN ({types})
      AND src.tier = 'unassigned'
      AND dst.tier IN ('cosmos','singularity')
      AND (e.via_cert = '' OR e.via_cert IS NULL)
    """
    viols = [f"{r['src_id']} --{r['edge_type']}--> {r['dst_id']}"
             for r in conn.execute(q).fetchall()]
    if viols:
        return "FAIL", f"{len(viols)} firewall violation(s) (of {n_unassigned} Unassigned)", viols
    return "PASS", f"0 violations across {n_unassigned} Unassigned node(s)", []


def check_kg4_orphan_aging(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=ORPHAN_AGING_DAYS)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    q = """
    SELECT id FROM nodes
      WHERE tier = 'satellite' AND updated_ts < ?
        AND id NOT IN (SELECT src_id FROM edges)
        AND id NOT IN (SELECT dst_id FROM edges)
    """
    stale = [r["id"] for r in conn.execute(q, (cutoff,)).fetchall()]
    return len(stale) == 0, f"{len(stale)} stale orphan Satellite(s) > {ORPHAN_AGING_DAYS}d", stale


def check_kg5_tier_matches_orbit(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    for r in conn.execute("SELECT id, tier, idx_b, idx_e FROM nodes").fetchall():
        cb, ce, tier = r["idx_b"], r["idx_e"], r["tier"]
        if cb is None or ce is None:
            if tier != "unassigned":
                viols.append(f"{r['id']}: NULL idx but tier={tier!r}")
            continue
        if tier == "unassigned":
            viols.append(f"{r['id']}: idx=({cb},{ce}) but tier=unassigned")
            continue
        expected = _canonical_orbit_family(cb, ce, 9)
        if tier != expected:
            viols.append(f"{r['id']}: tier={tier!r} ≠ orbit_family({cb},{ce})={expected!r}")
    return len(viols) == 0, f"{len(viols)} tier/orbit mismatch(es)", viols


def check_kg6_candidate_f_integrity(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    rows = conn.execute(
        "SELECT id, node_type, idx_b, idx_e, char_ord_sum "
        "FROM nodes WHERE tier != 'unassigned'"
    ).fetchall()
    for r in rows:
        nt = r["node_type"]
        if nt not in NODE_TYPE_RANK:
            viols.append(f"{r['id']}: node_type {nt!r} not in NODE_TYPE_RANK")
            continue
        expected_e = NODE_TYPE_RANK[nt]
        if r["idx_e"] != expected_e:
            viols.append(f"{r['id']}: idx_e={r['idx_e']} ≠ NODE_TYPE_RANK[{nt!r}]={expected_e}")
        if r["char_ord_sum"] is None:
            viols.append(f"{r['id']}: missing char_ord_sum")
            continue
        expected_b = dr(r["char_ord_sum"])
        if r["idx_b"] != expected_b:
            viols.append(f"{r['id']}: idx_b={r['idx_b']} ≠ dr({r['char_ord_sum']})={expected_b}")
    return len(viols) == 0, f"{len(viols)} Candidate F violation(s)", viols


def check_kg7_subject_coord_a1(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    rows = conn.execute(
        "SELECT id, subject_b, subject_e FROM nodes "
        "WHERE subject_b IS NOT NULL OR subject_e IS NOT NULL"
    ).fetchall()
    for r in rows:
        if r["subject_b"] is not None and not (1 <= r["subject_b"] <= 9):
            viols.append(f"{r['id']}: subject_b={r['subject_b']} out of {{1..9}}")
        if r["subject_e"] is not None and not (1 <= r["subject_e"] <= 9):
            viols.append(f"{r['id']}: subject_e={r['subject_e']} out of {{1..9}}")
    return len(viols) == 0, f"{len(viols)} subject-coord A1 violation(s)", viols


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or _db_path_default()
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr); return 2

    hard_fail = False

    def run_bool(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        ok, msg, detail = fn(conn)
        flag = "PASS" if ok else ("FAIL" if is_hard or args.strict else "WARN")
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... +{len(detail)-10} more")
        if not ok and (is_hard or args.strict):
            hard_fail = True

    run_bool("KG1", "No self-vetting", check_kg1_no_self_vetting, True)
    run_bool("KG2", "No contradicts cycles", check_kg2_no_contradicts_cycles, True)

    # KG3 has a tri-state result
    flag, msg, detail = check_kg3_firewall_with_precondition(conn)
    print(f"[{flag}] KG3  Theorem NT firewall — {msg}")
    if args.show_details and detail:
        for d in detail[:10]:
            print(f"         {d}")
    if flag == "FAIL":
        hard_fail = True

    run_bool("KG4", "Satellite orphan aging", check_kg4_orphan_aging, False)
    run_bool("KG5", "Tier ≡ orbit_family(idx)", check_kg5_tier_matches_orbit, True)
    run_bool("KG6", "Candidate F integrity [202]", check_kg6_candidate_f_integrity, True)
    run_bool("KG7", "Subject-coord A1 compliance", check_kg7_subject_coord_a1, True)

    if hard_fail:
        print("[FAIL] QA-KG consistency cert [225] v2"); return 1
    print("[PASS] QA-KG consistency cert [225] v2"); return 0


if __name__ == "__main__":
    sys.exit(main())
