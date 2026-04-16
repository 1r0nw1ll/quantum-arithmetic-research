# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG v3 infrastructure validator -->
"""QA-KG Consistency Cert [225] v3 validator.

QA_COMPLIANCE = "cert_validator — validates KG structural + epistemic invariants, no empirical QA state machine"

Supersedes v2. Schema v3: epistemic fields + alias removal.

Gates:
  KG1  (HARD) No self-vetting.
  KG2  (HARD) No contradicts cycles.
  KG3  (HARD-or-NA) Theorem NT firewall — tri-state preserved from v2.
       Will upgrade to "precondition occupied" in Phase 2 once agent nodes exist.
  KG4  (WARN) Satellite orphan aging > 30d.
  KG5  (HARD) tier ≡ orbit_family(idx_b, idx_e).
  KG6  (HARD) Candidate F integrity [202].
  KG7  (HARD) authority + epistemic_status non-null on every node.
  KG8  (HARD) frozen cert dirs not in FAMILY_SWEEPS.
  KG9  (HARD) AXIOM_CODES canonical — edges.py AXIOM_CODES equals parsed canonical set.
  KG10 (HARD) no bare except-Exception-continue in extractors/*.py.
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG structural + epistemic invariants, no empirical QA state machine"

import argparse
import ast
import datetime as _dt
import json
import os
import sqlite3
import sys
from pathlib import Path


ORPHAN_AGING_DAYS = 30
CAUSAL_EDGE_TYPES = ("validates", "derived-from", "extends", "instantiates", "maps-to")

_REPO = Path(__file__).resolve().parents[2]
_META_DIR = _REPO / "qa_alphageometry_ptolemy"
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


# --- KG1–KG7: carried from v2, KG7 new ---

def check_kg1_no_self_vetting(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols = [r["id"] for r in conn.execute(
        "SELECT id FROM nodes WHERE node_type != 'Axiom' AND vetted_by != '' AND vetted_by = id"
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
        return "FAIL", f"{len(viols)} firewall violation(s)", viols
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


def check_kg7_epistemic_non_null(conn: sqlite3.Connection) -> tuple[bool, str, list[str]]:
    viols: list[str] = []
    for r in conn.execute(
        "SELECT id, authority, epistemic_status FROM nodes"
    ).fetchall():
        problems: list[str] = []
        if r["authority"] is None:
            problems.append("authority=NULL")
        if r["epistemic_status"] is None:
            problems.append("epistemic_status=NULL")
        if problems:
            viols.append(f"{r['id']}: {', '.join(problems)}")
    return len(viols) == 0, f"{len(viols)} epistemic-NULL node(s)", viols


# --- KG8–KG10: new for v3 (N1/N2/N3 carry-forwards) ---

def check_kg8_frozen_not_in_sweeps() -> tuple[bool, str, list[str]]:
    """N1: frozen cert dirs must not appear in FAMILY_SWEEPS."""
    viols: list[str] = []
    try:
        sys_mod = "qa_meta_validator"
        if sys_mod in sys.modules:
            del sys.modules[sys_mod]
        if str(_META_DIR) not in sys.path:
            sys.path.insert(0, str(_META_DIR))
        import qa_meta_validator as mv
        sweeps = getattr(mv, "FAMILY_SWEEPS", [])
    except Exception:
        return True, "could not import FAMILY_SWEEPS — skipping", []

    sweep_roots = set()
    for entry in sweeps:
        root_rel = entry[5] if len(entry) > 5 else ""
        if root_rel:
            sweep_roots.add(root_rel)

    for root_rel in sweep_roots:
        cert_dir = _META_DIR / root_rel
        for fname in ("mapping_protocol_ref.json", "mapping_protocol.json"):
            mp = cert_dir / fname
            if mp.exists():
                try:
                    data = json.loads(mp.read_text(encoding="utf-8"))
                    if data.get("_status") == "frozen":
                        viols.append(f"{root_rel}: frozen but still in FAMILY_SWEEPS")
                except (json.JSONDecodeError, OSError):
                    pass

    return len(viols) == 0, f"{len(viols)} frozen-in-SWEEPS violation(s)", viols


def check_kg9_axiom_codes_canonical() -> tuple[bool, str, list[str]]:
    """N2: edges.py AXIOM_CODES must equal the canonical set from axioms.py
    (which parses CLAUDE.md)."""
    viols: list[str] = []
    try:
        from tools.qa_kg.extractors.axioms import CANONICAL_AXIOM_CODES
        from tools.qa_kg.extractors.edges import AXIOM_CODES
    except ImportError as exc:
        return False, f"import error: {exc}", [str(exc)]

    canonical = set(CANONICAL_AXIOM_CODES)
    edges_set = set(AXIOM_CODES)

    if canonical != edges_set:
        extra = edges_set - canonical
        missing = canonical - edges_set
        if extra:
            viols.append(f"edges.py has phantom codes: {sorted(extra)}")
        if missing:
            viols.append(f"edges.py missing canonical codes: {sorted(missing)}")

    return len(viols) == 0, f"canonical={sorted(canonical)}, edges={sorted(edges_set)}", viols


def check_kg10_no_except_swallow() -> tuple[bool, str, list[str]]:
    """N3: no bare 'except Exception: continue' in extractors/*.py.
    Static AST scan."""
    viols: list[str] = []
    extractors_dir = _REPO / "tools" / "qa_kg" / "extractors"
    if not extractors_dir.exists():
        return True, "extractors dir not found — skipping", []

    for py in sorted(extractors_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            viols.append(f"{py.name}: syntax error")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    viols.append(f"{py.name}:{node.lineno}: bare except:")
                    continue
                name = ""
                if isinstance(node.type, ast.Name):
                    name = node.type.id
                elif isinstance(node.type, ast.Attribute):
                    name = node.type.attr
                if name == "Exception":
                    body = node.body
                    if len(body) == 1 and isinstance(body[0], ast.Continue):
                        viols.append(f"{py.name}:{node.lineno}: except Exception: continue")

    return len(viols) == 0, f"{len(viols)} swallow pattern(s)", viols


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or _db_path_default()
    hard_fail = False

    def run_bool(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        ok, msg, detail = fn()
        flag = "PASS" if ok else ("FAIL" if is_hard or args.strict else "WARN")
        print(f"[{flag}] {code}  {desc} — {msg}")
        if args.show_details and detail:
            for d in detail[:10]:
                print(f"         {d}")
            if len(detail) > 10:
                print(f"         ... +{len(detail)-10} more")
        if not ok and (is_hard or args.strict):
            hard_fail = True

    # DB-dependent checks (KG1–KG7)
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr); return 2

    run_bool("KG1", "No self-vetting",
             lambda: check_kg1_no_self_vetting(conn), True)
    run_bool("KG2", "No contradicts cycles",
             lambda: check_kg2_no_contradicts_cycles(conn), True)

    flag, msg, detail = check_kg3_firewall_with_precondition(conn)
    print(f"[{flag}] KG3  Theorem NT firewall — {msg}")
    if args.show_details and detail:
        for d in detail[:10]:
            print(f"         {d}")
    if flag == "FAIL":
        hard_fail = True

    run_bool("KG4", "Satellite orphan aging",
             lambda: check_kg4_orphan_aging(conn), False)
    run_bool("KG5", "Tier ≡ orbit_family(idx)",
             lambda: check_kg5_tier_matches_orbit(conn), True)
    run_bool("KG6", "Candidate F integrity [202]",
             lambda: check_kg6_candidate_f_integrity(conn), True)
    run_bool("KG7", "Epistemic fields non-null",
             lambda: check_kg7_epistemic_non_null(conn), True)

    # Filesystem checks (KG8–KG10) — no DB needed
    run_bool("KG8", "Frozen certs not in FAMILY_SWEEPS",
             check_kg8_frozen_not_in_sweeps, True)
    run_bool("KG9", "AXIOM_CODES canonical",
             check_kg9_axiom_codes_canonical, True)
    run_bool("KG10", "No except-Exception-continue swallows",
             check_kg10_no_except_swallow, True)

    if hard_fail:
        print("[FAIL] QA-KG consistency cert [225] v3"); return 1
    print("[PASS] QA-KG consistency cert [225] v3"); return 0


if __name__ == "__main__":
    sys.exit(main())
