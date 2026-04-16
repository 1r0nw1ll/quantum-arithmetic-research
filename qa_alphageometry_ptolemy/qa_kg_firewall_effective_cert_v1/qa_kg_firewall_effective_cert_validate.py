# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 2 infrastructure validator -->
"""QA-KG Firewall Effective Cert [227] v1 validator.

QA_COMPLIANCE = "cert_validator — validates KG firewall effectiveness, no empirical QA state machine"

Phase 2: validates that the Theorem NT firewall (agent authority +
promote protocol) is effective at runtime and in the persisted graph.

Gates:
  FE1  (HARD) No unpromoted agent causal edges in DB.
  FE2  (HARD) via_cert on promoted-from edges resolves to PASS in
       _meta_ledger.json within staleness (14d + git_head match).
  FE3  (HARD) No promoted-from cycles.
  FE4  (HARD) Ephemeral test DB: FirewallViolation on unauthorized
       agent→Cosmos causal; success after promote. Real-shape fixtures.
  FE5  (WARN) Oldest unpromoted agent note.
  FE6  (HARD) promoted-from provenance contains broadcast_payload_snapshot.
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG firewall effectiveness, no empirical QA state machine"

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_META_DIR = _REPO / "qa_alphageometry_ptolemy"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CAUSAL_EDGE_TYPES = ("validates", "derived-from", "extends", "instantiates", "maps-to")
PROMOTED_FROM_EDGE = "promoted-from"
LEDGER_STALENESS_DAYS = 14
PROVENANCE_REQUIRED_KEYS = {"session", "signed_ts", "promoter_node_id", "broadcast_payload_snapshot"}


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


def _current_git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(_REPO),
        )
        return r.stdout.strip() or "UNKNOWN"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "UNKNOWN"


def _load_ledger() -> dict:
    p = _META_DIR / "_meta_ledger.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# --- FE1: No unpromoted agent causal edges ---

def check_fe1_no_unpromoted_agent_causal(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """Every edge where src.authority=agent AND edge_type in CAUSAL_EDGE_TYPES
    must have a corresponding promoted-from edge on the same src_id."""
    types = ",".join(f"'{t}'" for t in CAUSAL_EDGE_TYPES)
    q = f"""
    SELECT e.src_id, e.dst_id, e.edge_type FROM edges e
    JOIN nodes src ON src.id = e.src_id
    WHERE e.edge_type IN ({types})
      AND src.authority = 'agent'
      AND NOT EXISTS (
          SELECT 1 FROM edges pf
          WHERE pf.src_id = e.src_id
            AND pf.edge_type = '{PROMOTED_FROM_EDGE}'
            AND pf.via_cert != ''
      )
    """
    viols = [f"{r['src_id']} --{r['edge_type']}--> {r['dst_id']}"
             for r in conn.execute(q).fetchall()]
    return len(viols) == 0, f"{len(viols)} unpromoted agent causal edge(s)", viols


# --- FE2: via_cert resolves to PASS in ledger within staleness ---

def check_fe2_via_cert_ledger_fresh(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """For every promoted-from edge, via_cert resolves to PASS in _meta_ledger.json
    within 14d AND git_head == HEAD."""
    ledger = _load_ledger()
    head = _current_git_head()
    now = dt.datetime.now(dt.timezone.utc)
    rows = conn.execute(
        f"SELECT src_id, dst_id, via_cert FROM edges WHERE edge_type = '{PROMOTED_FROM_EDGE}'"
    ).fetchall()
    if not rows:
        return True, "0 promoted-from edges — nothing to check", []
    viols: list[str] = []
    for r in rows:
        vc = r["via_cert"]
        entry = ledger.get(vc)
        if entry is None:
            viols.append(f"{r['src_id']}: via_cert={vc!r} not in ledger")
            continue
        if entry.get("status") != "PASS":
            viols.append(f"{r['src_id']}: via_cert={vc!r} status={entry.get('status')!r}")
            continue
        try:
            entry_ts = dt.datetime.fromisoformat(entry["ts"].replace("Z", "+00:00"))
            age = (now - entry_ts).days
        except (KeyError, ValueError):
            viols.append(f"{r['src_id']}: via_cert={vc!r} bad ts")
            continue
        if age > LEDGER_STALENESS_DAYS:
            viols.append(f"{r['src_id']}: via_cert={vc!r} stale ({age}d)")
            continue
        if head != "UNKNOWN" and entry.get("git_head") != head:
            viols.append(
                f"{r['src_id']}: via_cert={vc!r} git_head mismatch "
                f"({entry.get('git_head')!r} != {head!r})"
            )
    return len(viols) == 0, f"{len(viols)} ledger violation(s) across {len(rows)} edge(s)", viols


# --- FE3: No promoted-from cycles ---

def check_fe3_no_promoted_from_cycles(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """DFS on promoted-from edges: no cycles."""
    edges = conn.execute(
        f"SELECT src_id, dst_id FROM edges WHERE edge_type = '{PROMOTED_FROM_EDGE}'"
    ).fetchall()
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["src_id"], []).append(e["dst_id"])
    cycles: list[str] = []
    for start in adj:
        visited: set[str] = set()
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for nxt in adj.get(node, []):
                if nxt == start:
                    cycles.append(" → ".join(path + [nxt]))
                elif nxt not in visited:
                    stack.append((nxt, path + [nxt]))
    return len(cycles) == 0, f"{len(cycles)} cycle(s)", cycles


# --- FE4: Ephemeral test DB — runtime firewall test ---

def check_fe4_ephemeral_firewall_test() -> tuple[bool, str, list[str]]:
    """Spin up tempfile DB with real-shape fixtures, verify:
    1. agent→Cosmos causal WITHOUT promoted-from raises FirewallViolation
    2. agent→Cosmos causal WITH promoted-from succeeds
    3. Non-causal edge always succeeds for agent
    """
    from tools.qa_kg.kg import KG, Node, Edge, FirewallViolation, _now
    from tools.qa_kg.schema import init_db

    viols: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "fe4_test.db"
        conn = init_db(db_path)
        kg = KG(conn)

        # Real-shape fixtures
        kg.upsert_node(Node(
            id="agent:fe4_test", node_type="Thought",
            title="Agent observation for FE4", body="test fixture body",
            authority="agent", epistemic_status="observation",
            method="collab_bus", source_locator="collab:fe4test",
        ))
        kg.upsert_node(Node(
            id="cert:fe4_target", node_type="Cert",
            title="Target cert for FE4", body="certified content here",
            authority="derived", epistemic_status="certified",
            method="cert_validator", source_locator="file:qa_alphageometry_ptolemy/fe4",
        ))
        kg.upsert_node(Node(
            id="rule:fe4_promoter", node_type="Rule",
            title="Promoter rule for FE4", body="internal rule content",
            authority="internal", epistemic_status="interpretation",
            method="memory_rules", source_locator="file:CLAUDE.md",
        ))

        # Test 1: agent causal WITHOUT promoted-from must raise FirewallViolation
        try:
            kg.upsert_edge(Edge(
                src_id="agent:fe4_test", dst_id="cert:fe4_target",
                edge_type="validates", via_cert="cert:225",
            ))
            viols.append("FE4-T1: agent→Cosmos causal did NOT raise FirewallViolation")
        except FirewallViolation:
            pass  # expected

        # Test 2: non-causal edge always succeeds
        try:
            kg.upsert_edge(Edge(
                src_id="agent:fe4_test", dst_id="cert:fe4_target",
                edge_type="keyword-co-occurs",
                confidence=0.3, method="keyword",
            ))
        except FirewallViolation:
            viols.append("FE4-T2: non-causal edge raised FirewallViolation")

        # Test 3: insert promoted-from, then causal edge succeeds
        conn.execute(
            """INSERT INTO edges (src_id, dst_id, edge_type, confidence, method,
                                  provenance, via_cert, created_ts)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("agent:fe4_test", "rule:fe4_promoter", PROMOTED_FROM_EDGE,
             1.0, "promote",
             json.dumps({
                 "session": "fe4-test",
                 "signed_ts": _now(),
                 "promoter_node_id": "rule:fe4_promoter",
                 "broadcast_payload_snapshot": {"ts": _now(), "session": "fe4-test"},
             }),
             "cert:225", _now()),
        )
        conn.commit()
        try:
            kg.upsert_edge(Edge(
                src_id="agent:fe4_test", dst_id="cert:fe4_target",
                edge_type="validates", via_cert="cert:225",
            ))
        except FirewallViolation:
            viols.append("FE4-T3: agent causal WITH promoted-from raised FirewallViolation")

    return len(viols) == 0, f"{len(viols)} ephemeral test failure(s)", viols


# --- FE5: WARN oldest unpromoted agent note ---

def check_fe5_warn_oldest_unpromoted(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """WARN: find agent nodes with no outgoing promoted-from edge.
    Always passes (WARN only)."""
    q = """
    SELECT n.id, n.created_ts FROM nodes n
    WHERE n.authority = 'agent'
      AND NOT EXISTS (
          SELECT 1 FROM edges pf
          WHERE pf.src_id = n.id AND pf.edge_type = ?
      )
    ORDER BY n.created_ts ASC LIMIT 5
    """
    rows = conn.execute(q, (PROMOTED_FROM_EDGE,)).fetchall()
    if not rows:
        return True, "all agent nodes promoted (or no agent nodes)", []
    details = [f"{r['id']} (created {r['created_ts']})" for r in rows]
    return True, f"{len(details)} unpromoted agent note(s) (oldest shown)", details


# --- FE6: promoted-from provenance contains broadcast_payload_snapshot ---

def check_fe6_provenance_snapshot(
    conn: sqlite3.Connection,
) -> tuple[bool, str, list[str]]:
    """Every promoted-from edge must have provenance that JSON-parses
    and contains keys {session, signed_ts, promoter_node_id,
    broadcast_payload_snapshot}. broadcast_payload_snapshot must not be null."""
    rows = conn.execute(
        f"SELECT src_id, dst_id, provenance FROM edges WHERE edge_type = '{PROMOTED_FROM_EDGE}'"
    ).fetchall()
    if not rows:
        return True, "0 promoted-from edges — nothing to check", []
    viols: list[str] = []
    for r in rows:
        prov_raw = r["provenance"]
        if not prov_raw:
            viols.append(f"{r['src_id']}→{r['dst_id']}: empty provenance")
            continue
        try:
            prov = json.loads(prov_raw)
        except json.JSONDecodeError:
            viols.append(f"{r['src_id']}→{r['dst_id']}: provenance is not valid JSON")
            continue
        missing = PROVENANCE_REQUIRED_KEYS - set(prov.keys())
        if missing:
            viols.append(f"{r['src_id']}→{r['dst_id']}: missing keys {sorted(missing)}")
            continue
        if prov.get("broadcast_payload_snapshot") is None:
            viols.append(f"{r['src_id']}→{r['dst_id']}: broadcast_payload_snapshot is null")
    return len(viols) == 0, f"{len(viols)} provenance violation(s) across {len(rows)} edge(s)", viols


# --- Main ---

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

    # FE4 is filesystem-only (ephemeral DB)
    run_bool("FE4", "Ephemeral firewall test", check_fe4_ephemeral_firewall_test, True)

    # DB-dependent checks
    try:
        conn = _connect(db)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 2

    run_bool("FE1", "No unpromoted agent causal edges",
             lambda: check_fe1_no_unpromoted_agent_causal(conn), True)
    run_bool("FE2", "via_cert ledger freshness",
             lambda: check_fe2_via_cert_ledger_fresh(conn), True)
    run_bool("FE3", "No promoted-from cycles",
             lambda: check_fe3_no_promoted_from_cycles(conn), True)
    run_bool("FE5", "Oldest unpromoted agent notes",
             lambda: check_fe5_warn_oldest_unpromoted(conn), False)
    run_bool("FE6", "Promoted-from provenance snapshot",
             lambda: check_fe6_provenance_snapshot(conn), True)

    if hard_fail:
        print("[FAIL] QA-KG firewall effective cert [227] v1")
        return 1
    print("[PASS] QA-KG firewall effective cert [227] v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
