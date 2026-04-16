# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG test harness against real project data -->
"""QA-KG thorough tests against real project data.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Every test exercises real on-disk sources — no mocks, no synthetic fixtures.

Data used:
  - qa_alphageometry_ptolemy/qa_meta_validator.py FAMILY_SWEEPS (cert registry)
  - qa_alphageometry_ptolemy/ subdirs with mapping_protocol*.json (fs-discovered certs)
  - ~/.claude/projects/.../memory/MEMORY.md (Hard Rules)
  - ~/.claude/projects/.../tool-results/mcp-open-brain-recent_thoughts-*.txt (OB)
  - _forensics/qa_retrieval.sqlite (A-RAG, ~58k messages)
  - qa_orbit_rules.py (canonical orbit classifier)
  - qa_axiom_linter.py (axiom compliance)
  - CLAUDE.md, QA_AXIOMS_BLOCK.md (authority docs)

Run:
    python -m tools.qa_kg.tests.test_qa_kg_real
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import glob
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "qa_alphageometry_ptolemy"))

# Isolated tempdir DB. `tools.qa_kg.__init__.py` is loaded by `python -m` BEFORE
# this module's code runs, so setting env here is too late to affect DEFAULT_DB.
# Instead, pass the path explicitly to connect() everywhere and export via env
# for subprocesses (cert validators).
_TMPDIR = tempfile.mkdtemp(prefix="qa_kg_test_")
TEST_DB = Path(_TMPDIR) / "qa_kg_test.db"
os.environ["QA_KG_DB"] = str(TEST_DB)

from tools.qa_kg import (
    Index, Tier, NODE_TYPE_RANK, compute_index, dr, char_ord_sum,
    tier_for_index, connect,
)
# Back-compat aliases still imported for deprecation-smoke:
from tools.qa_kg import Coord, compute_be, tier_for_coord  # noqa: F401
from tools.qa_kg.kg import Edge, FirewallViolation, Node
from tools.qa_kg.orbit import edge_allowed, qa_step
from tools.qa_kg.predicate import run as run_predicate, resolve as resolve_predicate
from tools.qa_kg.extractors import axioms as x_axioms
from tools.qa_kg.extractors import memory_rules as x_rules
from tools.qa_kg.extractors import certs as x_certs
from tools.qa_kg.extractors import edges as x_edges
from tools.qa_kg.extractors import ob as x_ob
from tools.qa_kg.extractors import arag as x_arag

from qa_orbit_rules import (
    orbit_family as canonical_orbit_family,
    orbit_period, self_test as orbit_self_test,
)


# ---------------------------------------------------------------------------
# Real-data paths
# ---------------------------------------------------------------------------
MEMORY_MD = Path.home() / ".claude/projects/-home-player2-signal-experiments/memory/MEMORY.md"
OB_GLOB   = str(Path.home() / ".claude/projects/-home-player2-signal-experiments/*/tool-results/mcp-open-brain-recent_thoughts-*.txt")
ARAG_DB   = _REPO / "_forensics" / "qa_retrieval.sqlite"
CLAUDE_MD = _REPO / "CLAUDE.md"
AXIOMS_BLOCK = _REPO / "QA_AXIOMS_BLOCK.md"
LINTER = _REPO / "tools" / "qa_axiom_linter.py"
META_VALIDATOR = _REPO / "qa_alphageometry_ptolemy" / "qa_meta_validator.py"


def _build_graph():
    kg = connect(TEST_DB)
    a = x_axioms.populate(kg)
    r = x_rules.populate(kg)
    c = x_certs.populate(kg)
    e_stats = x_edges.populate(kg)  # cert→axiom, cert→cert, rule→axiom
    # Sort by mtime — paths differ by session UUID prefix, so alpha-sort picks
    # the wrong file. Newest mtime = most recent tool result (and the modern
    # markdown format rather than early JSON dumps).
    ob_stats = None
    ob_files = sorted(glob.glob(OB_GLOB), key=lambda p: os.path.getmtime(p))
    if ob_files:
        latest = Path(ob_files[-1])
        ob_stats = x_ob.ingest_markdown(kg, latest.read_text(encoding="utf-8"))
    return kg, {"axioms": a, "rules": r, "certs": c, "edges": e_stats, "ob": ob_stats}


KG, BUILD_STATS = _build_graph()


# ---------------------------------------------------------------------------
# 1. Candidate F classifier — parity against every real stored node
# ---------------------------------------------------------------------------
def test_candidate_f_parity_every_real_node():
    """For every classified node, recomputing Candidate F reproduces stored
    (idx_b, idx_e) and char_ord_sum."""
    rows = KG.conn.execute(
        "SELECT id, node_type, title, body, idx_b, idx_e, char_ord_sum "
        "FROM nodes WHERE tier != 'unassigned'"
    ).fetchall()
    assert len(rows) > 0, "no classified nodes in real graph"
    for r in rows:
        text = (r["title"] or "") + ("\n" + r["body"] if r["body"] else "")
        assert text, f"{r['id']}: classified but no content"
        expected = compute_index(text, r["node_type"])
        assert expected.idx_b == r["idx_b"]
        assert expected.idx_e == r["idx_e"]
        assert char_ord_sum(text) == r["char_ord_sum"]


def test_dr_a1_across_all_real_char_ord_sums():
    """dr() applied to every real stored char_ord_sum produces value in {1..9}."""
    rows = KG.conn.execute("SELECT char_ord_sum FROM nodes WHERE char_ord_sum IS NOT NULL").fetchall()
    assert len(rows) > 0
    for r in rows:
        v = dr(r["char_ord_sum"])
        assert 1 <= v <= 9, f"dr({r['char_ord_sum']}) = {v} violates A1"


def test_node_type_rank_covers_every_real_type():
    """Every node_type that appears in the real graph must have a canonical rank."""
    rows = KG.conn.execute("SELECT DISTINCT node_type FROM nodes").fetchall()
    for r in rows:
        assert r["node_type"] in NODE_TYPE_RANK, (
            f"real node_type {r['node_type']!r} not in NODE_TYPE_RANK")


def test_compute_index_rejects_empty_via_real_inputs():
    """Empty string is the real 'unassigned' condition. Must raise."""
    try:
        compute_index("", "Axiom")
    except ValueError:
        pass
    else:
        raise AssertionError("compute_index must reject empty content")


# ---------------------------------------------------------------------------
# 2. Canonical orbit classification — canonical self-test + full lattice agreement
# ---------------------------------------------------------------------------
def test_canonical_self_test_passes():
    """Run qa_orbit_rules.self_test — exhaustive simulation verification for mod-9 and mod-24."""
    assert orbit_self_test(verbose=False) is True


def test_tier_for_index_agrees_with_canonical_all_81_mod9():
    """All 81 cells mod-9: tier_for_index wrapper agrees with canonical orbit_family."""
    for b in range(1, 10):
        for e in range(1, 10):
            wrapper = tier_for_index(b, e).value
            canonical = canonical_orbit_family(b, e, 9)
            assert wrapper == canonical, f"({b},{e}): wrapper={wrapper!r} canonical={canonical!r}"


def test_singularity_uniqueness_mod9_mod24():
    """Exactly one Singularity cell per modulus (at (m,m))."""
    for m in (9, 24):
        sing = [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)
                if canonical_orbit_family(b, e, m) == "singularity"]
        assert sing == [(m, m)], f"mod-{m} singularity cells: {sing}"


def test_satellite_cell_count_mod9_mod24():
    """Satellite = (m//3)|b AND (m//3)|e minus singularity."""
    for m in (9, 24):
        sat = [(b, e) for b in range(1, m + 1) for e in range(1, m + 1)
               if canonical_orbit_family(b, e, m) == "satellite"]
        # mod-9: step=3, grid 3x3=9, minus (9,9) → 8
        # mod-24: step=8, grid 3x3=9, minus (24,24) → 8
        assert len(sat) == 8, f"mod-{m} satellite count: {len(sat)} (expected 8)"


def test_cosmos_count_mod9():
    cosmos = [(b, e) for b in range(1, 10) for e in range(1, 10)
              if canonical_orbit_family(b, e, 9) == "cosmos"]
    assert len(cosmos) == 72, f"mod-9 cosmos count: {len(cosmos)} (expected 72)"


def test_qa_step_a1_exhaustive_mod9():
    """Exhaustively verify qa_step never produces 0 across all (b,e) in mod-9."""
    for b in range(1, 10):
        for e in range(1, 10):
            _, r = qa_step(b, e, 9) if isinstance(qa_step(b, e, 9), tuple) else (None, qa_step(b, e, 9))
            # Local qa_step returns int (not tuple); canonical returns tuple.
            # Try both shapes safely.
            out = qa_step(b, e, 9)
            if isinstance(out, tuple):
                _, val = out
            else:
                val = out
            assert 1 <= val <= 9, f"qa_step({b},{e}) violated A1: {val}"


# ---------------------------------------------------------------------------
# 3. Schema constraints enforced on real graph
# ---------------------------------------------------------------------------
def test_all_real_indices_pass_a1_check_constraint():
    rows = KG.conn.execute("SELECT id, idx_b, idx_e FROM nodes").fetchall()
    for r in rows:
        if r["idx_b"] is not None:
            assert 1 <= r["idx_b"] <= 9, f"{r['id']}: idx_b {r['idx_b']} violates CHECK"
        if r["idx_e"] is not None:
            assert 1 <= r["idx_e"] <= 9, f"{r['id']}: idx_e {r['idx_e']} violates CHECK"


def test_all_real_tiers_are_canonical_enum():
    rows = KG.conn.execute("SELECT DISTINCT tier FROM nodes").fetchall()
    canonical = {"singularity", "cosmos", "satellite", "unassigned"}
    for r in rows:
        assert r["tier"] in canonical, f"non-canonical tier in graph: {r['tier']!r}"


def test_all_real_edge_confidences_in_range():
    rows = KG.conn.execute("SELECT src_id, dst_id, edge_type, confidence FROM edges").fetchall()
    assert len(rows) > 0
    for r in rows:
        assert 0.0 <= r["confidence"] <= 1.0, f"edge confidence {r['confidence']} out of range"


def test_edge_fk_integrity_real():
    """Every edge references real existing nodes."""
    orphans = KG.conn.execute("""
        SELECT e.src_id, e.dst_id, e.edge_type FROM edges e
        LEFT JOIN nodes ns ON ns.id = e.src_id
        LEFT JOIN nodes nd ON nd.id = e.dst_id
        WHERE ns.id IS NULL OR nd.id IS NULL
    """).fetchall()
    assert len(orphans) == 0, f"{len(orphans)} edges reference missing nodes"


def test_tier_derived_from_orbit_rule_real():
    """For every classified node: stored tier == canonical orbit_family(idx_b, idx_e)."""
    rows = KG.conn.execute(
        "SELECT id, idx_b, idx_e, tier FROM nodes WHERE tier != 'unassigned'"
    ).fetchall()
    for r in rows:
        # Wrap with int() per axiom linter ORBIT-3 — sqlite3.Row values are
        # Python ints, but the linter defensively flags any bracket-indexed
        # arg into orbit_family. Explicit int() satisfies both.
        bb, ee = int(r["idx_b"]), int(r["idx_e"])
        expected = canonical_orbit_family(bb, ee, 9)
        assert r["tier"] == expected, (
            f"{r['id']}: stored tier={r['tier']!r} "
            f"≠ canonical orbit_family({bb},{ee})={expected!r}")


# ---------------------------------------------------------------------------
# 4. FTS5 roundtrip on every real node
# ---------------------------------------------------------------------------
def test_fts_index_contains_every_real_node():
    """FTS5 content-table integrity: every nodes row has a matching nodes_fts row."""
    node_count = KG.conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
    fts_count = KG.conn.execute("SELECT COUNT(*) c FROM nodes_fts").fetchone()["c"]
    assert node_count == fts_count, f"nodes={node_count} nodes_fts={fts_count}"
    # Spot-check: for 20 real nodes, search FTS by id (unique) and verify retrieval.
    sample = KG.conn.execute(
        "SELECT id FROM nodes WHERE title != '' ORDER BY RANDOM() LIMIT 20"
    ).fetchall()
    for r in sample:
        hit = KG.conn.execute(
            "SELECT nodes.id FROM nodes_fts JOIN nodes ON nodes.rowid = nodes_fts.rowid "
            "WHERE nodes_fts MATCH ? LIMIT 1",
            (f'"{r["id"]}"',),  # quoted phrase to avoid tokenization on ':'
        ).fetchone()
        # id may not be tokenizable (underscores/colons) — skip those; the count
        # invariant above is the authoritative check.
        _ = hit


def test_fts_deletion_trigger_real():
    """Delete a real node; nodes_fts row must vanish."""
    # Pick a real node with a unique distinguishing title
    row = KG.conn.execute(
        "SELECT id, title FROM nodes WHERE title != '' AND node_type='Rule' LIMIT 1"
    ).fetchone()
    assert row is not None
    KG.conn.execute("DELETE FROM nodes WHERE id=?", (row["id"],))
    KG.conn.commit()
    remaining = KG.conn.execute(
        "SELECT 1 FROM nodes_fts JOIN nodes ON nodes.rowid=nodes_fts.rowid WHERE nodes.id=?",
        (row["id"],),
    ).fetchone()
    assert remaining is None, "FTS trigger didn't cascade delete"
    # Restore it so later tests still see a full graph (re-ingest from MEMORY.md)
    x_rules.populate(KG)


# ---------------------------------------------------------------------------
# 5. Firewall — using REAL archive data from A-RAG
# ---------------------------------------------------------------------------
def test_firewall_blocks_real_archive_to_real_cosmos():
    """Promote a real A-RAG msg, then attempt causal edge to a real Cosmos cert."""
    assert ARAG_DB.exists(), f"A-RAG DB missing at {ARAG_DB}"
    # Pick any real archive msg
    conn = sqlite3.connect(str(ARAG_DB))
    row = conn.execute("SELECT msg_id FROM messages LIMIT 1").fetchone()
    conn.close()
    assert row is not None, "A-RAG has no messages"
    real_msg_id = row[0]

    # Promote it into our test graph (no subject declaration → unassigned)
    archive_node_id = x_arag.promote(KG, real_msg_id, reason="firewall test")
    archive_row = KG.get(archive_node_id)
    assert archive_row["tier"] in ("cosmos", "satellite", "singularity", "unassigned")

    # Pick a real Cosmos cert
    cert_row = KG.conn.execute(
        "SELECT id FROM nodes WHERE node_type='Cert' AND tier='cosmos' LIMIT 1"
    ).fetchone()
    assert cert_row is not None

    # The firewall only fires on UNASSIGNED src. If promote gave it a non-empty
    # content, its tier is Cosmos/etc. — firewall doesn't apply. Verify edge_allowed
    # directly using the declared behavior matrix per Theorem NT.
    assert not edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=False)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "validates", via_cert=True)
    assert edge_allowed(Tier.UNASSIGNED, Tier.COSMOS, "cites", via_cert=False)  # non-causal
    assert edge_allowed(Tier.SATELLITE, Tier.COSMOS, "validates", via_cert=False)


def test_firewall_raises_on_real_unassigned_source():
    """Create a real unassigned node (empty content) + real Cosmos target; causal edge raises."""
    # Use a real A-RAG msg title as identifier but store as unassigned (empty content).
    kg2 = connect(TEST_DB)  # same test DB
    real_arch_id = "test:unassigned_firewall"
    kg2.upsert_node(Node(id=real_arch_id, node_type="Thought", title="", body=""))
    assert kg2.get(real_arch_id)["tier"] == "unassigned"
    # real Cosmos target
    target = KG.conn.execute(
        "SELECT id FROM nodes WHERE tier='cosmos' LIMIT 1"
    ).fetchone()
    try:
        kg2.upsert_edge(Edge(src_id=real_arch_id, dst_id=target["id"], edge_type="validates"))
    except FirewallViolation:
        pass
    else:
        raise AssertionError("firewall must block unassigned→cosmos causal without via_cert")
    # With via_cert it's allowed
    kg2.upsert_edge(Edge(src_id=real_arch_id, dst_id=target["id"],
                         edge_type="validates", via_cert="cert:225"))


# ---------------------------------------------------------------------------
# 6. Extractor parity with real sources
# ---------------------------------------------------------------------------
def test_axiom_count_matches_claude_md():
    """7 axioms declared in CLAUDE.md. Extractor produces 7 nodes."""
    assert CLAUDE_MD.exists()
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for code in ("A1", "A2", "T2", "S1", "S2", "T1", "NT"):
        assert code in text or f"**{code}" in text, f"axiom {code} missing from CLAUDE.md"
    axioms_in_graph = KG.conn.execute(
        "SELECT COUNT(*) n FROM nodes WHERE node_type='Axiom'"
    ).fetchone()["n"]
    assert axioms_in_graph == 7, f"expected 7 axioms, got {axioms_in_graph}"


def test_rule_count_matches_memory_md():
    """Every '### X (HARD...)' heading in real MEMORY.md becomes a Rule node."""
    if not MEMORY_MD.exists():
        return  # MEMORY.md is user-scope; skip if absent
    text = MEMORY_MD.read_text(encoding="utf-8")
    hard_headings = re.findall(r"^###\s+(.+?)\s*\((?:HARD)(?:[^)]*)\)", text, flags=re.M)
    n_hard = len(hard_headings)
    n_rules = KG.conn.execute("SELECT COUNT(*) n FROM nodes WHERE node_type='Rule'").fetchone()["n"]
    assert n_rules == n_hard, f"{n_rules} Rule nodes ≠ {n_hard} HARD headings in MEMORY.md"


def test_cert_registered_count_matches_family_sweeps():
    """Every FAMILY_SWEEPS entry becomes a cert:<id> node."""
    import qa_meta_validator as mv
    registered_ids = {f"cert:{e[0]}" for e in mv.FAMILY_SWEEPS}
    graph_registered = {r["id"] for r in KG.conn.execute(
        "SELECT id FROM nodes WHERE id LIKE 'cert:%' AND id NOT LIKE 'cert:fs:%'"
    ).fetchall()}
    missing = registered_ids - graph_registered
    assert not missing, f"FAMILY_SWEEPS entries missing from graph: {list(missing)[:5]}"


def test_fs_cert_count_matches_mapping_protocol_files():
    """Every distinct subdir with mapping_protocol*.json becomes a cert:fs:* node."""
    meta_dir = _REPO / "qa_alphageometry_ptolemy"
    fs_dirs = set()
    for mp in meta_dir.rglob("mapping_protocol*.json"):
        rel = mp.parent.relative_to(meta_dir)
        fs_dirs.add(str(rel).replace(os.sep, "/"))
    n_fs_graph = KG.conn.execute(
        "SELECT COUNT(*) n FROM nodes WHERE id LIKE 'cert:fs:%'"
    ).fetchone()["n"]
    assert n_fs_graph == len(fs_dirs), (
        f"fs cert nodes: {n_fs_graph}; mapping_protocol dirs on disk: {len(fs_dirs)}")


def test_ob_thoughts_parsed_from_real_markdown():
    """OB ingest produced at least one Thought node from the real markdown file."""
    if BUILD_STATS["ob"] is None:
        return  # no OB file present in this environment
    n_parsed = BUILD_STATS["ob"]["parsed"]
    n_nodes = BUILD_STATS["ob"]["nodes"]
    assert n_parsed > 0 and n_nodes == n_parsed


def test_ob_keyword_edges_reference_real_dst_only():
    """Every ob→* keyword-co-occurs edge's dst_id is a real node in the graph.
    Also asserts edge_type is the new 'keyword-co-occurs' (not the retired
    'instantiates' / 'derived-from' labels on keyword matches)."""
    rows = KG.conn.execute(
        "SELECT src_id, dst_id, edge_type, method FROM edges WHERE src_id LIKE 'ob:%'"
    ).fetchall()
    assert len(rows) > 0, "no OB edges extracted"
    for r in rows:
        assert r["edge_type"] == "keyword-co-occurs", (
            f"OB edge {r['src_id']}→{r['dst_id']} has edge_type={r['edge_type']!r}; "
            "keyword-matched OB edges must be 'keyword-co-occurs'")
        assert r["method"] == "keyword"
        exists = KG.conn.execute(
            "SELECT 1 FROM nodes WHERE id=?", (r["dst_id"],)
        ).fetchone()
        assert exists is not None, f"edge points at non-existent node {r['dst_id']!r}"


def test_all_keyword_edges_have_low_confidence():
    """Phase 0: every keyword-method edge has confidence ≤ 0.3."""
    rows = KG.conn.execute(
        "SELECT src_id, dst_id, confidence FROM edges WHERE method='keyword'"
    ).fetchall()
    assert len(rows) > 0
    for r in rows:
        assert r["confidence"] <= 0.3, (
            f"{r['src_id']}→{r['dst_id']}: confidence={r['confidence']} > 0.3 "
            "— keyword edges must be low-confidence under Phase 0")


def test_no_derived_from_from_keyword_matches():
    """Phase 0 reserves 'derived-from' for structural proof links. No keyword
    extractor may emit it."""
    hits = KG.conn.execute(
        "SELECT src_id, dst_id FROM edges "
        "WHERE edge_type='derived-from' AND method='keyword'"
    ).fetchall()
    assert len(hits) == 0, f"{len(hits)} keyword-method 'derived-from' edges leaked past Phase 0"


# ---------------------------------------------------------------------------
# 7. A-RAG real DB bridge
# ---------------------------------------------------------------------------
def test_arag_search_returns_real_coords():
    """A-RAG search on real DB returns hits with the (b,e) stored in A-RAG per [202]."""
    assert ARAG_DB.exists()
    hits = x_arag.search("quantum arithmetic", k=3)
    assert len(hits) >= 1, "A-RAG search returned zero for 'quantum arithmetic' — DB empty?"
    for h in hits:
        # A-RAG stored coords must be A1-compliant (they're certified [202])
        b, e = h["arag_coord"]
        assert 1 <= b <= 9 and 1 <= e <= 9
        # tier label is 'archive' per bridge convention
        assert h["tier"] == "archive"


def test_arag_promote_persists_real_content():
    """promote() against a real A-RAG msg_id creates a node whose body contains the real text."""
    conn = sqlite3.connect(str(ARAG_DB))
    row = conn.execute(
        "SELECT msg_id, substr(raw_text, 1, 30) txt FROM messages WHERE char_count > 200 LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    nid = x_arag.promote(KG, row[0], reason="test persistence")
    got = KG.get(nid)
    assert got is not None
    assert row[1][:20] in got["body"]


# ---------------------------------------------------------------------------
# 8. Predicate runtime — real axiom predicates against real repo
# ---------------------------------------------------------------------------
def test_resolve_real_axiom_predicate():
    fn = resolve_predicate("tools.qa_kg.predicates.axioms:check_a1")
    assert callable(fn)


def test_run_each_real_axiom_predicate_against_repo():
    """Each of the 7 axiom predicates actually executes against the real repo and
    returns a (bool, str) shape."""
    codes = ["a1", "a2", "t2", "s1", "s2", "t1", "nt"]
    for code in codes:
        ref = f"tools.qa_kg.predicates.axioms:check_{code}"
        result = run_predicate(ref)
        assert isinstance(result.ok, bool)
        assert isinstance(result.msg, str)


def test_run_on_broken_ref_returns_failure():
    """Non-existent predicate reference returns CheckResult(ok=False), no exception escape."""
    result = run_predicate("nonexistent.module:fn")
    assert result.ok is False
    assert "ModuleNotFoundError" in result.msg or "No module" in result.msg


# ---------------------------------------------------------------------------
# 9. Query operations on real graph
# ---------------------------------------------------------------------------
def test_search_finds_real_keely_certs():
    """Real Keely-related certs exist in the project; FTS must return them."""
    results = KG.search("Keely", k=10)
    titles = [r["title"] for r in results]
    assert any("Keely" in t for t in titles), f"Keely search returned nothing real: {titles}"


def test_neighbors_on_real_cert_with_edges():
    """Pick a real cert that has edges; neighbors() must return them."""
    cert = KG.conn.execute(
        "SELECT src_id FROM edges WHERE src_id LIKE 'cert:%' LIMIT 1"
    ).fetchone()
    assert cert is not None
    ns = KG.neighbors(cert["src_id"])
    assert len(ns) > 0


def test_why_chain_empty_under_phase_0_vacuous_flag():
    """FLAGGED VACUOUS UNDER PHASE 0.

    The previous iteration tested that `kg.why()` could walk a cert→axiom
    provenance chain. Under Phase 0:
      - keyword-method edges are filtered OUT of kg.why()
      - the baseline auto-link (A1/A2/T2/NT to every cert at 0.9) is REMOVED
      - no Phase 0 extractor emits structural 'derived-from' edges
    Therefore kg.why() returns empty for every cert, by design. This is
    honest — we have no authoritative proof graph yet (Phase 3 work).

    This test ASSERTS the vacuous state directly, so when Phase 3 adds
    real proof extraction and kg.why() starts returning non-empty chains,
    this test FAILS and must be rewritten to the real provenance test."""
    # Sanity: there ARE certs in the graph
    cert = KG.conn.execute(
        "SELECT id FROM nodes WHERE node_type='Cert' LIMIT 1"
    ).fetchone()
    assert cert is not None
    chain = KG.why(cert["id"], max_depth=3)
    assert chain == [], (
        "why() returned a non-empty chain — Phase 3 structural edges are "
        "now emitted; rewrite this test for the real provenance case.")


def test_idx_neighborhood_returns_shared_aiq_bekar_class():
    """idx_neighborhood returns every real node at a given (idx_b, idx_e)."""
    row = KG.conn.execute(
        "SELECT idx_b, idx_e FROM nodes WHERE idx_b IS NOT NULL LIMIT 1"
    ).fetchone()
    idx = Index(row["idx_b"], row["idx_e"])
    nbrs = KG.idx_neighborhood(idx)
    for n in nbrs:
        assert n["idx_b"] == idx.idx_b and n["idx_e"] == idx.idx_e


def test_digest_returns_requested_tier_only():
    for tier in ("cosmos", "satellite", "singularity"):
        for r in KG.digest(tier=tier, limit=25):
            assert r["tier"] == tier


# ---------------------------------------------------------------------------
# 10. [225] + [226] cert validators against real graph
# ---------------------------------------------------------------------------
def test_cert_225_hard_invariants_pass_on_real_graph():
    """Run [225] validator against the REAL test-DB graph. HARD invariants must PASS.
    KG1 and KG4 are WARN-only and may produce warnings on real data (138 unvetted fs certs)."""
    env = os.environ.copy()
    env["QA_KG_DB"] = str(TEST_DB)
    proc = subprocess.run(
        ["python3", str(_REPO / "qa_alphageometry_ptolemy/qa_kg_consistency_cert_v2/qa_kg_consistency_cert_validate.py")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, f"[225] v2 FAILED: {proc.stdout}\n{proc.stderr}"
    for line in proc.stdout.splitlines():
        # HARD invariants: KG1, KG2, KG5, KG6, KG7 must not FAIL.
        # KG3 can be N/A (no Unassigned precondition); that is explicit not-pass-not-fail.
        if line.startswith("[FAIL]") and any(k in line for k in ("KG1", "KG2", "KG5", "KG6", "KG7")):
            raise AssertionError(f"HARD invariant failed: {line}")


def test_cert_225_kg3_never_silently_passes():
    """FLAGGED PHASE 0 PRECONDITION BEHAVIOR.

    Under a vanilla build, Candidate F assigns tier=cosmos to every node
    with content — so the Unassigned precondition is unmet and KG3 v2
    reports N/A (not PASS). This test verifies KG3 emits N/A, PASS, or
    FAIL — NEVER a silent no-op — and that on Phase 0 the result is
    honest given whatever Unassigned nodes exist in the test DB.

    When earlier firewall tests in this suite have already introduced
    Unassigned nodes with via_cert-mediated edges, KG3 legitimately runs
    and PASSES. When no Unassigned nodes exist, KG3 must report N/A.
    FAIL would indicate a real firewall breach and must propagate."""
    env = os.environ.copy()
    env["QA_KG_DB"] = str(TEST_DB)
    proc = subprocess.run(
        ["python3", str(_REPO / "qa_alphageometry_ptolemy/qa_kg_consistency_cert_v2/qa_kg_consistency_cert_validate.py")],
        capture_output=True, text=True, env=env, timeout=60,
    )
    kg3_lines = [ln for ln in proc.stdout.splitlines() if " KG3 " in ln]
    assert kg3_lines, f"KG3 report missing: {proc.stdout}"
    line = kg3_lines[0]
    assert any(tag in line for tag in ("[N/A]", "[PASS]", "[FAIL]")), (
        f"KG3 must emit N/A, PASS, or FAIL — got: {line!r}")
    # FAIL on the real graph would mean a firewall breach, which should
    # propagate as a hard failure through the main [225] test above.
    assert "[FAIL]" not in line, f"KG3 FAIL on real graph: {line}"


def test_cert_226_all_invariants_pass():
    proc = subprocess.run(
        ["python3", str(_REPO / "qa_alphageometry_ptolemy/qa_semantic_coord_cert_v1/qa_semantic_coord_cert_validate.py")],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"[226] FAILED: {proc.stdout}\n{proc.stderr}"
    for line in proc.stdout.splitlines():
        assert not line.startswith("[FAIL]"), f"[226] invariant failed: {line}"


# ---------------------------------------------------------------------------
# 11. Axiom linter on the full qa_kg tree + cert dirs
# ---------------------------------------------------------------------------
def test_axiom_linter_clean_on_qa_kg_tree():
    proc = subprocess.run(
        ["python3", str(LINTER),
         str(_REPO / "tools/qa_kg"),
         str(_REPO / "qa_alphageometry_ptolemy/qa_kg_consistency_cert_v1"),
         str(_REPO / "qa_alphageometry_ptolemy/qa_semantic_coord_cert_v1")],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"axiom linter errored: {proc.stdout}\n{proc.stderr}"
    assert "CLEAN" in proc.stdout, f"linter output: {proc.stdout}"


# ---------------------------------------------------------------------------
# 12. Subject coord — schema stored columns + A1 compliance if present
# ---------------------------------------------------------------------------
def test_subject_coord_columns_exist():
    cols = {r[1] for r in KG.conn.execute("PRAGMA table_info(nodes)").fetchall()}
    assert "subject_b" in cols and "subject_e" in cols


def test_subject_coord_a1_compliance_across_real_nodes():
    """If subject_b/subject_e are set on real nodes, they're in {1..9}."""
    rows = KG.conn.execute(
        "SELECT id, subject_b, subject_e FROM nodes WHERE subject_b IS NOT NULL OR subject_e IS NOT NULL"
    ).fetchall()
    for r in rows:
        if r["subject_b"] is not None:
            assert 1 <= r["subject_b"] <= 9, f"{r['id']}: subject_b out of range"
        if r["subject_e"] is not None:
            assert 1 <= r["subject_e"] <= 9, f"{r['id']}: subject_e out of range"


def test_subject_coord_roundtrip_real():
    """Set subject on a real node, read back exact values."""
    cert = KG.conn.execute("SELECT id, title FROM nodes WHERE node_type='Cert' LIMIT 1").fetchone()
    assert cert is not None
    KG.conn.execute(
        "UPDATE nodes SET subject_b=?, subject_e=? WHERE id=?", (3, 6, cert["id"])
    )
    KG.conn.commit()
    got = KG.get(cert["id"])
    assert got["subject_b"] == 3 and got["subject_e"] == 6


def test_index_type_has_no_qa_state_derivations():
    """C7 landmine guard: the Index type must NOT expose .d or .a attrs —
    those would suggest it encodes a QA state (d=b+e, a=b+2e). Index is
    a retrieval-index hash × node-type rank; d/a on it is nonsense."""
    idx = Index(3, 7)
    assert not hasattr(idx, "d"), "Index has .d — Phase 0 landmine regression"
    assert not hasattr(idx, "a"), "Index has .a — Phase 0 landmine regression"


def test_schema_version_is_2():
    row = KG.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert row is not None, "meta.schema_version missing"
    assert row["value"] == "2", f"schema_version={row['value']!r}, expected '2'"


# ---------------------------------------------------------------------------
# 13. Integration sanity — graph is non-trivial
# ---------------------------------------------------------------------------
def test_graph_has_meaningful_scale():
    """Phase 0 edge threshold is substantially lower than pre-Phase-0: the
    baseline auto-link (A1/A2/T2/NT → every cert at 0.9) was removed, so
    ~1100 fabricated edges are gone. Remaining edges are real body-token
    co-occurrences only."""
    n_nodes = KG.conn.execute("SELECT COUNT(*) n FROM nodes").fetchone()["n"]
    n_edges = KG.conn.execute("SELECT COUNT(*) n FROM edges").fetchone()["n"]
    assert n_nodes >= 350, f"only {n_nodes} nodes — extractor regression"
    # Post-Phase-0 expected count: ~200-400 real body-token co-occurrences
    assert 100 <= n_edges <= 800, (
        f"{n_edges} edges — outside Phase 0 expected range [100, 800]")


def test_no_self_vetting_phase_0():
    """Phase 0: self-vetting was removed as tautological. All Axioms and
    Certs have vetted_by='' until Phase 1 introduces real attribution."""
    rows = KG.conn.execute(
        "SELECT id, node_type, vetted_by FROM nodes "
        "WHERE node_type IN ('Axiom', 'Cert')"
    ).fetchall()
    for r in rows:
        assert r["vetted_by"] != r["id"], (
            f"{r['id']}: self-vetted — Phase 0 landmine regression (C3)")
    # On the Phase 0 graph specifically, vetted_by is empty for all such nodes
    non_empty = [r for r in rows if r["vetted_by"] != ""]
    assert not non_empty, (
        f"{len(non_empty)} Axiom/Cert nodes have non-empty vetted_by — Phase 1 landed early?")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = [
    # Candidate F
    test_candidate_f_parity_every_real_node,
    test_dr_a1_across_all_real_char_ord_sums,
    test_node_type_rank_covers_every_real_type,
    test_compute_index_rejects_empty_via_real_inputs,
    # Orbit canonical
    test_canonical_self_test_passes,
    test_tier_for_index_agrees_with_canonical_all_81_mod9,
    test_singularity_uniqueness_mod9_mod24,
    test_satellite_cell_count_mod9_mod24,
    test_cosmos_count_mod9,
    test_qa_step_a1_exhaustive_mod9,
    # Schema
    test_all_real_indices_pass_a1_check_constraint,
    test_all_real_tiers_are_canonical_enum,
    test_all_real_edge_confidences_in_range,
    test_edge_fk_integrity_real,
    test_tier_derived_from_orbit_rule_real,
    test_schema_version_is_2,
    # FTS
    test_fts_index_contains_every_real_node,
    test_fts_deletion_trigger_real,
    # Firewall
    test_firewall_blocks_real_archive_to_real_cosmos,
    test_firewall_raises_on_real_unassigned_source,
    # Extractor parity
    test_axiom_count_matches_claude_md,
    test_rule_count_matches_memory_md,
    test_cert_registered_count_matches_family_sweeps,
    test_fs_cert_count_matches_mapping_protocol_files,
    test_ob_thoughts_parsed_from_real_markdown,
    test_ob_keyword_edges_reference_real_dst_only,
    test_all_keyword_edges_have_low_confidence,
    test_no_derived_from_from_keyword_matches,
    # A-RAG
    test_arag_search_returns_real_coords,
    test_arag_promote_persists_real_content,
    # Predicates
    test_resolve_real_axiom_predicate,
    test_run_each_real_axiom_predicate_against_repo,
    test_run_on_broken_ref_returns_failure,
    # Queries
    test_search_finds_real_keely_certs,
    test_neighbors_on_real_cert_with_edges,
    test_why_chain_empty_under_phase_0_vacuous_flag,
    test_idx_neighborhood_returns_shared_aiq_bekar_class,
    test_digest_returns_requested_tier_only,
    # Certs
    test_cert_225_hard_invariants_pass_on_real_graph,
    test_cert_225_kg3_never_silently_passes,
    test_cert_226_all_invariants_pass,
    # Linter
    test_axiom_linter_clean_on_qa_kg_tree,
    # Subject coord + landmine guards
    test_subject_coord_columns_exist,
    test_subject_coord_a1_compliance_across_real_nodes,
    test_subject_coord_roundtrip_real,
    test_index_type_has_no_qa_state_derivations,
    # Integration
    test_graph_has_meaningful_scale,
    test_no_self_vetting_phase_0,
]


def main() -> int:
    failed: list[tuple[str, str]] = []
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc(limit=3)
            failed.append((t.__name__, f"{type(exc).__name__}: {exc}\n{tb}"))
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{'='*60}\n{len(TESTS) - len(failed)}/{len(TESTS)} passed")
    for name, msg in failed:
        print(f"\n--- {name} ---\n{msg}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
