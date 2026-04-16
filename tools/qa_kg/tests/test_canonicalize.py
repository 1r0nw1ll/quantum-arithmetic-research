# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 5 test module; grounds in tools/qa_kg/canonicalize.py (Dale, 2026), docs/specs/QA_MEM_SCOPE.md (Dale, 2026) -->
"""Unit tests for tools/qa_kg/canonicalize.py — Phase 5 determinism.

QA_COMPLIANCE = "memory_infra_test — exercises canonicalization pure function"

Covers:
  - excluded columns are absent from output
  - node/edge/meta ordering is deterministic
  - unicode NFC normalization
  - empty graph produces empty canonical bytes
  - graph_hash is stable across two in-process rebuilds
  - fixture_hash depends ONLY on files[], not metadata (C#4)
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra_test — exercises canonicalization pure function"

import json
import unicodedata

import pytest

from tools.qa_kg.schema import init_db
from tools.qa_kg.kg import KG, Node, Edge
from tools.qa_kg.canonicalize import (
    EXCLUDED_NODE_COLS,
    EXCLUDED_EDGE_COLS,
    META_WHITELIST,
    canonicalize_nodes,
    canonicalize_edges,
    canonicalize_meta,
    canonicalize_graph,
    graph_hash,
    compute_fixture_hash,
)


@pytest.fixture
def empty_kg():
    conn = init_db(":memory:")
    yield KG(conn)
    conn.close()


def _seed_graph(kg: KG) -> None:
    """Minimal graph with 2 nodes + 1 structural edge."""
    kg.upsert_node(Node(
        id="axiom:A1", node_type="Axiom",
        title="A1 — No-Zero", body="States in {1..N}.",
        authority="primary", epistemic_status="axiom",
        method="axioms_block", source_locator="file:QA_AXIOMS_BLOCK.md#A1",
    ))
    kg.upsert_node(Node(
        id="cert:test", node_type="Cert",
        title="test cert", body="exercises canonical edges",
        authority="derived", epistemic_status="certified",
        method="cert_validator", source_locator="file:test",
    ))
    kg.upsert_edge(Edge(
        src_id="cert:test", dst_id="axiom:A1",
        edge_type="validates",
        method="structural", provenance="test seed",
    ))


def test_excluded_node_columns_absent(empty_kg):
    _seed_graph(empty_kg)
    nodes_bytes = canonicalize_nodes(empty_kg.conn)
    for line in nodes_bytes.decode("utf-8").splitlines():
        rec = json.loads(line)
        for col in EXCLUDED_NODE_COLS:
            assert col not in rec, f"excluded column {col} leaked into canonical output"


def test_excluded_edge_columns_absent(empty_kg):
    _seed_graph(empty_kg)
    edges_bytes = canonicalize_edges(empty_kg.conn)
    for line in edges_bytes.decode("utf-8").splitlines():
        rec = json.loads(line)
        for col in EXCLUDED_EDGE_COLS:
            assert col not in rec


def test_meta_whitelist_enforced(empty_kg):
    # Insert a non-whitelisted meta row and assert it does NOT appear.
    empty_kg.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('ephemeral_build_id', 'XYZ')"
    )
    empty_kg.conn.commit()
    meta_bytes = canonicalize_meta(empty_kg.conn)
    text = meta_bytes.decode("utf-8")
    assert "ephemeral_build_id" not in text
    for key in META_WHITELIST:
        # schema_version should be present when inserted by init_db
        pass  # presence tested elsewhere via graph_hash stability


def test_node_order_by_id_lex(empty_kg):
    # Insert in reverse alpha order; canonical should emit sorted.
    for i in ("z_last", "a_first", "m_middle"):
        empty_kg.upsert_node(Node(
            id=f"axiom:{i}", node_type="Axiom",
            title=i, body=i,
            authority="primary", epistemic_status="axiom",
            method="axioms_block", source_locator=f"file:{i}",
        ))
    nodes_bytes = canonicalize_nodes(empty_kg.conn)
    ids = [json.loads(line)["id"] for line in nodes_bytes.decode("utf-8").splitlines()]
    assert ids == sorted(ids), f"node order not sorted: {ids}"


def test_edge_order_by_triple_lex(empty_kg):
    # Seed a few edges with varying (src, dst, type) and assert ordering.
    for name in ("z_node", "a_node"):
        empty_kg.upsert_node(Node(
            id=f"axiom:{name}", node_type="Axiom",
            title=name, body=name,
            authority="primary", epistemic_status="axiom",
            method="axioms_block", source_locator=f"file:{name}",
        ))
    empty_kg.upsert_node(Node(
        id="cert:x", node_type="Cert", title="x", body="x",
        authority="derived", epistemic_status="certified",
        method="cert_validator", source_locator="file:x",
    ))
    for t in ("validates", "extends"):
        empty_kg.upsert_edge(Edge(
            src_id="cert:x", dst_id="axiom:z_node",
            edge_type=t, method="structural",
        ))
        empty_kg.upsert_edge(Edge(
            src_id="cert:x", dst_id="axiom:a_node",
            edge_type=t, method="structural",
        ))
    edges_bytes = canonicalize_edges(empty_kg.conn)
    triples = [
        (json.loads(l)["src_id"], json.loads(l)["dst_id"], json.loads(l)["edge_type"])
        for l in edges_bytes.decode("utf-8").splitlines()
    ]
    assert triples == sorted(triples), f"edge order not sorted: {triples}"


def test_unicode_nfc_normalization(empty_kg):
    # "café" composed vs decomposed should produce identical bytes.
    composed = "caf\u00e9"  # café, NFC
    decomposed = "cafe\u0301"  # NFD
    assert composed != decomposed  # sanity: different code-point sequences
    # Both normalize to NFC.
    assert unicodedata.normalize("NFC", composed) == unicodedata.normalize("NFC", decomposed)
    # Insert a node with the decomposed form; canonical output should
    # contain the composed form.
    empty_kg.upsert_node(Node(
        id="axiom:cafe", node_type="Axiom",
        title=decomposed, body=decomposed,
        authority="primary", epistemic_status="axiom",
        method="axioms_block", source_locator="file:cafe",
    ))
    nodes_bytes = canonicalize_nodes(empty_kg.conn)
    assert composed.encode("utf-8") in nodes_bytes
    # Even if the DB stored the decomposed form, canonical output is NFC.


def test_empty_graph_deterministic(empty_kg):
    # init_db populates schema_version in meta, so meta isn't strictly empty.
    # But nodes + edges are empty.
    n, e, m = canonicalize_graph(empty_kg.conn)
    assert n == b""
    assert e == b""
    # meta has schema_version
    assert b"schema_version" in m


def test_graph_hash_stable_across_two_runs(empty_kg):
    _seed_graph(empty_kg)
    h1 = graph_hash(empty_kg.conn)
    # Touch updated_ts by re-upserting one node. Hash should be stable
    # because updated_ts is in EXCLUDED_NODE_COLS.
    empty_kg.upsert_node(Node(
        id="axiom:A1", node_type="Axiom",
        title="A1 — No-Zero", body="States in {1..N}.",
        authority="primary", epistemic_status="axiom",
        method="axioms_block", source_locator="file:QA_AXIOMS_BLOCK.md#A1",
    ))
    h2 = graph_hash(empty_kg.conn)
    assert h1 == h2, f"hash drifted after timestamp-only touch: {h1} vs {h2}"


def test_compute_fixture_hash_content_only():
    # Two manifests with identical files[] but different metadata must
    # produce the same fixture_hash. C#4 contract.
    files = [
        {"path": "a.md", "sha256": "deadbeef", "bytes": 100},
        {"path": "b.md", "sha256": "cafebabe", "bytes": 200},
    ]
    m1 = {"files": files, "metadata": {"captured_at_utc": "2026-04-16T10:00:00Z",
                                        "repo_head": "aaa"}}
    m2 = {"files": list(files), "metadata": {"captured_at_utc": "2026-05-01T23:59:59Z",
                                              "repo_head": "bbb"}}
    assert compute_fixture_hash(m1) == compute_fixture_hash(m2)
    # Changing a file content changes the hash.
    m3 = {"files": files + [{"path": "c.md", "sha256": "1234", "bytes": 50}],
          "metadata": m1["metadata"]}
    assert compute_fixture_hash(m3) != compute_fixture_hash(m1)


def test_compute_fixture_hash_order_independent():
    # Files may arrive unsorted in manifest input; compute_fixture_hash
    # sorts internally. Two unsorted manifests with same files[] → same hash.
    f1 = [{"path": "a.md", "sha256": "x", "bytes": 1},
          {"path": "z.md", "sha256": "y", "bytes": 2}]
    f2 = list(reversed(f1))
    m1 = {"files": f1, "metadata": {}}
    m2 = {"files": f2, "metadata": {}}
    assert compute_fixture_hash(m1) == compute_fixture_hash(m2)


def test_arag_extractor_respects_db_path_override(tmp_path):
    """C#1 call-site coverage — arag.search accepts db_path kwarg and uses it."""
    import sqlite3 as _sq
    from tools.qa_kg.extractors import arag as x_arag

    # Build a tiny A-RAG-shape DB at an alternate path.
    alt = tmp_path / "tiny_arag.sqlite"
    c = _sq.connect(str(alt))
    c.executescript("""
        CREATE TABLE messages (
            msg_id TEXT PRIMARY KEY, source TEXT, role TEXT, conv_title TEXT,
            create_time_utc TEXT, b INTEGER, e INTEGER, raw_text TEXT
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            msg_id UNINDEXED, raw_text,
            content='messages', content_rowid='rowid'
        );
        CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, msg_id, raw_text)
                VALUES (new.rowid, new.msg_id, new.raw_text);
        END;
    """)
    c.execute(
        "INSERT INTO messages VALUES ('m1','chatgpt','user','conv','2026-04-16T00:00:00Z',1,1,'unique_marker_token')"
    )
    c.commit()
    c.close()

    # search() with db_path override should find it.
    rows = x_arag.search("unique_marker_token", k=5, db_path=alt)
    assert len(rows) == 1
    assert rows[0]["msg_id"] == "m1"

    # search() without override hits the default path; if the default
    # doesn't exist (test host), FileNotFoundError is expected.
    # Skip assertion on default — we only care that the override worked.
