"""QA-KG canonical graph serialization — Phase 5 determinism.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Pure canonicalization module. No DB writes, no network, no wall-clock reads.
The only state this module touches is the read-only sqlite3.Connection
passed in by the caller.

Cert [228] QA_KG_DETERMINISM_CERT.v1 anchors:

- EXCLUDED columns (build-time metadata, not content identity):
    nodes.created_ts, nodes.updated_ts, nodes.vetted_ts
    nodes.last_check_ts, nodes.last_check_ok, nodes.last_check_msg
    edges.created_ts
- INCLUDED ordering:
    nodes  -> ORDER BY id ASC (string-lex)
    edges  -> ORDER BY src_id, dst_id, edge_type ASC (lex triple)
    meta   -> whitelist {'schema_version'}, sorted by key
- Per-record encoding:
    json.dumps(record, sort_keys=True, ensure_ascii=False,
               separators=(",", ":"))
    One record per line. Trailing newline after the last record.
- Three byte-streams concatenated with b"\\n" separators:
    graph_hash = sha256(nodes_jsonl + b"\\n" + edges_jsonl + b"\\n" + meta_jsonl)
- Unicode: NFC-normalize every string value at serialize time (closes the
  cross-platform HFS+/APFS NFD trap noted in Phase 5 plan C#3 edge case).

This module is NOT pinned to manifest.repo_head for D2/D3 reproducibility
(see build_context.materialize_pinned_repo docstring). It IS the test
harness that makes the determinism claim; it runs at working-tree HEAD.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import hashlib
import json
import sqlite3
import unicodedata


EXCLUDED_NODE_COLS: frozenset[str] = frozenset({
    "created_ts",
    "updated_ts",
    "vetted_ts",
    "last_check_ts",
    "last_check_ok",
    "last_check_msg",
})

EXCLUDED_EDGE_COLS: frozenset[str] = frozenset({
    "created_ts",
})

META_WHITELIST: frozenset[str] = frozenset({
    "schema_version",
})


def _nfc(value):
    """NFC-normalize strings; pass non-strings through unchanged.

    Closes the macOS HFS+ NFD edge case where a composed character on one
    host and a decomposed sequence on another produce byte-different JSON
    for identical semantic content.
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def _canonicalize_rows(
    rows: list[sqlite3.Row],
    excluded: frozenset[str],
) -> bytes:
    """Encode a list of already-ordered rows as JSONL bytes."""
    if not rows:
        return b""
    parts: list[str] = []
    for row in rows:
        rec = {k: _nfc(row[k]) for k in row.keys() if k not in excluded}
        parts.append(json.dumps(
            rec,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ))
    return ("\n".join(parts) + "\n").encode("utf-8")


def canonicalize_nodes(conn: sqlite3.Connection) -> bytes:
    rows = conn.execute("SELECT * FROM nodes ORDER BY id ASC").fetchall()
    return _canonicalize_rows(rows, EXCLUDED_NODE_COLS)


def canonicalize_edges(conn: sqlite3.Connection) -> bytes:
    rows = conn.execute(
        "SELECT * FROM edges ORDER BY src_id ASC, dst_id ASC, edge_type ASC"
    ).fetchall()
    return _canonicalize_rows(rows, EXCLUDED_EDGE_COLS)


def canonicalize_meta(conn: sqlite3.Connection) -> bytes:
    rows = conn.execute("SELECT key, value FROM meta ORDER BY key ASC").fetchall()
    filtered = [r for r in rows if r["key"] in META_WHITELIST]
    if not filtered:
        return b""
    parts = [
        json.dumps(
            {"key": _nfc(r["key"]), "value": _nfc(r["value"])},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for r in filtered
    ]
    return ("\n".join(parts) + "\n").encode("utf-8")


def canonicalize_graph(conn: sqlite3.Connection) -> tuple[bytes, bytes, bytes]:
    """Return (nodes_jsonl, edges_jsonl, meta_jsonl) byte-streams.

    Caller responsible for hashing the concatenation (see graph_hash).
    """
    return (
        canonicalize_nodes(conn),
        canonicalize_edges(conn),
        canonicalize_meta(conn),
    )


def graph_hash(conn: sqlite3.Connection) -> str:
    """SHA256 hex of the canonical graph.

    Formula:
        sha256(nodes_jsonl + b"\\n" + edges_jsonl + b"\\n" + meta_jsonl)

    The b"\\n" separators are fixed domain separators; they prevent a node
    JSONL ending mid-record from colliding with an edge JSONL starting
    mid-record (defense-in-depth for unparseable-but-same-bytes edge cases).
    """
    n, e, m = canonicalize_graph(conn)
    h = hashlib.sha256()
    h.update(n)
    h.update(b"\n")
    h.update(e)
    h.update(b"\n")
    h.update(m)
    return h.hexdigest()


def compute_fixture_hash(manifest: dict) -> str:
    """Content-hash of a CANONICAL_MANIFEST.json file list.

    Hash consumes ONLY the files[] portion, not metadata (captured_at_utc,
    repo_head, qa_compliance). Re-capturing byte-identical content at a
    different time or from a different HEAD produces identical hash. This
    is deliberate per Phase 5 plan C#4: fixture_hash attests to content
    identity, not capture-ceremony identity.
    """
    files = sorted(manifest["files"], key=lambda f: f["path"])
    payload = json.dumps(
        files,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
