"""SQLite schema for QA-KG.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Candidate F retrieval index. Column names `idx_b/idx_e` intentionally differ
from QA state names `(b,e)`: these are a content-hash × node-type-rank
retrieval partition, NOT a QA state. A QA state pair is derived via d=b+e,
a=b+2e; attempting that on idx_b/idx_e produces nonsense. The reserved
`subject_b/subject_e` columns are the only ones that encode declared QA
state subjects, populated exclusively from cert metadata.

Edges carry a `method` field to record HOW the edge was established:
  - "keyword": regex / body-token match (confidence ≤ 0.3)
  - "cert_registry": from FAMILY_SWEEPS membership
  - "structural": derived by a validator-emitted proof link
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import os
import sqlite3
from pathlib import Path


SCHEMA_VERSION = 2  # Phase 0 rename: coord_b/coord_e → idx_b/idx_e; method on edges

DEFAULT_DB = Path(os.environ.get(
    "QA_KG_DB",
    str(Path(__file__).resolve().parent / "qa_kg.db"),
))


DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT PRIMARY KEY,
    node_type      TEXT NOT NULL,
    title          TEXT NOT NULL,
    body           TEXT NOT NULL DEFAULT '',
    tier           TEXT NOT NULL,
    idx_b          INTEGER,                     -- Candidate F retrieval-index b (hash partition)
    idx_e          INTEGER,                     -- Candidate F retrieval-index e (node_type rank)
    char_ord_sum   INTEGER,                     -- [202] observable that drives idx_b
    subject_b      INTEGER,                     -- declared SUBJECT state (nullable); only populated
    subject_e      INTEGER,                     -- from cert metadata — this is the QA state the
                                                -- node is ABOUT, not the node's own position
    source         TEXT NOT NULL DEFAULT '',
    vetted_by      TEXT NOT NULL DEFAULT '',    -- must be a DIFFERENT node id (not self)
    vetted_ts      TEXT NOT NULL DEFAULT '',
    predicate_ref  TEXT NOT NULL DEFAULT '',
    last_check_ts  TEXT NOT NULL DEFAULT '',
    last_check_ok  INTEGER,
    last_check_msg TEXT NOT NULL DEFAULT '',
    created_ts     TEXT NOT NULL,
    updated_ts     TEXT NOT NULL,
    CHECK (idx_b IS NULL OR idx_b BETWEEN 1 AND 9),
    CHECK (idx_e IS NULL OR idx_e BETWEEN 1 AND 9),
    CHECK (subject_b IS NULL OR subject_b BETWEEN 1 AND 9),
    CHECK (subject_e IS NULL OR subject_e BETWEEN 1 AND 9),
    CHECK (tier IN ('singularity','cosmos','satellite','unassigned'))
);

CREATE INDEX IF NOT EXISTS idx_nodes_tier   ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_idx_be ON nodes(idx_b, idx_e);
CREATE INDEX IF NOT EXISTS idx_nodes_type   ON nodes(node_type);

CREATE TABLE IF NOT EXISTS edges (
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    method      TEXT NOT NULL DEFAULT '',      -- 'keyword' | 'cert_registry' | 'structural' | ''
    provenance  TEXT NOT NULL DEFAULT '',
    via_cert    TEXT NOT NULL DEFAULT '',
    created_ts  TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, edge_type),
    FOREIGN KEY (src_id) REFERENCES nodes(id),
    FOREIGN KEY (dst_id) REFERENCES nodes(id),
    CHECK (confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst  ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
    id UNINDEXED, title, body,
    content='nodes', content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS nodes_fts_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, title, body) VALUES (new.rowid, new.id, new.title, new.body);
END;
CREATE TRIGGER IF NOT EXISTS nodes_fts_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title, body)
        VALUES ('delete', old.rowid, old.id, old.title, old.body);
END;
CREATE TRIGGER IF NOT EXISTS nodes_fts_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title, body)
        VALUES ('delete', old.rowid, old.id, old.title, old.body);
    INSERT INTO nodes_fts(rowid, id, title, body) VALUES (new.rowid, new.id, new.title, new.body);
END;

CREATE TABLE IF NOT EXISTS query_log (
    ts         TEXT NOT NULL,
    query      TEXT NOT NULL,
    node_id    TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    session    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_query_log_node ON query_log(node_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def init_db(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
    conn.executescript(DDL)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn
