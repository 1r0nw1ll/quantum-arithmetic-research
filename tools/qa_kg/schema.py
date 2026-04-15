"""SQLite schema for QA-KG.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Single-file DB. Source of truth stays the files; graph is a derived, rebuildable index.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import os
import sqlite3
from pathlib import Path


DEFAULT_DB = Path(os.environ.get(
    "QA_KG_DB",
    str(Path(__file__).resolve().parent / "qa_kg.db"),
))


DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT PRIMARY KEY,        -- stable key, e.g. "cert:225" or "axiom:A1"
    node_type      TEXT NOT NULL,           -- Axiom|Cert|Concept|Person|Work|Thought|Rule|Claim
    title          TEXT NOT NULL,
    body           TEXT NOT NULL DEFAULT '',
    tier           TEXT NOT NULL,           -- singularity|cosmos|satellite|unassigned
    coord_b        INTEGER NOT NULL,
    coord_e        INTEGER NOT NULL,
    source         TEXT NOT NULL DEFAULT '', -- file path or OB thought id
    vetted_by      TEXT NOT NULL DEFAULT '', -- cert id that vets this node (if any)
    predicate_ref  TEXT NOT NULL DEFAULT '', -- dotted.path:to.callable for claim nodes
    last_check_ts  TEXT NOT NULL DEFAULT '',
    last_check_ok  INTEGER,                  -- NULL=never run, 0=fail, 1=pass
    last_check_msg TEXT NOT NULL DEFAULT '',
    created_ts     TEXT NOT NULL,
    updated_ts     TEXT NOT NULL,
    CHECK (coord_b BETWEEN 1 AND 9),
    CHECK (coord_e BETWEEN 1 AND 9),
    CHECK (tier IN ('singularity','cosmos','satellite','unassigned'))
);

CREATE INDEX IF NOT EXISTS idx_nodes_tier ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_coord ON nodes(coord_b, coord_e);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);

CREATE TABLE IF NOT EXISTS edges (
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,  -- cites|validates|extends|contradicts|maps-to|
                                -- depends-on|observer-projection-of|replaces|
                                -- derived-from|co-author|instantiates|falsifies
    confidence  REAL NOT NULL DEFAULT 1.0,
    provenance  TEXT NOT NULL DEFAULT '',
    via_cert    TEXT NOT NULL DEFAULT '',   -- cert id if firewall-mediated
    created_ts  TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_id, edge_type),
    FOREIGN KEY (src_id) REFERENCES nodes(id),
    FOREIGN KEY (dst_id) REFERENCES nodes(id),
    CHECK (confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst_id);
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

-- Query telemetry — used to weight hot Cosmos nodes in digests.
CREATE TABLE IF NOT EXISTS query_log (
    ts         TEXT NOT NULL,
    query      TEXT NOT NULL,
    node_id    TEXT NOT NULL,
    rank       INTEGER NOT NULL,
    session    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_query_log_node ON query_log(node_id);
"""


def init_db(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open (and create if needed) the QA-KG SQLite DB."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
    conn.executescript(DDL)
    conn.commit()
    return conn
