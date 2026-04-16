"""SQLite schema for QA-KG.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Schema v4 adds the four ranker-input columns (confidence, valid_from,
valid_until, domain) per Phase 4 of the QA-MEM scope. These feed the
authority-tiered ranker formula (see tools/qa_kg/ranker.py and cert
[254] qa_kg_authority_ranker_cert_v1). Defaults: confidence=1.0,
valid_from/valid_until/domain=''. Extractors that have measured signal
(e.g., source_claims with extraction_method='ocr') set confidence
explicitly; legacy nodes absorb the defaults on rebuild without
behavioural change.

Schema v3 adds epistemic fields (authority, epistemic_status, method,
source_locator, lifecycle_state) per Phase 1 of the QA-MEM scope. These are
ORTHOGONAL to the Candidate F retrieval-index columns (idx_b/idx_e): they
record HOW we know a node, not WHERE it sits on the lattice.

Authority values (locked — 4 crisp):
  - primary   = external canon predating QA project (Dale/Ben/Keely/Pond/
                Wildberger/Levin/Briddell, plus QA_AXIOMS_BLOCK.md source).
  - derived   = outputs of the cert machinery (validator results, structural
                provenance).
  - internal  = Will-authored project material (MEMORY.md, CLAUDE.md Hard
                Rules, repo docs, Will-captured OB without originSessionId).
  - agent     = Claude/Codex/OpenCode outputs (OB with originSessionId,
                collab-bus events with session:* identity).

Epistemic_status values: axiom, source_claim, source_work, certified,
observation, interpretation, conjecture. The allowed authority ×
epistemic_status matrix is enforced by cert [252] EF3; see
qa_alphageometry_ptolemy/qa_kg_epistemic_fields_cert_v1/allowed_matrix.json.

Phase 3: `source_work` added as the epistemic_status for primary-source
containers (a book, paper, wiki page) to keep them structurally distinct
from `source_claim` (a quoted fragment within a work). SCHEMA_VERSION
stays at 3 — this is an additive enum extension, not a structural change.
Old DBs rely on application-level `_validate_node_fields` in kg.py
plus the [252] EF3 allowed-matrix gate for runtime enforcement. Fresh
DBs get the updated CHECK constraint at DDL time. The canonical
migration path for a stale DB is:
    rm tools/qa_kg/qa_kg.db && python -m tools.qa_kg.cli populate
init_db emits a logged warning if an existing DB's CHECK constraint
does not contain 'source_work' — pointing at this rebuild path.

Lifecycle_state ∈ {current, deprecated, superseded, withdrawn}. Phase 3
activates `supersedes` edges alongside `lifecycle_state` and introduces
the extractor bridge translating `_status: frozen` file markers to
`lifecycle_state` on cert nodes (see tools/qa_kg/extractors/certs.py).

Back-compat aliases (Coord/compute_be/tier_for_coord) removed in v3 per the
Phase-0 pin; forward-failing test
tools/qa_kg/tests/test_kg_basic.py::test_back_compat_aliases_scheduled_for_removal_in_v3
flips from expect-present to expect-absent at SCHEMA_VERSION >= 3.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import logging
import os
import sqlite3
from pathlib import Path

_log = logging.getLogger("qa_kg.schema")


SCHEMA_VERSION = 4  # Phase 4: ranker-input columns (confidence, valid_from/until, domain)
# Phase 3 (2026-04-16): added "source_work" to EPISTEMIC_STATUSES as an
# additive enum extension. SCHEMA_VERSION stayed 3 for that change.
# Phase 4 (2026-04-16): bumped to 4 for the four new columns. CHECK
# constraints on ALTER TABLE ADD COLUMN are not enforced by SQLite, so
# old DBs rely on application-level _validate_node_fields in kg.py +
# cert [254] R6/R9 for runtime safety. _check_v4_columns_drift logs at
# init_db. Canonical migration: rm tools/qa_kg/qa_kg.db && python -m
# tools.qa_kg.cli build.

AUTHORITIES = ("primary", "derived", "internal", "agent")
EPISTEMIC_STATUSES = (
    "axiom", "source_claim", "source_work", "certified",
    "observation", "interpretation", "conjecture",
)
LIFECYCLE_STATES = ("current", "deprecated", "superseded", "withdrawn")

DEFAULT_DB = Path(os.environ.get(
    "QA_KG_DB",
    str(Path(__file__).resolve().parent / "qa_kg.db"),
))


_AUTH_LIST = ",".join(f"'{a}'" for a in AUTHORITIES)
_EPI_LIST = ",".join(f"'{s}'" for s in EPISTEMIC_STATUSES)
_LC_LIST = ",".join(f"'{s}'" for s in LIFECYCLE_STATES)

DDL = f"""
CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT PRIMARY KEY,
    node_type      TEXT NOT NULL,
    title          TEXT NOT NULL,
    body           TEXT NOT NULL DEFAULT '',
    tier           TEXT NOT NULL,
    idx_b          INTEGER,
    idx_e          INTEGER,
    char_ord_sum   INTEGER,
    subject_b      INTEGER,
    subject_e      INTEGER,
    source         TEXT NOT NULL DEFAULT '',
    vetted_by      TEXT NOT NULL DEFAULT '',
    vetted_ts      TEXT NOT NULL DEFAULT '',
    predicate_ref  TEXT NOT NULL DEFAULT '',
    last_check_ts  TEXT NOT NULL DEFAULT '',
    last_check_ok  INTEGER,
    last_check_msg TEXT NOT NULL DEFAULT '',
    -- Phase 1 (schema v3) epistemic fields
    authority        TEXT,
    epistemic_status TEXT,
    method           TEXT,
    source_locator   TEXT,
    lifecycle_state  TEXT NOT NULL DEFAULT 'current',
    -- Phase 4 (schema v4) ranker-input fields
    confidence       REAL NOT NULL DEFAULT 1.0,
    valid_from       TEXT NOT NULL DEFAULT '',
    valid_until      TEXT NOT NULL DEFAULT '',
    domain           TEXT NOT NULL DEFAULT '',
    created_ts     TEXT NOT NULL,
    updated_ts     TEXT NOT NULL,
    CHECK (idx_b IS NULL OR idx_b BETWEEN 1 AND 9),
    CHECK (idx_e IS NULL OR idx_e BETWEEN 1 AND 9),
    CHECK (subject_b IS NULL OR subject_b BETWEEN 1 AND 9),
    CHECK (subject_e IS NULL OR subject_e BETWEEN 1 AND 9),
    CHECK (tier IN ('singularity','cosmos','satellite','unassigned')),
    CHECK (authority IS NULL OR authority IN ({_AUTH_LIST})),
    CHECK (epistemic_status IS NULL OR epistemic_status IN ({_EPI_LIST})),
    CHECK (lifecycle_state IN ({_LC_LIST})),
    CHECK (confidence BETWEEN 0.0 AND 1.0)
);

CREATE INDEX IF NOT EXISTS idx_nodes_tier        ON nodes(tier);
CREATE INDEX IF NOT EXISTS idx_nodes_idx_be      ON nodes(idx_b, idx_e);
CREATE INDEX IF NOT EXISTS idx_nodes_type        ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_nodes_authority   ON nodes(authority);
CREATE INDEX IF NOT EXISTS idx_nodes_epistemic   ON nodes(epistemic_status);
CREATE INDEX IF NOT EXISTS idx_nodes_domain      ON nodes(domain);
CREATE INDEX IF NOT EXISTS idx_nodes_valid_until ON nodes(valid_until);

CREATE TABLE IF NOT EXISTS edges (
    src_id      TEXT NOT NULL,
    dst_id      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    method      TEXT NOT NULL DEFAULT '',
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


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _check_epistemic_enum_drift(conn: sqlite3.Connection) -> None:
    """Detect old-DB CHECK-constraint drift against current EPISTEMIC_STATUSES.

    Phase 3 adds 'source_work' as an additive enum value. SQLite cannot
    ALTER an existing CHECK constraint in place, so a DB created before
    this phase will still carry the old check string. Application-level
    validation in kg._validate_node_fields and cert [252] EF3 cover
    runtime safety, but we log a loud warning so the user has a clean
    pointer to the rebuild path if they hit enum rejections at insert
    time.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes'"
    ).fetchone()
    if row is None or row[0] is None:
        return
    sql = row[0]
    missing = [s for s in EPISTEMIC_STATUSES if f"'{s}'" not in sql]
    if missing:
        _log.warning(
            "qa_kg.db CHECK constraint is missing epistemic_status values %s. "
            "Application-level validation still enforces the current enum, but "
            "DB-level CHECK will not. To align the DB CHECK with the current "
            "schema, rebuild: rm tools/qa_kg/qa_kg.db && "
            "python -m tools.qa_kg.cli populate",
            missing,
        )


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """Idempotent ALTER TABLE migration for DBs created under earlier versions.

    SQLite CHECK constraints on ALTER TABLE ADD COLUMN are not enforced
    retroactively, so kg.py also validates authority/epistemic_status/
    lifecycle_state at insert time.
    """
    cols = _column_names(conn, "nodes")
    if "authority" not in cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN authority TEXT")
    if "epistemic_status" not in cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN epistemic_status TEXT")
    if "method" not in cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN method TEXT")
    if "source_locator" not in cols:
        conn.execute("ALTER TABLE nodes ADD COLUMN source_locator TEXT")
    if "lifecycle_state" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'current'"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_authority ON nodes(authority)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_epistemic ON nodes(epistemic_status)"
    )


def _migrate_to_v4(conn: sqlite3.Connection) -> None:
    """Idempotent ALTER TABLE migration for the four Phase 4 ranker columns.

    SQLite cannot retroactively apply CHECK on ALTER TABLE ADD COLUMN —
    application-level _validate_node_fields in kg.py catches confidence
    out-of-range. _check_v4_columns_drift logs at init_db time pointing at
    the rebuild path when the live nodes table is missing v4 columns.
    """
    cols = _column_names(conn, "nodes")
    if "confidence" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0"
        )
    if "valid_from" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN valid_from TEXT NOT NULL DEFAULT ''"
        )
    if "valid_until" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN valid_until TEXT NOT NULL DEFAULT ''"
        )
    if "domain" not in cols:
        conn.execute(
            "ALTER TABLE nodes ADD COLUMN domain TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_domain ON nodes(domain)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_nodes_valid_until ON nodes(valid_until)"
    )


def _check_v4_columns_drift(conn: sqlite3.Connection) -> None:
    """Detect old-DB lacking the Phase 4 ranker columns / CHECK constraint.

    Mirrors _check_epistemic_enum_drift: logs a loud warning pointing to
    the rebuild path. Application-level validation in kg.py covers the
    runtime; this is operator-facing diagnostics.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes'"
    ).fetchone()
    if row is None or row[0] is None:
        return
    sql = row[0]
    if "confidence" not in sql or "BETWEEN 0.0 AND 1.0" not in sql:
        _log.warning(
            "qa_kg.db nodes table is missing Phase 4 columns or CHECK on "
            "confidence. Application-level validation still enforces ranges, "
            "but DB-level CHECK will not. To align: "
            "rm tools/qa_kg/qa_kg.db && python -m tools.qa_kg.cli build"
        )


def init_db(path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
    conn.executescript(DDL)
    _migrate_to_v3(conn)
    _migrate_to_v4(conn)
    _check_epistemic_enum_drift(conn)
    _check_v4_columns_drift(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn
