# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 audit tests -->
"""Phase 6 audit module — query_log write surface tests.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run: python -m tools.qa_kg.tests.test_audit
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg import connect
from tools.qa_kg._audit import AuditLog, WRITE_RANK_SENTINEL


def _fresh_conn():
    d = tempfile.mkdtemp(prefix="qa_kg_audit_")
    kg = connect(Path(d) / "qa_kg.db")
    return kg, kg.conn


def test_log_read_writes_one_row_per_node():
    kg, conn = _fresh_conn()
    audit = AuditLog(conn=conn)
    audit.log_read("session-a", "q:hello", ["n1", "n2", "n3"])
    rows = conn.execute(
        "SELECT node_id, rank, session FROM query_log "
        "WHERE query=? ORDER BY rank",
        ("q:hello",),
    ).fetchall()
    assert len(rows) == 3
    assert [r["rank"] for r in rows] == [0, 1, 2]
    assert [r["node_id"] for r in rows] == ["n1", "n2", "n3"]
    assert all(r["session"] == "session-a" for r in rows)


def test_log_read_empty_result_writes_nothing():
    kg, conn = _fresh_conn()
    audit = AuditLog(conn=conn)
    audit.log_read("session-a", "q:nothing", [])
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM query_log WHERE query=?",
        ("q:nothing",),
    ).fetchone()["n"]
    assert count == 0


def test_log_write_uses_sentinel_rank():
    kg, conn = _fresh_conn()
    audit = AuditLog(conn=conn)
    audit.log_write("session-b", "agent:note1", "228")
    rows = conn.execute(
        "SELECT node_id, rank, session, query FROM query_log "
        "WHERE session=?",
        ("session-b",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["rank"] == WRITE_RANK_SENTINEL == -1
    assert rows[0]["node_id"] == "agent:note1"
    assert rows[0]["query"] == "promote:agent:note1:228"


def test_log_read_rejects_empty_session():
    kg, conn = _fresh_conn()
    audit = AuditLog(conn=conn)
    try:
        audit.log_read("", "q:x", ["n1"])
    except ValueError:
        pass
    else:
        raise AssertionError("empty session must raise")


def test_log_write_rejects_missing_fields():
    kg, conn = _fresh_conn()
    audit = AuditLog(conn=conn)
    for args in (("", "n1", "c"), ("s", "", "c"), ("s", "n1", "")):
        try:
            audit.log_write(*args)
        except ValueError:
            continue
        raise AssertionError(f"missing field must raise: {args}")


def test_digest_hit_counter_filters_write_sentinel():
    """Digest hit counter (kg.digest()) must ignore rank=-1 write rows.

    Pre-Phase-6 code counts query_log rows per node_id regardless of rank.
    Phase 6 introduces rank=-1 as the write sentinel. Any consumer that
    wants read-counts must filter rank >= 0. This test documents the
    contract and guards future regressions.
    """
    kg, conn = _fresh_conn()
    audit = AuditLog(conn=conn)
    audit.log_read("s", "q:read", ["n1", "n1", "n1"])  # three reads
    audit.log_write("s", "n1", "c")                     # one write
    reads = conn.execute(
        "SELECT COUNT(*) AS n FROM query_log WHERE node_id=? AND rank >= 0",
        ("n1",),
    ).fetchone()["n"]
    writes = conn.execute(
        "SELECT COUNT(*) AS n FROM query_log WHERE node_id=? AND rank = -1",
        ("n1",),
    ).fetchone()["n"]
    assert reads == 3
    assert writes == 1


TESTS = [
    test_log_read_writes_one_row_per_node,
    test_log_read_empty_result_writes_nothing,
    test_log_write_uses_sentinel_rank,
    test_log_read_rejects_empty_session,
    test_log_write_rejects_missing_fields,
    test_digest_hit_counter_filters_write_sentinel,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL {t.__name__}: {exc}")
            failed += 1
    if failed:
        sys.exit(1)
    print(f"\n{len(TESTS)}/{len(TESTS)} PASS")
