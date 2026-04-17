# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 MCP audit module; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
"""QA-KG Phase 6 audit module — write-to-query_log helpers for the MCP surface.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Every MCP tool call must log to the existing `query_log` table before
returning to the caller. The MCP server constructor requires a non-None
audit_log instance, so the audit path cannot be bypassed at
construction. Individual tool bodies call log_read or log_write before
returning.

Schema v4 `query_log` is reused verbatim — no new table, no new columns.
Columns: ts, query, node_id, rank, session. Semantics:

  Read ops (qa_kg_search, qa_kg_get_node, qa_kg_neighbors):
    one row per returned node_id, rank = 0..k-1, query = canonical call
    string, session = MCP session id.

  Write op (qa_kg_promote_agent_note):
    one row, rank = -1 (sentinel for "write"), node_id = promote target,
    query = f"promote:{node_id}:{via_cert}", session = MCP session id.

Rank = -1 is the agreed sentinel; pre-Phase-6 consumers of query_log
(the digest hit counter) filter to rank >= 0, so they are unaffected.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sqlite3
import time
from dataclasses import dataclass


WRITE_RANK_SENTINEL = -1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class AuditLog:
    """Handle on the query_log write surface.

    Constructed once per MCP server instance, passed to every tool. The
    MCP server refuses to start if audit_log is None (construction-time
    guard), so tool bodies can assume the handle is live.
    """
    conn: sqlite3.Connection

    def log_read(self, session: str, query: str, node_ids: list[str]) -> None:
        """Log a read-tool call. One row per returned node_id."""
        if not session:
            raise ValueError("AuditLog.log_read requires non-empty session")
        if not isinstance(query, str):
            raise ValueError("AuditLog.log_read requires string query")
        now = _now()
        rows = [
            (now, query, node_id, rank, session)
            for rank, node_id in enumerate(node_ids)
        ]
        if rows:
            self.conn.executemany(
                "INSERT INTO query_log(ts, query, node_id, rank, session) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            self.conn.commit()

    def log_write(self, session: str, node_id: str, via_cert: str) -> None:
        """Log a write-tool call (promote). One row, rank=-1 sentinel."""
        if not session:
            raise ValueError("AuditLog.log_write requires non-empty session")
        if not node_id:
            raise ValueError("AuditLog.log_write requires non-empty node_id")
        if not via_cert:
            raise ValueError("AuditLog.log_write requires non-empty via_cert")
        query = f"promote:{node_id}:{via_cert}"
        self.conn.execute(
            "INSERT INTO query_log(ts, query, node_id, rank, session) "
            "VALUES (?,?,?,?,?)",
            (_now(), query, node_id, WRITE_RANK_SENTINEL, session),
        )
        self.conn.commit()
