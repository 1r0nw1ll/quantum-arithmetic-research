#!/usr/bin/env python3
# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 MCP stdio server; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
"""QA-KG Phase 6 MCP stdio server — agent-facing read-only / read-write surface.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Implements a minimal MCP-compatible JSON-RPC server over stdio with
exactly FOUR tools:

  READ_ONLY (always visible):
    qa_kg_search      — authority-tiered ranked retrieval
    qa_kg_get_node    — single-node lookup with epistemic fields
    qa_kg_neighbors   — incoming/outgoing edges for a node

  READ_WRITE (visible only under --cap read_write):
    qa_kg_promote_agent_note — escalate an AgentNote via kg.promote()

The server:

  * Declares its capability at spawn via `--cap {read_only,read_write}`.
    The flag is the only capability declaration path; no MCP message
    can mutate the mask after start-up. A READ_ONLY session's tools/list
    response omits the promote tool entirely (plan W6).
  * Requires an AuditLog at construction. Every tool call goes through
    audit.log_read / log_write before returning to the caller (plan W7).
  * Rate-limits qa_kg_promote_agent_note via
    tools.qa_kg_mcp.rate_limit.increment() under flock on
    _agent_writes.json. Over-cap raises RateLimitExceeded, surfaced to
    the agent as a JSON-RPC error (plan W4).
  * Deep-copies and stamps broadcast_payload["mcp_session"] AFTER the
    copy so an agent-provided spoofed key is overwritten, not preserved
    (plan v2 M3).

Session id is `mcp:<uuid4>` by default; overridable via the MCP
`initialize` params (`clientInfo.sessionId`) so a well-behaved client
can align its collab-bus session_done broadcasts with the MCP session.

Stdlib only — the `mcp` python SDK is not a project dependency.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import argparse
import copy
import datetime as _dt
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg import connect
from tools.qa_kg._audit import AuditLog
from tools.qa_kg.kg import FirewallViolation, KG
from tools.qa_kg_mcp.capabilities import (
    Capability, parse_capability, visible_tools,
)
from tools.qa_kg_mcp.rate_limit import RateLimitExceeded


_log = logging.getLogger("qa_kg_mcp.server")


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "qa-kg-mcp"
SERVER_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Tool schemas — JSON Schema draft-07 shape per MCP convention.
# ---------------------------------------------------------------------------


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "qa_kg_search": {
        "name": "qa_kg_search",
        "description": (
            "Authority-tiered ranked retrieval over the QA-KG. Returns "
            "ranked hits with score breakdown and contradiction labels. "
            "READ ONLY."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":         {"type": "string"},
                "min_authority": {"type": "string", "default": "internal",
                                  "enum": ["primary", "derived", "internal", "agent"]},
                "domain":        {"type": ["string", "null"], "default": None},
                "valid_at":      {"type": ["string", "null"], "default": None,
                                  "description": "ISO-8601 timestamp"},
                "k":             {"type": "integer", "default": 10, "minimum": 1,
                                  "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "qa_kg_get_node": {
        "name": "qa_kg_get_node",
        "description": (
            "Return a single node with epistemic fields + lifecycle_state. "
            "Returns {error: 'not_found'} on miss (never raises to agent). "
            "READ ONLY."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
    "qa_kg_neighbors": {
        "name": "qa_kg_neighbors",
        "description": (
            "Return neighboring edges for a node. READ ONLY."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id":    {"type": "string"},
                "edge_types": {"type": ["array", "null"],
                               "items": {"type": "string"}, "default": None},
                "direction":  {"type": "string", "default": "both",
                               "enum": ["in", "out", "both"]},
            },
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
    "qa_kg_promote_agent_note": {
        "name": "qa_kg_promote_agent_note",
        "description": (
            "Escalate an AgentNote so it can emit causal edges. Calls "
            "kg.promote() with all Phase 2 invariants + Phase 6 rate-limit. "
            "WRITE — exclusively for AgentNote promotion. Cannot create new "
            "nodes; promotion attaches a promoted-from edge to an existing "
            "agent-authored node."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id":           {"type": "string"},
                "via_cert":          {"type": "string"},
                "promoter_node_id":  {"type": "string"},
                "broadcast_payload": {"type": "object"},
            },
            "required": ["node_id", "via_cert", "promoter_node_id",
                         "broadcast_payload"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Server object.
# ---------------------------------------------------------------------------


class MCPServer:
    """Minimal stdio JSON-RPC server implementing the MCP initialize /
    tools/list / tools/call flow.

    Construction requires a non-None audit_log — the MCP write surface
    cannot be assembled without its audit handle. This is the "cannot be
    bypassed" part of plan §6 (audit_log=non-None).
    """

    def __init__(
        self,
        capability: Capability,
        kg: KG,
        audit_log: AuditLog,
        *,
        session_id: str | None = None,
        agent_writes_path: Path | None = None,
        ledger_path: Path | None = None,
    ):
        if audit_log is None:
            raise ValueError(
                "MCPServer requires non-None audit_log — MCP write surface "
                "cannot be assembled without its audit handle."
            )
        self._cap = capability
        self._kg = kg
        self._audit = audit_log
        self._session_id = session_id or f"mcp:{uuid.uuid4().hex[:12]}"
        self._agent_writes_path = agent_writes_path
        self._ledger_path = ledger_path
        self._registered_tools: dict[str, Callable] = self._build_registry()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def registered_tools(self) -> list[str]:
        return sorted(self._registered_tools.keys())

    # -- registry ---------------------------------------------------------

    def _build_registry(self) -> dict[str, Callable]:
        visible = visible_tools(self._cap)
        registry: dict[str, Callable] = {}
        if "qa_kg_search" in visible:
            registry["qa_kg_search"] = self._tool_search
        if "qa_kg_get_node" in visible:
            registry["qa_kg_get_node"] = self._tool_get_node
        if "qa_kg_neighbors" in visible:
            registry["qa_kg_neighbors"] = self._tool_neighbors
        if "qa_kg_promote_agent_note" in visible:
            registry["qa_kg_promote_agent_note"] = self._tool_promote
        return registry

    # -- tool bodies ------------------------------------------------------

    def _tool_search(self, args: dict) -> dict:
        query = args["query"]
        min_authority = args.get("min_authority", "internal")
        domain = args.get("domain")
        valid_at_raw = args.get("valid_at")
        k = int(args.get("k", 10))
        valid_at = None
        if valid_at_raw:
            valid_at = _dt.datetime.fromisoformat(
                valid_at_raw.replace("Z", "+00:00")
            )
        hits = self._kg.search_authority_ranked(
            query, min_authority=min_authority, domain=domain,
            valid_at=valid_at, k=k,
        )
        serialized = [
            {
                "node_id":              h.node["id"],
                "title":                h.node["title"],
                "node_type":            h.node["node_type"],
                "authority":            h.authority,
                "score":                h.score,
                "score_breakdown":      h.score_breakdown,
                "contradiction_state":  h.contradiction_state,
                "provenance_depth":     h.provenance_depth,
                "lifecycle_state":      h.node["lifecycle_state"],
            }
            for h in hits
        ]
        self._audit.log_read(
            self._session_id,
            f"search:{query}:ma={min_authority}:k={k}",
            [h["node_id"] for h in serialized],
        )
        return {"hits": serialized, "query": query, "count": len(serialized)}

    def _tool_get_node(self, args: dict) -> dict:
        node_id = args["node_id"]
        row = self._kg.get(node_id)
        if row is None:
            self._audit.log_read(
                self._session_id, f"get_node:{node_id}", [],
            )
            return {"error": "not_found", "node_id": node_id}
        self._audit.log_read(
            self._session_id, f"get_node:{node_id}", [node_id],
        )
        return {
            "node_id":          row["id"],
            "node_type":        row["node_type"],
            "title":            row["title"],
            "body":             row["body"],
            "tier":             row["tier"],
            "idx_b":            row["idx_b"],
            "idx_e":            row["idx_e"],
            "subject_b":        row["subject_b"],
            "subject_e":        row["subject_e"],
            "authority":        row["authority"],
            "epistemic_status": row["epistemic_status"],
            "method":           row["method"],
            "source_locator":   row["source_locator"],
            "lifecycle_state":  row["lifecycle_state"],
            "confidence":       row["confidence"],
            "valid_from":       row["valid_from"],
            "valid_until":      row["valid_until"],
            "domain":           row["domain"],
        }

    def _tool_neighbors(self, args: dict) -> dict:
        node_id = args["node_id"]
        edge_types = args.get("edge_types")
        direction = args.get("direction", "both")
        rows = self._kg.neighbors(
            node_id, edge_types=edge_types, direction=direction,
        )
        out = [
            {
                "src_id":    r["src_id"],
                "dst_id":    r["dst_id"],
                "edge_type": r["edge_type"],
                "via_cert":  r["via_cert"],
                "method":    r["method"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]
        # Log the "other end" of each edge as the touched node_id.
        touched = sorted({
            (r["dst_id"] if r["src_id"] == node_id else r["src_id"])
            for r in rows
        })
        self._audit.log_read(
            self._session_id,
            f"neighbors:{node_id}:dir={direction}",
            touched,
        )
        return {"node_id": node_id, "neighbors": out, "count": len(out)}

    def _tool_promote(self, args: dict) -> dict:
        node_id = args["node_id"]
        via_cert = args["via_cert"]
        promoter_node_id = args["promoter_node_id"]
        raw_payload = args["broadcast_payload"]
        if not isinstance(raw_payload, dict):
            return {"error": "broadcast_payload must be an object"}
        # Plan v2 M3: deep-copy FIRST so mutations can't leak back to caller,
        # then stamp mcp_session AFTER the copy so any spoofed mcp_session
        # key in the agent's payload is OVERWRITTEN, not preserved.
        payload = copy.deepcopy(raw_payload)
        payload["mcp_session"] = self._session_id
        payload["mcp_stamp_ts"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
        )
        # Audit the intent BEFORE the rate-limit call so a cap-triggered
        # rejection is still visible in the audit log.
        self._audit.log_write(self._session_id, node_id, via_cert)
        self._kg.promote(
            agent_note_id=node_id,
            via_cert=via_cert,
            promoter_node_id=promoter_node_id,
            broadcast_payload=payload,
            ledger_path=self._ledger_path,
            mcp_session=self._session_id,
            agent_writes_path=self._agent_writes_path,
        )
        return {
            "status": "promoted",
            "node_id": node_id,
            "via_cert": via_cert,
            "promoter_node_id": promoter_node_id,
            "mcp_session": self._session_id,
        }

    # -- JSON-RPC plumbing ------------------------------------------------

    def handle(self, request: dict) -> dict | None:
        """Handle one JSON-RPC request, return the response dict (or None
        for notifications).

        Errors use JSON-RPC 2.0 error codes:
          -32601 method not found
          -32602 invalid params
          -32603 internal error
          -32000 (reserved) capability / firewall rejection
        """
        if not isinstance(request, dict):
            return self._error(None, -32600, "request must be an object")
        rid = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if method == "initialize":
            client_session = (
                params.get("clientInfo", {}).get("sessionId")
                if isinstance(params.get("clientInfo"), dict) else None
            )
            if isinstance(client_session, str) and client_session:
                self._session_id = client_session
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                        "capability": self._cap.value,
                        "sessionId": self._session_id,
                    },
                },
            }

        if method == "tools/list":
            tools = [TOOL_SCHEMAS[name] for name in self.registered_tools]
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools}}

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if tool_name not in self._registered_tools:
                # Tool is either unknown OR hidden by capability mask.
                # Identical error in both cases — don't leak the masked
                # write surface to READ_ONLY sessions.
                return self._error(
                    rid, -32601,
                    f"tool {tool_name!r} not available under capability "
                    f"{self._cap.value}",
                )
            try:
                result = self._registered_tools[tool_name](tool_args)
            except RateLimitExceeded as exc:
                return self._error(rid, -32000, f"rate_limit: {exc}")
            except FirewallViolation as exc:
                return self._error(rid, -32000, f"firewall: {exc}")
            except (ValueError, KeyError) as exc:
                return self._error(rid, -32602, f"invalid params: {exc}")
            return {"jsonrpc": "2.0", "id": rid, "result": result}

        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": rid, "result": {}}

        if method is None:
            return self._error(rid, -32600, "missing method")

        return self._error(rid, -32601, f"method {method!r} not found")

    def _error(self, rid, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": code, "message": message},
        }

    def serve_stdio(self) -> None:
        """Read newline-delimited JSON-RPC requests from stdin and write
        responses to stdout. Exit on EOF.
        """
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.stdout.write(json.dumps(self._error(
                    None, -32700, f"parse error: {exc}",
                )) + "\n")
                sys.stdout.flush()
                continue
            response = self.handle(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


def _build_server_from_args(argv: list[str] | None = None) -> MCPServer:
    parser = argparse.ArgumentParser(
        description="QA-KG Phase 6 MCP stdio server.",
    )
    parser.add_argument(
        "--cap", required=True,
        help="Capability class: read_only or read_write.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to qa_kg.db (default: tools/qa_kg/qa_kg.db).",
    )
    parser.add_argument(
        "--agent-writes", default=None,
        help="Path to _agent_writes.json (default: "
             "qa_alphageometry_ptolemy/_agent_writes.json).",
    )
    parser.add_argument(
        "--ledger", default=None,
        help="Path to _meta_ledger.json (default: "
             "qa_alphageometry_ptolemy/_meta_ledger.json).",
    )
    parser.add_argument(
        "--session", default=None,
        help="Explicit MCP session id (default: mcp:<uuid4>).",
    )
    args = parser.parse_args(argv)

    capability = parse_capability(args.cap)
    kg = connect(Path(args.db) if args.db else None)
    audit = AuditLog(conn=kg.conn)
    return MCPServer(
        capability=capability,
        kg=kg,
        audit_log=audit,
        session_id=args.session,
        agent_writes_path=(Path(args.agent_writes) if args.agent_writes
                           else None),
        ledger_path=(Path(args.ledger) if args.ledger else None),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[qa_kg_mcp] %(message)s",
        stream=sys.stderr,
    )
    server = _build_server_from_args(argv)
    _log.info("qa-kg-mcp started cap=%s session=%s",
              server._cap.value, server.session_id)
    try:
        server.serve_stdio()
    except KeyboardInterrupt:
        _log.info("interrupted — exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
