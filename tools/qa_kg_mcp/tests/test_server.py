# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 MCP server tests -->
"""Phase 6 MCP stdio server + capability mask tests.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run: python -m tools.qa_kg_mcp.tests.test_server
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg import connect
from tools.qa_kg._audit import AuditLog
from tools.qa_kg.kg import Node
from tools.qa_kg_mcp.capabilities import (
    Capability, parse_capability, visible_tools,
)
from tools.qa_kg_mcp.server import MCPServer, PROTOCOL_VERSION


def _make_server(cap: Capability, tmp: Path):
    db = tmp / "t.db"
    kg = connect(db)
    audit = AuditLog(conn=kg.conn)
    return MCPServer(
        capability=cap, kg=kg, audit_log=audit,
        session_id="test-session",
        agent_writes_path=tmp / "_agent_writes.json",
        ledger_path=tmp / "_meta_ledger.json",
    ), kg


def test_parse_capability_rejects_garbage():
    for bad in ("", "READ_ONLY", "admin", "read-only", None):
        try:
            parse_capability(bad)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"parse_capability must reject {bad!r}")


def test_visible_tools_masks_write_for_read_only():
    assert "qa_kg_promote_agent_note" not in visible_tools(Capability.READ_ONLY)
    assert "qa_kg_search" in visible_tools(Capability.READ_ONLY)
    assert "qa_kg_promote_agent_note" in visible_tools(Capability.READ_WRITE)


def test_mcp_server_refuses_none_audit_log():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        kg = connect(tmp / "t.db")
        try:
            MCPServer(
                capability=Capability.READ_ONLY, kg=kg, audit_log=None,  # type: ignore[arg-type]
            )
        except ValueError as e:
            assert "audit_log" in str(e)
        else:
            raise AssertionError("MCPServer must require non-None audit_log")


def test_initialize_returns_capability_and_session():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, _ = _make_server(Capability.READ_ONLY, tmp)
        resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {}})
        assert resp["id"] == 1
        info = resp["result"]["serverInfo"]
        assert info["capability"] == "read_only"
        assert info["sessionId"] == "test-session"
        assert resp["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_tools_list_hides_promote_under_read_only():
    """Plan W6: READ_ONLY session's tools/list must omit promote."""
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, _ = _make_server(Capability.READ_ONLY, tmp)
        resp = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        assert names == {"qa_kg_search", "qa_kg_get_node", "qa_kg_neighbors"}


def test_tools_list_shows_promote_under_read_write():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, _ = _make_server(Capability.READ_WRITE, tmp)
        resp = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        assert names == {"qa_kg_search", "qa_kg_get_node",
                         "qa_kg_neighbors", "qa_kg_promote_agent_note"}


def test_promote_call_rejected_under_read_only():
    """READ_ONLY session's tools/call for promote returns method-not-found."""
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, _ = _make_server(Capability.READ_ONLY, tmp)
        resp = srv.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "qa_kg_promote_agent_note",
                       "arguments": {}},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32601


def test_get_node_roundtrip_logs_to_query_log():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, kg = _make_server(Capability.READ_ONLY, tmp)
        kg.upsert_node(Node(
            id="axiom:demo", node_type="Axiom", title="Demo",
            body="D",
            authority="primary", epistemic_status="axiom",
        ))
        resp = srv.handle({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "qa_kg_get_node",
                       "arguments": {"node_id": "axiom:demo"}},
        })
        assert resp["result"]["node_id"] == "axiom:demo"
        assert resp["result"]["authority"] == "primary"
        # Audit row
        count = kg.conn.execute(
            "SELECT COUNT(*) AS n FROM query_log WHERE session=?",
            ("test-session",),
        ).fetchone()["n"]
        assert count == 1


def test_get_node_miss_logs_empty_read():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, kg = _make_server(Capability.READ_ONLY, tmp)
        resp = srv.handle({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "qa_kg_get_node",
                       "arguments": {"node_id": "does:not:exist"}},
        })
        assert resp["result"]["error"] == "not_found"


def _make_promote_setup(tmp: Path):
    """Build a minimal KG + ledger for promote tests, reusing the pattern
    from tools/qa_kg/tests/test_kg_basic.py::_make_promote_fixtures."""
    srv, kg = _make_server(Capability.READ_WRITE, tmp)
    kg.upsert_node(Node(
        id="agent:np", node_type="Thought", title="agent",
        body="agent body", authority="agent",
        epistemic_status="observation",
    ))
    kg.upsert_node(Node(
        id="rule:prom", node_type="Rule", title="rule",
        body="internal rule", authority="internal",
        epistemic_status="interpretation",
    ))
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    from tools.qa_kg.kg import _current_git_head
    ledger = {
        "225": {"status": "PASS", "ts": now_iso,
                "git_head": _current_git_head()},
    }
    (tmp / "_meta_ledger.json").write_text(
        json.dumps(ledger), encoding="utf-8",
    )
    return srv, kg


def test_promote_tool_success_writes_edge_with_mcp_session():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, kg = _make_promote_setup(tmp)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        resp = srv.handle({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {
                "name": "qa_kg_promote_agent_note",
                "arguments": {
                    "node_id": "agent:np",
                    "via_cert": "225",
                    "promoter_node_id": "rule:prom",
                    "broadcast_payload": {"ts": now_iso,
                                          "session": "test-session"},
                },
            },
        })
        assert resp["result"]["status"] == "promoted"
        edge = kg.conn.execute(
            "SELECT provenance FROM edges WHERE edge_type='promoted-from'"
        ).fetchone()
        prov = json.loads(edge["provenance"])
        snap = prov["broadcast_payload_snapshot"]
        assert snap["mcp_session"] == "test-session"
        assert "mcp_stamp_ts" in snap


def test_promote_tool_overwrites_spoofed_mcp_session():
    """Plan v2 M3 / W3b: agent-provided mcp_session in broadcast_payload is
    OVERWRITTEN by the server's deep-copy + stamp, not preserved."""
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, kg = _make_promote_setup(tmp)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        spoofed = {"ts": now_iso, "session": "test-session",
                   "mcp_session": "SPOOFED-attacker-controlled"}
        resp = srv.handle({
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": {
                "name": "qa_kg_promote_agent_note",
                "arguments": {
                    "node_id": "agent:np",
                    "via_cert": "225",
                    "promoter_node_id": "rule:prom",
                    "broadcast_payload": spoofed,
                },
            },
        })
        assert resp["result"]["status"] == "promoted"
        edge = kg.conn.execute(
            "SELECT provenance FROM edges WHERE edge_type='promoted-from'"
        ).fetchone()
        snap = json.loads(edge["provenance"])["broadcast_payload_snapshot"]
        assert snap["mcp_session"] == "test-session"
        assert snap["mcp_session"] != "SPOOFED-attacker-controlled"
        # Caller's dict must be unmodified (deep-copy guard).
        assert spoofed["mcp_session"] == "SPOOFED-attacker-controlled"


def test_promote_rate_limit_surfaces_as_jsonrpc_error():
    import tools.qa_kg_mcp.rate_limit as rl
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        srv, kg = _make_promote_setup(tmp)
        old_cap = rl.MAX_WRITES_PER_SESSION
        rl.MAX_WRITES_PER_SESSION = 1
        try:
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            payload = {"ts": now_iso, "session": "test-session"}
            r1 = srv.handle({
                "jsonrpc": "2.0", "id": 12, "method": "tools/call",
                "params": {
                    "name": "qa_kg_promote_agent_note",
                    "arguments": {"node_id": "agent:np", "via_cert": "225",
                                  "promoter_node_id": "rule:prom",
                                  "broadcast_payload": payload},
                },
            })
            assert r1["result"]["status"] == "promoted"
            r2 = srv.handle({
                "jsonrpc": "2.0", "id": 13, "method": "tools/call",
                "params": {
                    "name": "qa_kg_promote_agent_note",
                    "arguments": {"node_id": "agent:np", "via_cert": "225",
                                  "promoter_node_id": "rule:prom",
                                  "broadcast_payload": payload},
                },
            })
            assert "error" in r2 and r2["error"]["code"] == -32000
            assert "rate_limit" in r2["error"]["message"]
        finally:
            rl.MAX_WRITES_PER_SESSION = old_cap


def test_subprocess_read_only_hides_promote():
    """Plan W6 (integration) — spawn the server as a subprocess, list tools
    via stdio, assert promote is omitted. try/finally with terminate→wait
    →kill escalation per plan v2 N1."""
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        db = tmp / "t.db"
        connect(db).conn.close()  # seed the DB on-disk
        input_payload = (
            json.dumps({"jsonrpc": "2.0", "id": 1,
                        "method": "initialize", "params": {}}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2,
                          "method": "tools/list"}) + "\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "tools.qa_kg_mcp.server",
             "--cap", "read_only", "--db", str(db)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        try:
            out, err = proc.communicate(input=input_payload, timeout=15)
            lines = [ln for ln in out.splitlines() if ln.strip()]
            resp_by_id = {}
            for ln in lines:
                obj = json.loads(ln)
                if "id" in obj:
                    resp_by_id[obj["id"]] = obj
            tools = resp_by_id[2]["result"]["tools"]
            names = {t["name"] for t in tools}
            assert names == {"qa_kg_search", "qa_kg_get_node",
                             "qa_kg_neighbors"}, (
                f"READ_ONLY tools/list must omit promote; got {names!r}. "
                f"stderr: {err}"
            )
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()


TESTS = [
    test_parse_capability_rejects_garbage,
    test_visible_tools_masks_write_for_read_only,
    test_mcp_server_refuses_none_audit_log,
    test_initialize_returns_capability_and_session,
    test_tools_list_hides_promote_under_read_only,
    test_tools_list_shows_promote_under_read_write,
    test_promote_call_rejected_under_read_only,
    test_get_node_roundtrip_logs_to_query_log,
    test_get_node_miss_logs_empty_read,
    test_promote_tool_success_writes_edge_with_mcp_session,
    test_promote_tool_overwrites_spoofed_mcp_session,
    test_promote_rate_limit_surfaces_as_jsonrpc_error,
    test_subprocess_read_only_hides_promote,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
            failed += 1
    if failed:
        sys.exit(1)
    print(f"\n{len(TESTS)}/{len(TESTS)} PASS")
