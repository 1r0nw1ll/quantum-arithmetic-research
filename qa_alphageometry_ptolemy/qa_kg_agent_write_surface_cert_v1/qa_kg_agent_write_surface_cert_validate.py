# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 internal cert validator; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), tools/qa_kg_mcp/server.py (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
"""QA-KG Agent Write Surface Cert [255] v1 validator.

QA_COMPLIANCE = "cert_validator — validates MCP agent-write surface, no empirical QA state machine"

Phase 6: validates that the agent-facing MCP surface is exactly four tools,
agent-authored nodes cannot gain higher authority without explicit
delete+recreate, direct writes to qa_kg.db bypassing MCP are surfaced by
qa_security_audit, the per-session rate limit fires at cap, and every
MCP tool call is audited.

Gates (W1 through W8; see README.md for full descriptions):
  W1 (HARD) MCP tool surface = exactly four tools; no upsert_node / Node
            construction inside tools/qa_kg_mcp/.
  W2 (HARD) kg.upsert_node(authority='agent',...) callsites confined to
            extractors/agent_notes.py + tests/.
  W3a (HARD) security audit flags direct-DB-write Bash in wrapper ledger.
  W3b (HARD) security audit flags promoted-from edges missing mcp_session.
  W4 (HARD) rate_limit.increment raises at cap.
  W5 (HARD) authority immutable both directions.
  W6 (HARD) READ_ONLY MCP session omits qa_kg_promote_agent_note.
  W7 (HARD) every MCP tool call writes to query_log.
  W8 (HARD) no except-Exception-pass / bare except: pass swallows in
            tools/qa_kg_mcp/ or tools/qa_kg/_audit.py.
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates MCP agent-write surface, no empirical QA state machine"

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_CERT_DIR = _HERE.parent
_REPO = _HERE.parents[2]

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


EXPECTED_TOOL_NAMES = frozenset({
    "qa_kg_search", "qa_kg_get_node", "qa_kg_neighbors",
    "qa_kg_promote_agent_note",
})


# --- W1 -------------------------------------------------------------------


def check_w1_mcp_surface_is_four_tools() -> tuple[str, str]:
    """W1 HARD: AST scan — TOOL_SCHEMAS has exactly 4 entries; no Node
    constructors or upsert_node calls inside tools/qa_kg_mcp/."""
    server_path = _REPO / "tools" / "qa_kg_mcp" / "server.py"
    if not server_path.exists():
        return "FAIL", f"server.py missing at {server_path}"
    tree = ast.parse(server_path.read_text(encoding="utf-8"),
                     filename=str(server_path))

    # (a) Tool names come from the TOOL_SCHEMAS dict literal. Accepts both
    # plain Assign (TOOL_SCHEMAS = {...}) and AnnAssign with type annotation
    # (TOOL_SCHEMAS: dict[...] = {...}).
    tool_names: set[str] = set()
    for node in ast.walk(tree):
        target_name: str | None = None
        value: ast.AST | None = None
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            target_name = node.targets[0].id
            value = node.value
        elif (isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)):
            target_name = node.target.id
            value = node.value
        if target_name != "TOOL_SCHEMAS" or value is None:
            continue
        if not isinstance(value, ast.Dict):
            return "FAIL", "TOOL_SCHEMAS must be a dict literal"
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                tool_names.add(key.value)
    if tool_names != EXPECTED_TOOL_NAMES:
        return (
            "FAIL",
            f"TOOL_SCHEMAS = {sorted(tool_names)} != expected "
            f"{sorted(EXPECTED_TOOL_NAMES)}"
        )

    # (b) No Node construction or upsert_node calls anywhere under
    # tools/qa_kg_mcp/.
    mcp_dir = _REPO / "tools" / "qa_kg_mcp"
    banned: list[str] = []
    for py_path in sorted(mcp_dir.rglob("*.py")):
        if "/tests/" in py_path.as_posix() or py_path.name == "__init__.py":
            continue
        try:
            subtree = ast.parse(py_path.read_text(encoding="utf-8"),
                                filename=str(py_path))
        except SyntaxError as exc:
            return "FAIL", f"{py_path.name}: syntax error {exc}"
        for node in ast.walk(subtree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Node(...), Node.source_work(...), Node.source_claim(...)
            if isinstance(func, ast.Name) and func.id == "Node":
                banned.append(f"{py_path.name}:{node.lineno} Node(...)")
            elif isinstance(func, ast.Attribute) and (
                (isinstance(func.value, ast.Name) and func.value.id == "Node")
            ):
                banned.append(
                    f"{py_path.name}:{node.lineno} Node.{func.attr}(...)"
                )
            # kg.upsert_node(...), anything.upsert_node(...)
            elif isinstance(func, ast.Attribute) and func.attr == "upsert_node":
                banned.append(
                    f"{py_path.name}:{node.lineno} .upsert_node(...)"
                )
    if banned:
        return (
            "FAIL",
            f"MCP scope contains node-creation callsites: "
            + "; ".join(banned[:5])
        )

    return "PASS", f"MCP surface = {sorted(tool_names)}; no node-creation callsites"


# --- W2 -------------------------------------------------------------------


def check_w2_agent_upsert_confined_to_extractor() -> tuple[str, str]:
    """W2 HARD: kg.upsert_node(authority='agent',...) call-sites live
    only in extractors/agent_notes.py + tests/.

    AST walk over tools/qa_kg/ + qa_alphageometry_ptolemy/ for Call nodes
    whose function resolves to .upsert_node and whose kwargs contain
    authority="agent" (OR whose arg Node constructor has authority="agent").
    """
    # Allowed callsites:
    #   - the extractor that IS the agent-write path
    #   - qa_kg test suites (ephemeral agent fixtures)
    #   - qa_kg cert-family validators (each needs ephemeral agent nodes to
    #     exercise its own firewall gates — [227] FE, [228] D6, [255] W5,
    #     etc.). Anything else that sets authority='agent' is a leak.
    allowed_prefixes = (
        "tools/qa_kg/extractors/agent_notes.py",
        "tools/qa_kg/tests/",
    )
    allowed_glob_roots = (
        "qa_alphageometry_ptolemy/qa_kg_",
    )
    search_roots = [
        _REPO / "tools" / "qa_kg",
        _REPO / "qa_alphageometry_ptolemy",
    ]
    viols: list[str] = []
    pattern = re.compile(
        r"""authority\s*=\s*["']agent["']""",
    )
    for root in search_roots:
        for py in sorted(root.rglob("*.py")):
            rel = py.relative_to(_REPO).as_posix()
            if any(rel.startswith(p) for p in allowed_prefixes):
                continue
            if any(rel.startswith(p) for p in allowed_glob_roots):
                continue
            text = py.read_text(encoding="utf-8")
            if "authority" not in text or "agent" not in text:
                continue
            # Must also contain an upsert_node or Node constructor near
            # the authority="agent" line — otherwise string is incidental.
            for line_no, line in enumerate(text.splitlines(), 1):
                if not pattern.search(line):
                    continue
                ctx_lines = text.splitlines()[
                    max(0, line_no - 5):line_no + 4
                ]
                ctx = "\n".join(ctx_lines)
                if ("upsert_node" in ctx
                        or "Node(" in ctx or "Node.source_" in ctx):
                    viols.append(f"{rel}:{line_no}")
    if viols:
        return (
            "FAIL",
            f"agent-authority callsites outside extractor/tests: "
            + "; ".join(viols[:5])
        )
    return "PASS", "agent-authority upserts confined to extractor + tests"


# --- W3a / W3b ------------------------------------------------------------


def check_w3a_direct_write_detector_fires() -> tuple[str, str]:
    """W3a HARD: qa_security_audit.check_qa_kg_db_direct_writes flags an
    injected direct-write Bash entry in the wrapper ledger."""
    from tools.qa_security_audit import RESULTS as _SA_RESULTS

    # Inject synthetic ledger at a temp location and point the module at
    # it via monkey-patched ROOT.
    with tempfile.TemporaryDirectory() as td:
        fake_repo = Path(td)
        (fake_repo / "llm_qa_wrapper" / "ledger").mkdir(parents=True)
        ledger = fake_repo / "llm_qa_wrapper" / "ledger" / "enforced.jsonl"
        record = {
            "tool": "Bash",
            "decision": "ALLOW",
            "ts": "2026-04-16T22:00:00Z",
            "policy_payload": {
                "tool_input": {
                    "command": 'sqlite3 tools/qa_kg/qa_kg.db "INSERT INTO nodes VALUES (...)"'
                }
            },
        }
        ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")

        import tools.qa_security_audit as _sa
        old_root = _sa.ROOT
        _sa.ROOT = fake_repo
        _SA_RESULTS["pass"].clear()
        _SA_RESULTS["warn"].clear()
        _SA_RESULTS["fail"].clear()
        try:
            _sa.check_qa_kg_db_direct_writes()
        finally:
            _sa.ROOT = old_root
        if not _SA_RESULTS["fail"]:
            return (
                "FAIL",
                f"detector did NOT flag injected direct-write; "
                f"pass={_SA_RESULTS['pass']}, fail={_SA_RESULTS['fail']}"
            )
        flagged = _SA_RESULTS["fail"][0]
        if "direct-write" not in flagged:
            return "FAIL", f"unexpected FAIL message: {flagged}"
    return "PASS", "direct-write detector fires on synthetic ledger entry"


def check_w3b_mcp_provenance_detector_fires() -> tuple[str, str]:
    """W3b HARD: qa_security_audit.check_mcp_provenance flags a
    promoted-from edge whose provenance lacks mcp_session."""
    from tools.qa_kg import connect
    from tools.qa_kg.kg import PROMOTED_FROM_EDGE
    import tools.qa_security_audit as _sa

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "qa_kg.db"
        kg = connect(db_path)
        # Seed two nodes.
        kg.conn.execute(
            "INSERT INTO nodes(id,node_type,title,body,tier,authority,"
            "epistemic_status,lifecycle_state,confidence,created_ts,updated_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("a:1", "Thought", "a", "b", "cosmos", "agent",
             "observation", "current", 1.0, "t", "t"),
        )
        kg.conn.execute(
            "INSERT INTO nodes(id,node_type,title,body,tier,authority,"
            "epistemic_status,lifecycle_state,confidence,created_ts,updated_ts) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("r:1", "Rule", "r", "b", "cosmos", "internal",
             "interpretation", "current", 1.0, "t", "t"),
        )
        # Insert a promoted-from edge with NO mcp_session marker.
        kg.conn.execute(
            "INSERT INTO edges(src_id,dst_id,edge_type,confidence,method,"
            "provenance,via_cert,created_ts) VALUES (?,?,?,?,?,?,?,?)",
            ("a:1", "r:1", PROMOTED_FROM_EDGE, 1.0, "promote",
             json.dumps({"broadcast_payload_snapshot": {"ts": "2026-01"}}),
             "225", "t"),
        )
        kg.conn.commit()
        kg.conn.close()

        fake_repo = Path(td)
        tools_dir = fake_repo / "tools" / "qa_kg"
        tools_dir.mkdir(parents=True)
        (tools_dir / "qa_kg.db").write_bytes(db_path.read_bytes())
        old_root = _sa.ROOT
        _sa.ROOT = fake_repo
        _sa.RESULTS["pass"].clear()
        _sa.RESULTS["warn"].clear()
        _sa.RESULTS["fail"].clear()
        try:
            _sa.check_mcp_provenance()
        finally:
            _sa.ROOT = old_root
        if not _sa.RESULTS["fail"]:
            return (
                "FAIL",
                f"provenance detector did NOT flag missing mcp_session; "
                f"pass={_sa.RESULTS['pass']}, fail={_sa.RESULTS['fail']}"
            )
    return "PASS", "provenance detector fires on edge missing mcp_session"


# --- W4 -------------------------------------------------------------------


def check_w4_rate_limit_raises_at_cap() -> tuple[str, str]:
    """W4 HARD: rate_limit.increment raises at cap+1."""
    from tools.qa_kg_mcp.rate_limit import (
        RateLimitExceeded, increment, get_count,
    )
    with tempfile.TemporaryDirectory() as td:
        lpath = Path(td) / "_agent_writes.json"
        for _ in range(3):
            increment("s", ledger_path=lpath, max_writes=3)
        try:
            increment("s", ledger_path=lpath, max_writes=3)
        except RateLimitExceeded as exc:
            if "s" in str(exc):
                return "PASS", (
                    f"cap enforced (count={get_count('s', ledger_path=lpath)}, "
                    f"raised: {str(exc)[:60]}…)"
                )
            return "FAIL", f"raised but message lacks session: {exc}"
    return "FAIL", "cap+1 did not raise"


# --- W5 -------------------------------------------------------------------


def check_w5_authority_immutable_both_directions() -> tuple[str, str]:
    """W5 HARD: authority is immutable both directions + same-auth idempotent."""
    from tools.qa_kg import connect
    from tools.qa_kg.kg import FirewallViolation, Node

    with tempfile.TemporaryDirectory() as td:
        kg = connect(Path(td) / "t.db")

        # (a) agent → internal blocked
        kg.upsert_node(Node(
            id="x:1", node_type="Thought", title="t", body="b",
            authority="agent", epistemic_status="observation",
        ))
        try:
            kg.upsert_node(Node(
                id="x:1", node_type="Thought", title="t", body="b",
                authority="internal", epistemic_status="observation",
            ))
        except FirewallViolation as exc:
            if "authority_immutable" not in str(exc):
                return "FAIL", f"raised but wrong reason: {exc}"
        else:
            return "FAIL", "agent→internal not blocked"

        # (b) primary → agent blocked (silent-downgrade guard)
        kg.upsert_node(Node(
            id="sc:p", node_type="Claim", title="p", body="q",
            authority="primary", epistemic_status="source_claim",
            method="manual", source_locator="file:CLAUDE.md",
        ))
        try:
            kg.upsert_node(Node(
                id="sc:p", node_type="Claim", title="p", body="q",
                authority="agent", epistemic_status="observation",
            ))
        except FirewallViolation as exc:
            if "authority_immutable" not in str(exc):
                return "FAIL", f"downgrade raised but wrong reason: {exc}"
        else:
            return "FAIL", "primary→agent downgrade not blocked"

        # (c) agent → agent idempotent
        kg.upsert_node(Node(
            id="x:1", node_type="Thought", title="t", body="b2",
            authority="agent", epistemic_status="observation",
        ))
        row = kg.get("x:1")
        if row["body"] != "b2" or row["authority"] != "agent":
            return "FAIL", "same-auth upsert did not round-trip"

    return "PASS", "agent→internal blocked, primary→agent blocked, same-auth idempotent"


# --- W6 -------------------------------------------------------------------


def check_w6_read_only_hides_promote() -> tuple[str, str]:
    """W6 HARD: READ_ONLY server omits promote from tools/list."""
    from tools.qa_kg import connect
    from tools.qa_kg._audit import AuditLog
    from tools.qa_kg_mcp.capabilities import Capability
    from tools.qa_kg_mcp.server import MCPServer

    with tempfile.TemporaryDirectory() as td:
        kg = connect(Path(td) / "t.db")
        audit = AuditLog(conn=kg.conn)

        # READ_ONLY
        srv_ro = MCPServer(
            capability=Capability.READ_ONLY, kg=kg, audit_log=audit,
            session_id="w6-ro",
        )
        ro_resp = srv_ro.handle({"jsonrpc": "2.0", "id": 1,
                                 "method": "tools/list"})
        ro_names = {t["name"] for t in ro_resp["result"]["tools"]}
        if "qa_kg_promote_agent_note" in ro_names:
            return "FAIL", f"READ_ONLY exposed promote: {sorted(ro_names)}"
        if ro_names != {"qa_kg_search", "qa_kg_get_node", "qa_kg_neighbors"}:
            return "FAIL", f"READ_ONLY tools/list unexpected: {sorted(ro_names)}"

        # Also confirm tools/call rejects
        call_resp = srv_ro.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "qa_kg_promote_agent_note",
                       "arguments": {}},
        })
        if "error" not in call_resp or call_resp["error"]["code"] != -32601:
            return "FAIL", f"READ_ONLY promote call not rejected: {call_resp}"

        # READ_WRITE
        srv_rw = MCPServer(
            capability=Capability.READ_WRITE, kg=kg, audit_log=audit,
            session_id="w6-rw",
        )
        rw_resp = srv_rw.handle({"jsonrpc": "2.0", "id": 3,
                                 "method": "tools/list"})
        rw_names = {t["name"] for t in rw_resp["result"]["tools"]}
        if rw_names != EXPECTED_TOOL_NAMES:
            return "FAIL", f"READ_WRITE tools/list unexpected: {sorted(rw_names)}"

    return "PASS", "READ_ONLY hides promote; READ_WRITE exposes all 4"


# --- W7 -------------------------------------------------------------------


def check_w7_audit_every_tool_call() -> tuple[str, str]:
    """W7 HARD: every MCP tool call writes to query_log.
    Reads log k rows with rank >= 0; write logs 1 row with rank = -1.
    """
    from tools.qa_kg import connect
    from tools.qa_kg._audit import AuditLog
    from tools.qa_kg.kg import Node
    from tools.qa_kg_mcp.capabilities import Capability
    from tools.qa_kg_mcp.server import MCPServer
    from tools.qa_kg.kg import _current_git_head

    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        kg = connect(db)
        audit = AuditLog(conn=kg.conn)
        ledger = Path(td) / "_meta_ledger.json"
        awpath = Path(td) / "_agent_writes.json"

        kg.upsert_node(Node(
            id="axiom:A1", node_type="Axiom",
            title="No-Zero axiom", body="states in 1..N",
            authority="primary", epistemic_status="axiom",
        ))
        kg.upsert_node(Node(
            id="agent:a1", node_type="Thought",
            title="agent note", body="agent note body",
            authority="agent", epistemic_status="observation",
        ))
        kg.upsert_node(Node(
            id="rule:p1", node_type="Rule",
            title="rule", body="internal rule body",
            authority="internal", epistemic_status="interpretation",
        ))
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ledger.write_text(json.dumps({
            "225": {"status": "PASS", "ts": now_iso,
                    "git_head": _current_git_head()}
        }), encoding="utf-8")

        srv = MCPServer(
            capability=Capability.READ_WRITE, kg=kg, audit_log=audit,
            session_id="w7-s", agent_writes_path=awpath, ledger_path=ledger,
        )

        # Read: qa_kg_search — should log at least 1 row with rank >= 0
        resp_s = srv.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "qa_kg_search",
                       "arguments": {"query": "axiom",
                                     "min_authority": "primary", "k": 3}},
        })
        if "error" in resp_s:
            return "FAIL", f"search error: {resp_s['error']}"

        # Read: qa_kg_get_node
        srv.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "qa_kg_get_node",
                       "arguments": {"node_id": "axiom:A1"}},
        })

        # Read: qa_kg_neighbors (expect 0 neighbors on axiom:A1; still must log)
        srv.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "qa_kg_neighbors",
                       "arguments": {"node_id": "axiom:A1",
                                     "direction": "both"}},
        })

        # Write: qa_kg_promote_agent_note
        resp_p = srv.handle({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "qa_kg_promote_agent_note",
                       "arguments": {
                           "node_id": "agent:a1", "via_cert": "225",
                           "promoter_node_id": "rule:p1",
                           "broadcast_payload": {
                               "ts": now_iso, "session": "w7-s",
                           }}},
        })
        if "error" in resp_p:
            return "FAIL", f"promote error: {resp_p['error']}"

        # Verify query_log shape for this session.
        reads = kg.conn.execute(
            "SELECT COUNT(*) AS n FROM query_log "
            "WHERE session=? AND rank >= 0",
            ("w7-s",),
        ).fetchone()["n"]
        writes = kg.conn.execute(
            "SELECT COUNT(*) AS n FROM query_log "
            "WHERE session=? AND rank = -1",
            ("w7-s",),
        ).fetchone()["n"]
        if writes != 1:
            return "FAIL", f"promote produced {writes} write rows, want 1"
        if reads < 1:
            return "FAIL", f"reads produced {reads} rows, want >= 1"
        # get_node hit logs exactly one row → at least one read row must
        # correspond to axiom:A1.
        axiom_reads = kg.conn.execute(
            "SELECT COUNT(*) AS n FROM query_log "
            "WHERE session=? AND node_id=? AND rank >= 0",
            ("w7-s", "axiom:A1"),
        ).fetchone()["n"]
        if axiom_reads < 1:
            return "FAIL", "get_node did not log a row for axiom:A1"

    return "PASS", f"audit log shape correct (reads>=1, writes=1)"


# --- W8 -------------------------------------------------------------------


def check_w8_no_swallows_ast() -> tuple[str, str]:
    """W8 HARD: AST scan for except-Exception-pass / bare except: pass in
    tools/qa_kg_mcp/ and tools/qa_kg/_audit.py. Mirrors [228] D7."""
    targets: list[Path] = [_REPO / "tools" / "qa_kg" / "_audit.py"]
    mcp_dir = _REPO / "tools" / "qa_kg_mcp"
    for p in sorted(mcp_dir.rglob("*.py")):
        if "/tests/" in p.as_posix() or p.name == "__init__.py":
            continue
        targets.append(p)

    viols: list[str] = []
    for path in targets:
        if not path.exists():
            viols.append(f"{path.name}: file missing")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"),
                             filename=str(path))
        except SyntaxError as exc:
            viols.append(f"{path.name}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = node.body
            if node.type is None and (
                not body or (len(body) == 1 and isinstance(body[0], ast.Pass))
            ):
                viols.append(
                    f"{path.name}:{node.lineno} bare except: pass (or empty)"
                )
                continue
            if isinstance(node.type, ast.Name) and node.type.id in (
                "Exception", "BaseException"
            ):
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    viols.append(
                        f"{path.name}:{node.lineno} "
                        f"except {node.type.id}: pass swallow"
                    )
    if viols:
        return "FAIL", f"{len(viols)} swallow(s): " + "; ".join(viols[:5])
    return "PASS", f"no swallows across {len(targets)} files"


# --- Main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true",
                   help="treat WARN as FAIL (CI-strict mode)")
    args = p.parse_args(argv)

    hard_fail = False

    def run(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        try:
            status, msg = fn()
        except (FileNotFoundError, json.JSONDecodeError,
                subprocess.CalledProcessError, ImportError) as exc:
            status, msg = (
                ("FAIL" if is_hard else "WARN"),
                f"error: {type(exc).__name__}: {exc}",
            )
        print(f"[{status}] {code}   {desc} — {msg}")
        if status == "FAIL" and (is_hard or args.strict):
            hard_fail = True
        if status == "WARN" and args.strict:
            hard_fail = True

    run("W1",  "MCP surface = 4 tools (AST)", check_w1_mcp_surface_is_four_tools, True)
    run("W2",  "Agent upserts confined to extractor", check_w2_agent_upsert_confined_to_extractor, True)
    run("W3a", "Direct DB-write detector fires", check_w3a_direct_write_detector_fires, True)
    run("W3b", "MCP provenance detector fires", check_w3b_mcp_provenance_detector_fires, True)
    run("W4",  "Rate limit raises at cap", check_w4_rate_limit_raises_at_cap, True)
    run("W5",  "Authority immutable both directions", check_w5_authority_immutable_both_directions, True)
    run("W6",  "READ_ONLY hides promote", check_w6_read_only_hides_promote, True)
    run("W7",  "Every MCP tool call audited", check_w7_audit_every_tool_call, True)
    run("W8",  "No except-pass swallows", check_w8_no_swallows_ast, True)

    if hard_fail:
        print("[FAIL] QA-KG agent write surface cert [255] v1")
        return 1
    print("[PASS] QA-KG agent write surface cert [255] v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
