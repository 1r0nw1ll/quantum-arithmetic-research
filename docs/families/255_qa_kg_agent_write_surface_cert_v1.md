<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 cert family doc; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), tools/qa_kg_mcp/server.py (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
# [255] QA-KG Agent Write Surface Cert v1

**Status:** PASS (W1, W2, W3a, W3b, W4, W5, W6, W7, W8 — all HARD)
**Created:** 2026-04-16 (Phase 6 of QA-MEM)
**Source:** Will Dale + Claude; `docs/specs/QA_MEM_SCOPE.md` (Dale, 2026); `tools/qa_kg_mcp/server.py` (Dale, 2026); `tools/qa_kg_mcp/rate_limit.py` (Dale, 2026); `tools/qa_kg/_audit.py` (Dale, 2026); `tools/qa_security_audit.py` (Dale, 2026); `memory/project_qa_mem_review_role.md` (Dale, 2026).

## Claim

QA-MEM Phase 6 ships the **agent-integration firewall**: a minimal MCP
stdio server with exactly four tools, a capability mask that can hide
the write tool entirely, a per-session rate limit under flock, tamper-
evident `mcp_session` provenance stamping on every promote, immutable
authority (both directions), audit-log coverage on every tool call, and
two forensic detectors in `qa_security_audit` — one for direct Bash
writes to `qa_kg.db` in the LLM-wrapper ledger, one for
`promoted-from` edges missing the `mcp_session` marker.

The agent-write surface is EXACTLY one operation: **promote an existing
AgentNote**. The MCP server cannot create nodes. AgentNote rows are
inserted by `tools/qa_kg/extractors/agent_notes.py` reading collab-bus
events / OB thoughts with `originSessionId` / `collab_log_activity`
rows — NEVER by the agent directly.

Landing this cert with all gates PASS is the alpha-bar increment per
`memory/project_qa_mem_review_role.md`: after Phase 6 lands, QA-MEM may
be described as "alpha agent memory" and "authoritative project memory"
in scope docs. Phases 4.5 (corpus scale) and 5.1 (pinned-source D2/D3)
are reclassified as operational hardening — they are NOT alpha-bar
items per the original three-cert bar ([228] + [254] + [255]).

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| W1  | HARD | AST of `tools/qa_kg_mcp/server.py` — `TOOL_SCHEMAS` keys equal `{qa_kg_search, qa_kg_get_node, qa_kg_neighbors, qa_kg_promote_agent_note}`; no `Node(...)` or `.upsert_node(...)` callsite anywhere under `tools/qa_kg_mcp/`. |
| W2  | HARD | Grep/AST scan — every `kg.upsert_node(authority='agent', ...)` callsite lives in `tools/qa_kg/extractors/agent_notes.py`, `tools/qa_kg/tests/`, or a `qa_alphageometry_ptolemy/qa_kg_*_cert_v*/` validator (legitimate ephemeral fixtures). |
| W3a | HARD | `qa_security_audit.check_qa_kg_db_direct_writes` flags any ALLOWed Bash call in `llm_qa_wrapper/ledger/enforced.jsonl` matching `sqlite3 ... qa_kg.db ... INSERT/UPDATE/DELETE/REPLACE/DROP/ALTER`. Validator injects a synthetic ledger entry and asserts the detector fires. |
| W3b | HARD | `qa_security_audit.check_mcp_provenance` flags `promoted-from` edges whose `provenance.broadcast_payload_snapshot` lacks `mcp_session`. Validator inserts one such edge in an ephemeral DB and asserts the detector fires. |
| W4  | HARD | `rate_limit.increment` at cap = 3 raises `RateLimitExceeded` on the 4th call, with the session id in the message. |
| W5  | HARD | `kg.upsert_node` raises `FirewallViolation('authority_immutable')` on `agent→internal` AND on `primary→agent`. Same-authority re-upsert is idempotent. |
| W6  | HARD | In-process `MCPServer(Capability.READ_ONLY)` returns only `{qa_kg_search, qa_kg_get_node, qa_kg_neighbors}` from `tools/list`, and `tools/call` for `qa_kg_promote_agent_note` returns JSON-RPC error `-32601`. `Capability.READ_WRITE` exposes all four tools. |
| W7  | HARD | Every MCP tool call writes to `query_log`. Read tools: one row per returned node (rank >= 0). Write tool: one row, rank = -1 sentinel, `node_id = <promote target>`. Validator exercises all four tools in one session and asserts shape. |
| W8  | HARD | AST scan for `except Exception: pass` / `except BaseException: pass` / bare `except: pass` across `tools/qa_kg_mcp/*.py` (excluding tests / `__init__.py`) and `tools/qa_kg/_audit.py`. Mirrors [228] D7, [254] R8, [225] KG10. |

## Key design decisions (plan v2 review)

| ID | Decision | Rationale |
|----|----------|-----------|
| M1 | Per-session counter lives in `_agent_writes.json`, not `_meta_ledger.json` | Avoids a concurrent-writer race with `qa_meta_validator.py`. The two writers never share a file or flock. |
| M2 | Authority is immutable in **both** directions | Blocks both accidental upgrade (agent → primary) and silent downgrade (primary → agent from a buggy extractor re-run). Real corrections require explicit delete + recreate so the change is auditable. |
| M3 | `broadcast_payload` deep-copied AND `mcp_session` stamped AFTER the copy | An agent-provided `mcp_session` key is overwritten, not preserved. Caller's dict is unmodified. Spoofed stamps cannot survive the server. |
| S1 | Session counter decays on explicit `session_done` only | No polling. Crashed sessions are recovered via operator-authorized `python -m tools.qa_kg_mcp.cli reset-writes <session>`. |
| S2 | Marketing-language flip is full | Per the original three-cert bar ([228] + [254] + [255]), Phase 6 flips to "alpha agent memory" AND "authoritative project memory". Phases 4.5 / 5.1 are operational hardening, not gating. |
| N1 | W6 subprocess test uses try/finally with terminate → wait → kill escalation | Fixture teardown is not guaranteed on crash. |

## Compatibility

- Schema v4 (unchanged). No new columns. `query_log` reused as the audit surface (rank = -1 sentinel for writes).
- `_agent_writes.json` is a new file, gitignored, additive.
- `kg.promote()` grew two new optional kwargs (`mcp_session`, `agent_writes_path`). Pre-Phase-6 extractor callers pass neither; counter is not touched.
- `qa_meta_validator.py` gains a new entry for [255]; no change to pre-existing family validators.

## References

- `tools/qa_kg_mcp/server.py` — MCP stdio server
- `tools/qa_kg_mcp/capabilities.py` — capability mask
- `tools/qa_kg_mcp/rate_limit.py` — flock-guarded per-session counter
- `tools/qa_kg_mcp/cli.py` — operator CLI (`reset-writes`)
- `tools/qa_kg_mcp/README.md` — operator + architecture notes
- `tools/qa_kg/_audit.py` — query_log write helpers
- `tools/qa_security_audit.py` — W3a + W3b checks
- `qa_alphageometry_ptolemy/qa_kg_agent_write_surface_cert_v1/` — this cert
