<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 internal cert family; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026), tools/qa_kg_mcp/server.py (Dale, 2026) -->
# Cert Family [255] — QA_KG_AGENT_WRITE_SURFACE_CERT.v1

**Phase 6 of QA-MEM.** Validates the agent-facing MCP surface: the surface
is exactly four tools, agent-authored nodes cannot gain higher authority
without explicit delete+recreate, direct writes to the backing SQLite
file that bypass the MCP server are surfaced by the security audit, the
per-session rate limit fires at cap, and every MCP tool call is audited.

Landing this cert (with all gates PASS) is the alpha-bar increment: after
Phase 6 lands, QA-MEM may be described as "alpha agent memory" and
"authoritative project memory" in scope docs and memory. See
`docs/specs/QA_MEM_SCOPE.md` and `memory/project_qa_mem_review_role.md`.

## Primary sources

- `docs/specs/QA_MEM_SCOPE.md` (Dale, 2026)
- `memory/project_qa_mem_review_role.md` (Dale, 2026) — reviewer-role +
  alpha-bar discipline that gates Phase 6 landing
- `tools/qa_kg_mcp/server.py` (Dale, 2026) — MCP surface under test
- `tools/qa_kg_mcp/rate_limit.py` (Dale, 2026) — rate-limit module
- `tools/qa_kg/_audit.py` (Dale, 2026) — audit log helper
- `tools/qa_security_audit.py` (Dale, 2026) — forensic checks

## Gates

### W1 (HARD) — MCP tool surface is exactly four tools

AST scan of `tools/qa_kg_mcp/server.py` confirms:

1. `TOOL_SCHEMAS` maps exactly four tool names, matching the whitelist
   `{qa_kg_search, qa_kg_get_node, qa_kg_neighbors, qa_kg_promote_agent_note}`.
2. No `Node(...)` or `Node.source_work(...)` / `Node.source_claim(...)`
   call-site inside `tools/qa_kg_mcp/` — the MCP server cannot construct
   new nodes, so it cannot create `Cert / Axiom / SourceClaim / SourceWork
   / Theorem` nodes regardless of what tools are registered.
3. No `kg.upsert_node(...)` call-site inside `tools/qa_kg_mcp/`. The MCP
   server may read nodes and call `kg.promote()` (which writes a
   `promoted-from` edge, not a node), but it never inserts or updates a
   node row.

### W2 (HARD) — Agent-write entry-point is the extractor alone

Grep gate: every `kg.upsert_node(...)` call-site that passes
`authority="agent"` lives in `tools/qa_kg/extractors/agent_notes.py` or
`tools/qa_kg/tests/`. The MCP server has no agent-write callsite.

### W3a (HARD) — Direct-write Bash detection in the wrapper ledger

`tools/qa_security_audit.py::check_qa_kg_db_direct_writes` scans
`llm_qa_wrapper/ledger/enforced.jsonl` for any ALLOWed Bash call
containing `sqlite3 ... qa_kg.db ... INSERT/UPDATE/DELETE/REPLACE/DROP/
ALTER`. Any hit is a FAIL. The cert invokes the scan function directly
on a synthetic ledger fixture with an injected direct-write entry to
prove the detector fires, then again on the real live ledger (pass gate).

### W3b (HARD) — MCP provenance forensic check

`tools/qa_security_audit.py::check_mcp_provenance` scans `qa_kg.db` for
`promoted-from` edges whose `provenance.broadcast_payload_snapshot` is
missing `mcp_session`. Ephemeral DB fixture inserts (1) a promote via
the MCP server (stamp present) and (2) a raw `kg.promote()` call without
`mcp_session` (no stamp). The check must flag (2) and ignore (1).

### W4 (HARD) — Rate limit fires at cap

Ephemeral `_agent_writes.json` ledger. With cap = 3, the 4th
`rate_limit.increment(session)` raises `RateLimitExceeded` naming the
session.

### W5 (HARD) — Authority is immutable both directions

Ephemeral DB:
- Insert agent node. Attempt `upsert_node` with same id + `authority=
  internal` → `FirewallViolation("authority_immutable")`.
- Insert primary `source_claim` node. Attempt `upsert_node` with same id
  + `authority=agent` → `FirewallViolation("authority_immutable")`.
- Re-upsert with the same authority → no raise (idempotent).

### W6 (HARD) — READ_ONLY capability hides the promote tool

Spin up the MCP server in-process with `Capability.READ_ONLY`. Call
`handle({"method": "tools/list"})` and assert the response contains only
the three read tools. Repeat with `Capability.READ_WRITE` and assert
four tools are visible. Also verify `handle({"method": "tools/call",
"params": {"name": "qa_kg_promote_agent_note"}})` under READ_ONLY
returns JSON-RPC error code `-32601`.

### W7 (HARD) — Every MCP tool call logs to query_log

Ephemeral DB + in-process server. For each of the four tools, invoke
the tool and assert the expected `query_log` row shape:

- Read tools: one row per returned `node_id`, `rank >= 0`, `session =
  mcp_session`.
- Write tool (`qa_kg_promote_agent_note`): exactly one row, `rank = -1`
  sentinel, `node_id = <promote target>`.

### W8 (HARD) — No except-Exception-pass swallows

AST scan over `tools/qa_kg_mcp/*.py` (excluding `__init__.py`, tests)
and `tools/qa_kg/_audit.py`. Mirrors `[228]` D7 / `[254]` R8 / `[225]`
KG10 pattern.

## Running the validator

```bash
python qa_alphageometry_ptolemy/qa_kg_agent_write_surface_cert_v1/qa_kg_agent_write_surface_cert_validate.py
```

Or via the full meta-validator sweep:

```bash
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

## Alpha-bar flip

After `[255]` PASS lands in `_meta_ledger.json`, the same commit flips
the marketing language in `docs/specs/QA_MEM_SCOPE.md` and
`memory/project_qa_mem_review_role.md` from "Phase 6 remains gating" to
"alpha agent memory" + "authoritative project memory". Phase 4.5
(corpus scale) and Phase 5.1 (pinned-source D2/D3) are reclassified as
operational hardening — they are NOT alpha-bar items.
