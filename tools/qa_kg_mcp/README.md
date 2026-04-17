# QA-KG MCP Server (Phase 6)

Agent-facing MCP stdio server for QA-MEM. Exposes EXACTLY four tools:

| Tool | Capability | Purpose |
|---|---|---|
| `qa_kg_search` | READ_ONLY | Authority-tiered ranked retrieval |
| `qa_kg_get_node` | READ_ONLY | Single-node lookup + epistemic fields |
| `qa_kg_neighbors` | READ_ONLY | Incoming/outgoing edges for a node |
| `qa_kg_promote_agent_note` | READ_WRITE | Promote an AgentNote via `kg.promote()` |

The write surface is deliberately minimal: **agents cannot create nodes via MCP.** The only agent-write path is
(1) the extractor (`tools/qa_kg/extractors/agent_notes.py`) which creates AgentNote rows from collab-bus events,
and (2) `qa_kg_promote_agent_note` which escalates an existing AgentNote so it can emit causal edges.

## Starting the server

```bash
# Read-only capability: promote tool is hidden entirely
python -m tools.qa_kg_mcp.server --cap read_only

# Read-write capability: all four tools visible
python -m tools.qa_kg_mcp.server --cap read_write
```

The `--cap` flag is the ONLY capability declaration path. Agents cannot self-elevate — no MCP message mutates the mask after startup.

Optional flags: `--db <path>`, `--agent-writes <path>`, `--ledger <path>`, `--session <id>`.

## Rate limit

Each session may make at most `QA_KG_MCP_MAX_WRITES` (default 50) promote calls. Counter lives in
`qa_alphageometry_ptolemy/_agent_writes.json` (gitignored, separate from `_meta_ledger.json` per plan v2 M1).

The counter decays when the server receives a `session_done` broadcast for its session id.

### Recovering a crashed session

If the server crashes before broadcasting `session_done`, its counter persists. The operator-authorized recovery path:

```bash
python -m tools.qa_kg_mcp.cli reset-writes <session-id>
```

This is deliberately manual — long-running sessions are protected because auto-decay only fires on explicit
`session_done`, not on broadcast timeout or process exit.

## Audit log

Every tool call writes to `query_log` in `qa_kg.db` before returning:

- Reads: one row per returned node, `rank = 0..k-1`, `session = <mcp_session>`.
- Writes: one row, `rank = -1` sentinel, `session = <mcp_session>`.

Cert `[255]` W7 exercises this contract.

## Cert coverage

`qa_alphageometry_ptolemy/qa_kg_agent_write_surface_cert_v1/` gates (all HARD):

- W1 MCP tool surface cannot create non-AgentNote types (AST scan)
- W2 `kg.upsert_node(authority="agent", ...)` confined to `extractors/agent_notes.py` + tests
- W3a `cert_gate_hook.py` blocks direct Bash writes to `qa_kg.db`
- W3b `qa_security_audit` flags `promoted-from` edges without an `mcp_session` provenance marker
- W4 Rate limit raises at cap
- W5 Authority immutable both directions
- W6 READ_ONLY capability hides the promote tool in `tools/list`
- W7 Every MCP tool call logs to `query_log`
- W8 No `except Exception: pass` swallows in `tools/qa_kg_mcp/` or `tools/qa_kg/_audit.py`

## MCP protocol

Stdlib-only JSON-RPC 2.0 over stdio. Methods:

- `initialize` — returns `{protocolVersion, capabilities, serverInfo}`. Client may pass
  `clientInfo.sessionId` to align the MCP session with its collab-bus session.
- `tools/list` — returns the tool list visible under the declared capability.
- `tools/call` — invokes a tool. Rate-limit and firewall errors surface as JSON-RPC error
  code `-32000`; invalid params as `-32602`; unknown tools as `-32601`.
- `shutdown` — returns `{}` and the server exits on stdin EOF.
