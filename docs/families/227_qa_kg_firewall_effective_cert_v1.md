<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG infrastructure cert documentation -->
# [227] QA-KG Firewall Effective Cert v1

**Status:** Active (Phase 2)
**Validator:** `qa_alphageometry_ptolemy/qa_kg_firewall_effective_cert_v1/qa_kg_firewall_effective_cert_validate.py`
**Dependencies:** [225] v3 (graph consistency), [252] (epistemic fields), `tools/qa_kg/kg.py` (promote protocol)

## Purpose

Validates that the Phase 2 Theorem NT firewall is effective: agent-authority
nodes cannot emit causal edges into the knowledge graph without going through
the `kg.promote()` protocol, which requires a DB-backed `promoted-from` edge,
a fresh `_meta_ledger.json` entry, and a broadcast payload snapshot.

## Gates

| Gate | Level | Description |
|------|-------|-------------|
| FE1 | HARD | No unpromoted agent causal edges in DB |
| FE2 | HARD | `via_cert` on promoted-from edges resolves to PASS in `_meta_ledger.json` within 14d + git_head match |
| FE3 | HARD | No promoted-from cycles |
| FE4 | HARD | Ephemeral test DB: FirewallViolation on unauthorized agent causal; success after promote |
| FE5 | WARN | Oldest unpromoted agent note |
| FE6 | HARD | promoted-from provenance contains `broadcast_payload_snapshot` with required keys |

## Architecture

The firewall operates at two levels:

1. **Policy level** (`orbit.py::edge_allowed`): Returns `False` unconditionally
   for `authority=agent` + causal edge type. No string argument can bypass this.

2. **DB-backed bypass** (`kg.py::upsert_edge`): When `edge_allowed` rejects an
   agent causal edge, `upsert_edge` queries the DB for a `promoted-from` edge
   on the source node. Only if such an edge exists (created by `kg.promote()`)
   is the causal edge allowed through.

3. **Promote protocol** (`kg.py::promote`): Validates agent identity, promoter
   authority, ledger staleness (14d + git HEAD match), and broadcast timestamp
   (±60s window) before creating the `promoted-from` edge with provenance.

## Provenance schema

Every `promoted-from` edge carries a JSON provenance field:

```json
{
  "session": "<session name>",
  "signed_ts": "<ISO8601>",
  "promoter_node_id": "<node id>",
  "broadcast_payload_snapshot": {
    "ts": "<ISO8601>",
    "session": "<session>",
    "event_type": "kg_promotion"
  }
}
```

FE6 validates this structure on every promoted-from edge.
