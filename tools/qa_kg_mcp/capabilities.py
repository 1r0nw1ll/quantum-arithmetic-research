# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 capability masks; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
"""QA-KG Phase 6 MCP capability masks.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Two capability classes:
  READ_ONLY   — qa_kg_search, qa_kg_get_node, qa_kg_neighbors
  READ_WRITE  — READ_ONLY + qa_kg_promote_agent_note

Capability is declared at server spawn via the `--cap` CLI flag; the
agent cannot self-elevate because no MCP message mutates this mask.
A READ_ONLY server's tools/list response omits the promote tool
entirely, so the agent has no way to even discover the endpoint.
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from enum import Enum


class Capability(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


READ_TOOLS = frozenset({"qa_kg_search", "qa_kg_get_node", "qa_kg_neighbors"})
WRITE_TOOLS = frozenset({"qa_kg_promote_agent_note"})
ALL_TOOLS = READ_TOOLS | WRITE_TOOLS


def visible_tools(cap: Capability) -> frozenset[str]:
    """Return the tool names visible to a session with this capability."""
    if cap is Capability.READ_ONLY:
        return READ_TOOLS
    if cap is Capability.READ_WRITE:
        return ALL_TOOLS
    raise ValueError(f"unknown capability: {cap!r}")


def parse_capability(value: str) -> Capability:
    """Parse the CLI --cap flag value into a Capability enum.

    Accepts the enum value ('read_only' or 'read_write'). Raises ValueError
    on anything else — including empty string or typos — so mis-typed
    flags fail loudly at server startup rather than silently defaulting.
    """
    if value not in {c.value for c in Capability}:
        raise ValueError(
            f"invalid --cap {value!r}; must be one of "
            f"{sorted(c.value for c in Capability)}"
        )
    return Capability(value)
