"""QA-KG — indexed, firewalled view over repo artifacts.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Scope: a Candidate F retrieval-index [family 202] + keyword edge extractor +
schema-level firewall stub over cert families, MEMORY.md rules, OB thoughts,
and A-RAG archive. NOT a canonicity judgment; NOT an authority ranker; NOT a
memory substrate for agent reasoning. See docs/specs/QA_MEM_SCOPE.md.
"""
QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from tools.qa_kg.orbit import (
    Index, Tier, NODE_TYPE_RANK, compute_index, dr, char_ord_sum, tier_for_index,
    # deprecated aliases, one release cycle:
    Coord, compute_be, tier_for_coord,
)
from tools.qa_kg.kg import KG, Node, Edge, connect

__all__ = [
    "Index", "Tier", "NODE_TYPE_RANK",
    "compute_index", "dr", "char_ord_sum", "tier_for_index",
    "Coord", "compute_be", "tier_for_coord",
    "KG", "Node", "Edge", "connect",
]
