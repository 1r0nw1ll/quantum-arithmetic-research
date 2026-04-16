"""QA-KG — indexed, firewalled view over repo artifacts.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Schema v3: epistemic fields (authority/epistemic_status/method/source_locator/
lifecycle_state) + alias removal (Coord/compute_be/tier_for_coord dropped per
[225] v3 / schema v3 pin). See docs/specs/QA_MEM_SCOPE.md.
"""
QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from tools.qa_kg.orbit import (
    Index, Tier, NODE_TYPE_RANK, compute_index, dr, char_ord_sum, tier_for_index,
    CAUSAL_EDGE_TYPES, STRUCTURAL_EDGE_TYPES,
)
from tools.qa_kg.kg import (
    KG, Node, Edge, FirewallViolation, connect,
    PROMOTED_FROM_EDGE, LEDGER_STALENESS_DAYS,
)

__all__ = [
    "Index", "Tier", "NODE_TYPE_RANK",
    "compute_index", "dr", "char_ord_sum", "tier_for_index",
    "CAUSAL_EDGE_TYPES", "STRUCTURAL_EDGE_TYPES",
    "KG", "Node", "Edge", "FirewallViolation", "connect",
    "PROMOTED_FROM_EDGE", "LEDGER_STALENESS_DAYS",
]
