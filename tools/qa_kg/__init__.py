"""QA-KG — QA-native knowledge graph for the project's memory system.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Orbits are tiers, claims are predicates, Theorem NT is enforced structurally.
See memory/project_qa_mem_architecture.md for design.
"""
QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

from tools.qa_kg.orbit import Tier, coord_for, tier_for
from tools.qa_kg.kg import KG, connect

__all__ = ["Tier", "coord_for", "tier_for", "KG", "connect"]
