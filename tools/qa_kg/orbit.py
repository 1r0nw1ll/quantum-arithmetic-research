"""Orbit math, tier classification, (b,e) coord assignment.

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Singularity (9,9) — axioms + Theorem NT (immutable).
Cosmos 24-cycle — canonical (passing certs, MEMORY Hard Rules, frozen papers).
Satellite 8-cycle — working (OB thoughts, drafts, uncertified claims).
Unassigned — archive (A-RAG, old exports); cannot form causal edges into Cosmos
without passing through a Cert node. Enforced in kg.upsert_edge.

QA axioms respected:
- A1: states in {1..m}; qa_step = ((b+e-1) % m) + 1
- A2: d = b+e, a = b+2e (derived, raw for elements; mod only via T-operator)
- S1: no **2
- S2: int state only
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import hashlib
from dataclasses import dataclass
from enum import Enum


MOD = 9  # QA mod-9 for the memory graph; mod-24 reserved for applied work


class Tier(str, Enum):
    SINGULARITY = "singularity"
    COSMOS = "cosmos"
    SATELLITE = "satellite"
    UNASSIGNED = "unassigned"


def qa_step(b: int, e: int, m: int = MOD) -> int:
    """A1-compliant step: result in {1..m}, never 0."""
    return ((b + e - 1) % m) + 1


@dataclass(frozen=True)
class Coord:
    b: int
    e: int

    def __post_init__(self) -> None:
        if not (1 <= self.b <= MOD and 1 <= self.e <= MOD):
            raise ValueError(f"Coord ({self.b},{self.e}) violates A1: must be in [1,{MOD}]")

    @property
    def d(self) -> int:
        return self.b + self.e  # A2 raw

    @property
    def a(self) -> int:
        return self.b + 2 * self.e  # A2 raw

    @property
    def is_singularity(self) -> bool:
        return self.b == 9 and self.e == 9

    def __str__(self) -> str:
        return f"({self.b},{self.e})"


def coord_for(key: str) -> Coord:
    """Deterministic (b,e) coord from a stable key.

    Hash key → two independent digits in [1..9]. Reserves (9,9) for Singularity:
    any key that would land there gets deflected to (9,8).
    """
    h = hashlib.sha256(key.encode("utf-8")).digest()
    b = (h[0] % MOD) + 1
    e = (h[1] % MOD) + 1
    if b == 9 and e == 9:
        e = 8
    return Coord(b, e)


def tier_for(
    *,
    is_axiom: bool = False,
    vetted_by_cert: bool = False,
    is_memory_hard_rule: bool = False,
    is_frozen_paper: bool = False,
    is_archive: bool = False,
) -> Tier:
    """Classify a node into an orbit tier by its provenance flags."""
    if is_axiom:
        return Tier.SINGULARITY
    if vetted_by_cert or is_memory_hard_rule or is_frozen_paper:
        return Tier.COSMOS
    if is_archive:
        return Tier.UNASSIGNED
    return Tier.SATELLITE


def edge_allowed(src_tier: Tier, dst_tier: Tier, edge_type: str, via_cert: bool) -> bool:
    """Theorem NT firewall — structural, not policy.

    Archive (UNASSIGNED) cannot form causal edges into COSMOS or SINGULARITY
    unless the edge is mediated by a Cert node (via_cert=True).
    """
    causal = {"validates", "extends", "derived-from", "maps-to", "instantiates"}
    if edge_type not in causal:
        return True
    if src_tier is Tier.UNASSIGNED and dst_tier in (Tier.COSMOS, Tier.SINGULARITY):
        return via_cert
    return True
