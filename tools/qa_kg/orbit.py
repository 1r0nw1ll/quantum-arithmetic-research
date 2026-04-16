"""QA-KG retrieval index — Candidate F [family 202].

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

THIS IS NOT A QA STATE CLASSIFIER. The fields `idx_b` and `idx_e` produced
here are a RETRIEVAL INDEX PARTITION, computed exactly as A-RAG computes
its sector prefilter per cert [202] (tools/qa_retrieval/schema.py::compute_be):

    idx_b = dr(char_ord_sum(content))    # content invariant (Aiq Bekar)
    idx_e = NODE_TYPE_RANK[node_type]    # KG analog of A-RAG ROLE_RANK

Candidate F is designed for uniform distribution across {1..9} (that is why
A-RAG uses it: it partitions 58k messages into roughly equal retrieval
sectors). The tier produced by `qa_orbit_rules.orbit_family` on (idx_b, idx_e)
is a PARTITION of the lattice, not a canonicity judgment. Under Candidate F
most KG nodes land in Cosmos by construction (72/81 of the lattice is Cosmos).

**Do not treat `idx_b/idx_e` as a QA state.** A QA state is (b,e) with
derived d=b+e, a=b+2e. Computing d=idx_b+idx_e produces meaningless values
because idx_b is a UTF-8 byte-sum digital root and idx_e is a node-type
ordinal. QA states for knowledge nodes live in `subject_b/subject_e` (see
schema.py) and are populated ONLY from cert metadata that declares a subject
state — never from Candidate F output.

Structural / semantic / causal relationships between nodes live in the
`edges` table (validates / derived-from / extends / contradicts / etc.),
not in idx co-location.

Tier = canonical `qa_orbit_rules.orbit_family(idx_b, idx_e)` (required by
axiom linter ORBIT-4/5). The wrapper below adds the Unassigned case for
nodes with no declared content (no observer projection yet).
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Canonical orbit rule — required import per axiom linter rule ORBIT-5.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from qa_orbit_rules import orbit_family as _canonical_orbit_family


MOD = 9

CAUSAL_EDGE_TYPES = frozenset({
    "validates", "extends", "derived-from", "maps-to", "instantiates",
})

# Phase 3 structural edge types — FK / lifecycle metadata, NOT derivations.
# The Theorem NT firewall does NOT apply: a `quoted-from` pointer is an FK
# (SourceClaim→SourceWork), and `supersedes` is lifecycle metadata
# (newer→older). Keeping them non-causal lets agent-authored nodes cite
# SourceWorks without needing to go through kg.promote() for the citation
# itself; any causal claim BUILT on that citation still has to promote.
STRUCTURAL_EDGE_TYPES = frozenset({
    "quoted-from", "supersedes", "promoted-from",
})

# Sanity: no overlap between causal and structural — regression guard for
# any future refactor that moves an edge type between categories.
assert CAUSAL_EDGE_TYPES.isdisjoint(STRUCTURAL_EDGE_TYPES), (
    "Phase 3 invariant violated: causal/structural edge-type sets overlap"
)


def dr(n: int) -> int:
    """Aiq Bekar digital root, A1-compliant [family 202]. Mirrors
    tools/qa_retrieval/schema.py::dr exactly."""
    if n <= 0:
        return 9
    return 1 + ((n - 1) % 9)


# A-RAG ROLE_RANK (user=1, assistant=2, tool=3, system=4, thought=5, note=6)
# extended to cover KG node types (7, 8, 9).
NODE_TYPE_RANK: dict[str, int] = {
    "Claim":    1,
    "Cert":     2,
    "Generator":3,
    "Axiom":    4,
    "Thought":  5,
    "Rule":     6,
    "Concept":  7,
    "Person":   8,
    "Work":     9,
}


class Tier(str, Enum):
    SINGULARITY = "singularity"
    SATELLITE   = "satellite"
    COSMOS      = "cosmos"
    UNASSIGNED  = "unassigned"  # no content observed yet


def qa_step(b: int, e: int, m: int = MOD) -> int:
    """A1-compliant step: result in {1..m}, never 0. Kept for QA state
    arithmetic (not applied to retrieval-index idx_b/idx_e)."""
    return ((b + e - 1) % m) + 1


@dataclass(frozen=True)
class Index:
    """Retrieval-index pair (idx_b, idx_e) — NOT a QA state.

    No `.d` or `.a` properties: those are QA state derivations (d=b+e,
    a=b+2e) which are meaningless on a retrieval-index hash × node-type
    rank. If code elsewhere needs QA state derivations, it must use the
    node's `subject_b/subject_e` (populated from cert metadata) — not
    this index.
    """
    idx_b: int
    idx_e: int

    def __post_init__(self) -> None:
        if not (1 <= self.idx_b <= MOD and 1 <= self.idx_e <= MOD):
            raise ValueError(
                f"Index({self.idx_b},{self.idx_e}) violates A1: must be in [1,{MOD}]"
            )

    def __str__(self) -> str:
        return f"idx({self.idx_b},{self.idx_e})"


def tier_for_index(idx_b: int | None, idx_e: int | None, m: int = MOD) -> Tier:
    """Thin typed wrapper around canonical qa_orbit_rules.orbit_family.

    Unassigned when either coordinate is None (pre-observation). Otherwise
    delegates verbatim to the canonical classifier.
    """
    if idx_b is None or idx_e is None:
        return Tier.UNASSIGNED
    return Tier(_canonical_orbit_family(idx_b, idx_e, m))


def char_ord_sum(text: str) -> int:
    return sum(ord(c) for c in text)


def compute_index(content: str, node_type: str) -> Index:
    """Candidate F [family 202] — retrieval-index computation.

    Returns an Index, not a QA state. See module docstring.
    """
    if not isinstance(content, str) or len(content) == 0:
        raise ValueError("compute_index requires non-empty content")
    if node_type not in NODE_TYPE_RANK:
        raise ValueError(
            f"node_type must be canonical, got {node_type!r}. "
            f"Valid: {sorted(NODE_TYPE_RANK)}"
        )
    return Index(dr(char_ord_sum(content)), NODE_TYPE_RANK[node_type])


def edge_allowed(
    src_tier: Tier,
    dst_tier: Tier,
    edge_type: str,
    via_cert: bool,
    *,
    src_authority: str | None = None,
) -> bool:
    """Theorem NT firewall guard — two layers.

    Layer 1 (Phase 0): Unassigned → Cosmos/Singularity causal edges need
    via_cert. Under Candidate F, Unassigned is effectively unoccupied.

    Layer 2 (Phase 2): authority=agent → causal edges ALWAYS blocked at
    policy level. The only bypass is a DB-backed promoted-from edge check
    in kg.upsert_edge(). This prevents callers from passing a via_cert
    string to circumvent the firewall — only kg.promote() can create the
    promoted-from edge that upsert_edge queries for.
    """
    if edge_type not in CAUSAL_EDGE_TYPES:
        return True
    if src_tier is Tier.UNASSIGNED and dst_tier in (Tier.COSMOS, Tier.SINGULARITY):
        return via_cert
    if src_authority == "agent":
        return False  # DB-backed promoted-from check required (kg.upsert_edge)
    return True
