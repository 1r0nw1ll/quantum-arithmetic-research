"""Shared failure-type projection maps into the [76] failure algebra carrier."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from qa_failure_algebra_structure_cert_v1.failure_algebra_anchor import FAILURE_TYPES


FAILURE_PROJECTION_MAP_VERSION = "QA_FAILURE_PROJECTION_MAP.v1"
PROJECTION_MAP_NAME_STRUCTURAL = "structural_schema"
PROJECTION_MAP_NAME_INTEGRITY = "integrity_claim"

# Conservative projection from family-local fail types into [76] carrier tags.
# Unmapped fail types must be surfaced explicitly by consumers.
FAILURE_PROJECTION_MAP_V1_STRUCTURAL: Dict[str, str] = {
    "BRIDGE_SEMANTICS_HASH_MISMATCH": "OUT_OF_DOMAIN",
    "SCHEMA_TYPE_MISMATCH": "OUT_OF_DOMAIN",
    "SCHEMA_VALUE_INVALID": "OUT_OF_DOMAIN",
}

# Claim-integrity focused projection map.
FAILURE_PROJECTION_MAP_V1_INTEGRITY: Dict[str, str] = {
    "INVARIANT_DIFF_MAP_CLAIM_MISMATCH": "INVARIANT_VIOLATION",
    "FAILURE_ALGEBRA_ANCHOR_REF_MISMATCH": "INVARIANT_VIOLATION",
    "FAILURE_ALGEBRA_ANCHOR_ROLLUP_MISMATCH": "INVARIANT_VIOLATION",
}


def choose_projection_map_with_reason(
    raw_fail_types: Iterable[str],
) -> Tuple[str, Dict[str, str], str, List[str]]:
    """Select projection map by failure intent with deterministic audit metadata."""
    raw = set(raw_fail_types)
    hits = sorted(raw.intersection(FAILURE_PROJECTION_MAP_V1_INTEGRITY.keys()))
    if hits:
        reason = "integrity_claim because hit {" + ",".join(hits) + "}"
        return PROJECTION_MAP_NAME_INTEGRITY, FAILURE_PROJECTION_MAP_V1_INTEGRITY, reason, hits
    reason = "default structural_schema"
    return PROJECTION_MAP_NAME_STRUCTURAL, FAILURE_PROJECTION_MAP_V1_STRUCTURAL, reason, []


def choose_projection_map(raw_fail_types: Iterable[str]) -> Tuple[str, Dict[str, str]]:
    """Backward-compatible selector helper."""
    name, mapping, _reason, _hits = choose_projection_map_with_reason(raw_fail_types)
    return name, mapping

# Import-time closure guard: every projection target must be in the [76] carrier.
_all_targets = set(FAILURE_PROJECTION_MAP_V1_STRUCTURAL.values()) | set(FAILURE_PROJECTION_MAP_V1_INTEGRITY.values())
_invalid_targets = sorted(_all_targets - set(FAILURE_TYPES))
if _invalid_targets:
    raise ValueError(
        "Failure projection map has non-carrier targets: "
        + ", ".join(_invalid_targets)
    )
