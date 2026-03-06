"""Canonical QA algebra bridge semantics anchor.

This module is the single source of truth for the bridge semantics payload and hash.
Downstream cert families should import BRIDGE_SEMANTICS_SHA256 from here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


BRIDGE_SEMANTICS_ID = "QA_ALGEBRA_SEMANTICS.v1"
BRIDGE_GENERATOR_SEMANTICS_REF = "QA_GENERATORS_SIGMA_SHEAR.v1"
BRIDGE_WORD_APPLICATION_ORDER = "left_to_right"

BRIDGE_GENERATOR_DEFS: Dict[str, str] = {
    "sigma": "(b,e)->(b,e+b)",
    "mu": "(b,e)->(e,b)",
    "R": "R:=mu sigma mu; (b,e)->(b+e,e)",
    "lambda_k": "(b,e,k)->(k*b,k*e), k>=1",
    "nu": "(b,e)->(b/2,e/2) iff b,e both even; else ODD_BLOCK",
}

BRIDGE_CORE_PROPERTIES: List[str] = [
    "gcd invariant under sigma/mu/R",
    "state_to_word gives unique LR normal form on coprime states",
    "state_to_word_with_scale gives (word,scale,normalized)",
    "nu contraction depth on gcd is v2(g)",
]


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def build_bridge_semantics_payload() -> Dict[str, Any]:
    return {
        "semantics_id": BRIDGE_SEMANTICS_ID,
        "generator_semantics_ref": BRIDGE_GENERATOR_SEMANTICS_REF,
        "word_application_order": BRIDGE_WORD_APPLICATION_ORDER,
        "generator_defs": BRIDGE_GENERATOR_DEFS,
        "core_properties": BRIDGE_CORE_PROPERTIES,
    }


def compute_bridge_semantics_sha256() -> str:
    return hashlib.sha256(_canonical_json(build_bridge_semantics_payload()).encode("utf-8")).hexdigest()


BRIDGE_SEMANTICS_SHA256 = compute_bridge_semantics_sha256()

