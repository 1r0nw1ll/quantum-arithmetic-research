"""
qa_conjecture_core.py

Shared primitives for QA conjecture objects.

Provides:
- Conjecture dataclass with canonical serialization and hashing
- Loader/validator for conjecture JSON files
- Factory functions for each known conjecture type
- CLI: validate a conjecture file, or run self-test

Uses qa_cert_core for scalars, canonical JSON, and hashing.
Uses qa_meta_validator for structural + semantic validation.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

try:
    # When run as module: python -m qa_alphageometry_ptolemy.qa_conjecture_core
    from .qa_cert_core import (
        canonical_json, certificate_hash, full_hash,
        cert_id, utc_now_iso, ValidationResult,
    )
    from .qa_meta_validator import KNOWN_CONJECTURE_TYPES
except ImportError:
    # When run directly: python qa_conjecture_core.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qa_cert_core import (
        canonical_json, certificate_hash, full_hash,
        cert_id, utc_now_iso, ValidationResult,
    )
    from qa_meta_validator import KNOWN_CONJECTURE_TYPES

VALID_STATUSES = {"open", "supported", "refuted"}


# ============================================================================
# CONJECTURE DATACLASS
# ============================================================================

@dataclass
class Conjecture:
    """
    A QA conjecture: a falsifiable claim about the certificate algebra.

    Each conjecture has:
    - A structured claim (statement, scope, success/falsification conditions)
    - A deterministic validator contract (steps with ops)
    - A failure taxonomy (structured failure types with invariant_diff schemas)
    - A status (open, supported, refuted)

    Serialized to canonical JSON for deterministic hashing.
    """
    conjecture_type: str
    title: str
    claim: Dict[str, str]
    validator_contract: Dict[str, Any]
    failure_taxonomy: List[Dict[str, Any]]
    status: str = "open"

    # Metadata
    certificate_id: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.certificate_id is None:
            self.certificate_id = cert_id(f"QA_CONJ__{self.conjecture_type}")
        if self.timestamp is None:
            self.timestamp = utc_now_iso()
        if self.conjecture_type not in KNOWN_CONJECTURE_TYPES:
            raise ValueError(f"Unknown conjecture_type: {self.conjecture_type}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {self.status}")
        if not self.validator_contract.get("deterministic"):
            raise ValueError("validator_contract.deterministic must be true")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "certificate_type": "QA_CONJECTURE",
            "timestamp": self.timestamp,
            "conjecture_type": self.conjecture_type,
            "title": self.title,
            "claim": self.claim,
            "validator_contract": self.validator_contract,
            "failure_taxonomy": self.failure_taxonomy,
            "status": self.status,
        }

    def to_json(self, indent: int = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)

    def content_hash(self) -> str:
        return certificate_hash(self.to_dict())

    def content_hash_full(self) -> str:
        return full_hash(self.to_dict())


# ============================================================================
# LOADER
# ============================================================================

def load_conjecture(path: str) -> Conjecture:
    """Load a conjecture from a JSON file."""
    with open(path) as f:
        d = json.load(f)
    return Conjecture(
        conjecture_type=d["conjecture_type"],
        title=d["title"],
        claim=d["claim"],
        validator_contract=d["validator_contract"],
        failure_taxonomy=d["failure_taxonomy"],
        status=d.get("status", "open"),
        certificate_id=d.get("certificate_id"),
        timestamp=d.get("timestamp"),
    )


def validate_conjecture_file(path: str) -> ValidationResult:
    """Validate a conjecture JSON file through the meta-validator."""
    from qa_meta_validator import validate_json
    with open(path) as f:
        result = validate_json(f.read())
    v = ValidationResult()
    if not result.is_valid:
        for issue in (result.issues or []):
            v.fail(issue)
    return v


# ============================================================================
# CANONICAL LEDGER PATH
# ============================================================================

LEDGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "qa_ledger", "conjectures")


def ledger_path(filename: str) -> str:
    """Return full path to a conjecture in the canonical ledger."""
    return os.path.join(LEDGER_DIR, filename)


def list_ledger() -> List[str]:
    """List all conjecture JSON files in the canonical ledger."""
    if not os.path.isdir(LEDGER_DIR):
        return []
    return sorted(f for f in os.listdir(LEDGER_DIR) if f.endswith(".json"))


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_substrate_invariance_conjecture() -> Conjecture:
    """CONJ_SUBSTRATE_INVARIANCE: K is approximately invariant under substrate substitution."""
    return Conjecture(
        conjecture_type="CONJ_SUBSTRATE_INVARIANCE",
        title="K is approximately invariant under substrate substitution for isomorphic problem spaces",
        claim={
            "statement": (
                "For matched problem spaces P=<S,O,C,E,H>, K=log10(tau_blind/tau_agent) "
                "is invariant across substrates up to declared tolerance."
            ),
            "scope": "substrate-neutral cognition (cells, glia, silicon, collectives)",
            "success_condition": "|K_X - K_Y| <= epsilon (or within declared constant-factor window)",
            "falsification_condition": (
                "Reproducible K divergence beyond tolerance under isomorphic P and matched budgets"
            ),
        },
        validator_contract={
            "deterministic": True,
            "inputs": ["P_X", "P_Y", "task_suite", "budgets", "seeds", "epsilon"],
            "outputs": ["K_X", "K_Y", "pass_fail", "fail_type", "invariant_diff"],
            "steps": [
                {"op": "DECLARE_ISOMORPHIC_P",
                 "notes": "Verify operator + constraint name isomorphism"},
                {"op": "ESTIMATE_TAU_BLIND",
                 "notes": "Fixed baseline sampler (seeded)"},
                {"op": "MEASURE_TAU_AGENT",
                 "notes": "Fixed policy class + evaluation"},
                {"op": "COMPUTE_K",
                 "notes": "K = log10(tau_blind/tau_agent)"},
                {"op": "COMPARE_K",
                 "notes": "Check tolerance window"},
            ],
            "replay": {"canonical_json": True, "hash": "sha256"},
        },
        failure_taxonomy=[
            {"fail_type": "P_MISMATCH",
             "invariant_diff_schema": {"missing_ops": "list[str]", "missing_constraints": "list[str]"}},
            {"fail_type": "BASELINE_NONDETERMINISTIC",
             "invariant_diff_schema": {"seed": "str", "delta": "float"}},
            {"fail_type": "K_DIVERGENCE",
             "invariant_diff_schema": {"K_X": "float", "K_Y": "float", "epsilon": "float"}},
        ],
    )


def create_horizon_hierarchy_conjecture() -> Conjecture:
    """CONJ_HORIZON_HIERARCHY: Increasing H does not decrease K."""
    return Conjecture(
        conjecture_type="CONJ_HORIZON_HIERARCHY",
        title="Increasing horizon H (holding S,O,C,E fixed) does not decrease K",
        claim={
            "statement": (
                "With S,O,C,E fixed, K(H2) >= K(H1) for H2>H1; "
                "strict increase when a multi-step witness exists."
            ),
            "scope": "morphogenesis + tool-using agents",
            "success_condition": "K(H2) >= K(H1)",
            "falsification_condition": (
                "Reproducible K monotonicity violation under matched regimes"
            ),
        },
        validator_contract={
            "deterministic": True,
            "inputs": ["P", "H1", "H2", "budgets", "seeds"],
            "outputs": ["K_H1", "K_H2", "pass_fail", "fail_type", "invariant_diff"],
            "steps": [
                {"op": "FREEZE_SOCE", "notes": "Fix S,O,C,E"},
                {"op": "RUN_AGENT", "notes": "Execute under horizon H1"},
                {"op": "RUN_AGENT", "notes": "Execute under horizon H2"},
                {"op": "COMPUTE_K", "notes": "Compute K for each horizon"},
                {"op": "CHECK_MONOTONICITY", "notes": "Assert K(H2) >= K(H1)"},
            ],
            "replay": {"canonical_json": True, "hash": "sha256"},
        },
        failure_taxonomy=[
            {"fail_type": "H_NOT_APPLIED",
             "invariant_diff_schema": {"H_declared": "int", "H_effective": "int"}},
            {"fail_type": "POLICY_CLASS_CHANGED",
             "invariant_diff_schema": {"policy_before": "str", "policy_after": "str"}},
            {"fail_type": "K_MONOTONICITY_VIOLATION",
             "invariant_diff_schema": {"K_H1": "float", "K_H2": "float"}},
        ],
    )


def create_goal_collapse_equivalence_conjecture() -> Conjecture:
    """CONJ_GOAL_COLLAPSE_EQUIVALENCE: Goal decoupling and diversity collapse are the same obstruction."""
    return Conjecture(
        conjecture_type="CONJ_GOAL_COLLAPSE_EQUIVALENCE",
        title="Goal decoupling across scales is the same obstruction family as diversity/mode collapse",
        claim={
            "statement": (
                "There exists a witness isomorphism between (E_local != E_global) "
                "goal-decoupling failures and diversity-collapse failures, i.e., "
                "both are evaluation-misalignment obstructions."
            ),
            "scope": "oncology, morphogenesis, RL/search",
            "success_condition": "Construct isomorphism map between witness structures",
            "falsification_condition": (
                "No isomorphism exists under any consistent declaration of P"
            ),
        },
        validator_contract={
            "deterministic": True,
            "inputs": ["system_A_goal_decoupling", "system_B_collapse", "P_A", "P_B"],
            "outputs": ["iso_map", "pass_fail", "fail_type", "invariant_diff"],
            "steps": [
                {"op": "DECLARE_TWO_LEVEL_E",
                 "notes": "Define E_local and E_global"},
                {"op": "EXTRACT_MISALIGNMENT_WITNESS",
                 "notes": "Show optimizing E_local degrades E_global"},
                {"op": "EXTRACT_COLLAPSE_WITNESS",
                 "notes": "Obtain collapse signature under fixed constraints"},
                {"op": "CONSTRUCT_ISOMORPHISM",
                 "notes": "Map witness structures"},
                {"op": "VERIFY_BARRIER_EQUIVALENCE",
                 "notes": "Same obstruction family"},
            ],
            "replay": {"canonical_json": True, "hash": "sha256"},
        },
        failure_taxonomy=[
            {"fail_type": "NO_MISALIGNMENT_WITNESS",
             "invariant_diff_schema": {"reason": "str"}},
            {"fail_type": "NONISOMORPHIC_WITNESS",
             "invariant_diff_schema": {"mismatch": "str"}},
            {"fail_type": "BARRIER_TYPE_MISMATCH",
             "invariant_diff_schema": {"barrier_A": "str", "barrier_B": "str"}},
        ],
    )


# ============================================================================
# CLI + SELF-TEST
# ============================================================================

if __name__ == "__main__":
    # CLI: validate a specific file
    if len(sys.argv) > 1 and sys.argv[1] != "--self-test":
        path = sys.argv[1]
        conj = load_conjecture(path)
        print(f"Loaded: {conj.conjecture_type}")
        print(f"Title:  {conj.title}")
        print(f"Status: {conj.status}")
        print(f"Hash:   {conj.content_hash()}")
        vr = validate_conjecture_file(path)
        print(f"Valid:  {vr.is_valid}")
        if not vr.is_valid:
            for issue in vr.issues:
                print(f"  Issue: {issue}")
        sys.exit(0 if vr.is_valid else 1)

    # Self-test
    print("=== QA CONJECTURE CORE SELF-TEST ===\n")

    # Test 1: Factory functions produce valid conjectures
    factories = [
        ("SUBSTRATE_INVARIANCE", create_substrate_invariance_conjecture),
        ("HORIZON_HIERARCHY", create_horizon_hierarchy_conjecture),
        ("GOAL_COLLAPSE_EQUIVALENCE", create_goal_collapse_equivalence_conjecture),
    ]

    for label, factory in factories:
        conj = factory()
        d = conj.to_dict()
        assert d["certificate_type"] == "QA_CONJECTURE"
        assert d["conjecture_type"] in KNOWN_CONJECTURE_TYPES
        assert d["status"] == "open"
        assert conj.content_hash()  # non-empty
        print(f"[FACTORY] {label}: hash={conj.content_hash()} id={conj.certificate_id}")

    # Test 2: Load ledger files and verify hashes match content
    print()
    ledger_files = list_ledger()
    print(f"Ledger contains {len(ledger_files)} conjecture(s):")
    for fname in ledger_files:
        fpath = ledger_path(fname)
        conj = load_conjecture(fpath)
        vr = validate_conjecture_file(fpath)
        print(f"  [{conj.status.upper():>9s}] {fname}: "
              f"valid={vr.is_valid} hash={conj.content_hash()}")
        assert vr.is_valid, f"Ledger file {fname} failed validation: {vr.issues}"

    # Test 3: Canonical JSON is deterministic
    c1 = create_substrate_invariance_conjecture()
    c2 = create_substrate_invariance_conjecture()
    # IDs and timestamps will differ, but structure is the same
    # Set same ID/timestamp to test determinism
    c2.certificate_id = c1.certificate_id
    c2.timestamp = c1.timestamp
    assert c1.to_json() == c2.to_json(), "Canonical JSON not deterministic"
    assert c1.content_hash() == c2.content_hash(), "Hash not deterministic"
    print("\n[DETERMINISM] Canonical JSON + hash: deterministic")

    # Test 4: Bad conjecture type raises
    try:
        Conjecture(
            conjecture_type="CONJ_NONEXISTENT",
            title="bad", claim={"statement": "bad"},
            validator_contract={"deterministic": True, "steps": [{"op": "X"}]},
            failure_taxonomy=[{"fail_type": "X", "invariant_diff_schema": {}}],
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[GUARD] Unknown conjecture_type correctly rejected")

    # Test 5: Non-deterministic contract raises
    try:
        Conjecture(
            conjecture_type="CONJ_SUBSTRATE_INVARIANCE",
            title="bad", claim={"statement": "bad"},
            validator_contract={"deterministic": False, "steps": [{"op": "X"}]},
            failure_taxonomy=[{"fail_type": "X", "invariant_diff_schema": {}}],
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[GUARD] Non-deterministic contract correctly rejected")

    print("\nAll qa_conjecture_core self-tests PASSED")
