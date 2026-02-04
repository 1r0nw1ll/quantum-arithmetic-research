"""
qa_guardrail.py

QA Guardrail MVP - Agent security gate using QA Operating System principles.

Provides:
- guard(planned_move, ctx) -> {ok: bool, result: "ALLOW"|"DENY", fail_record?}
- Policy enforcement using QA failure algebra
- Instruction/content separation verification
- Generator authorization checking

Usage:
    python qa_guardrail.py                    # Run self-tests
    python qa_guardrail.py --validate         # Output as JSON
    python qa_guardrail.py --fixtures         # Validate golden fixtures

Based on QA_OS_SPEC.v1 security layer.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_cert_core import (
    canonical_json_compact,
    sha256_canonical,
    certificate_hash,
    ValidationResult,
    utc_now_iso,
)


# ============================================================================
# FAIL TYPES (from QA_OS_SPEC.v1 security.failure_algebra.core_fail_types)
# ============================================================================

FAIL_TYPES = {
    "OUT_OF_BOUNDS",
    "PARITY",
    "FIXED_Q",
    "INVARIANT_VIOLATION",
    "UNAUTHORIZED_GENERATOR",
    "DOMAIN_MISMATCH",
    "PROMOTION_FORBIDDEN",
    "SOURCE_NUMERIC_DRIFT",
    "SYMMETRY_DEFECT",
}

# Guardrail-specific fail types (extension)
GUARDRAIL_FAIL_TYPES = FAIL_TYPES | {
    "MISSING_CAPABILITY",
    "INSTRUCTION_CONTENT_BOUNDARY_VIOLATION",
    "POLICY_CONSTRAINT_VIOLATION",
    "TRACE_INVARIANT_BROKEN",
}


# ============================================================================
# AUTHORIZED GENERATORS (from QA_OS_SPEC.v1 kernel.generators.declared)
# ============================================================================

AUTHORIZED_GENERATORS: Set[str] = {"sigma", "mu", "lambda", "nu"}


# ============================================================================
# GUARDRAIL CONTEXT
# ============================================================================

class GuardrailContext:
    """
    Context for guardrail evaluation.

    Fields:
        active_generators: Set of generators currently enabled
        instruction_content_cert: Optional cert proving instruction/content separation
        trace_tail: Recent trace entries for continuity check
        policy: Policy constraints (allow/deny lists)
        capabilities: Granted capability tokens
    """

    def __init__(
        self,
        active_generators: Optional[Set[str]] = None,
        instruction_content_cert: Optional[Dict[str, Any]] = None,
        trace_tail: Optional[List[Dict[str, Any]]] = None,
        policy: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Set[str]] = None,
    ):
        self.active_generators = active_generators or AUTHORIZED_GENERATORS.copy()
        self.instruction_content_cert = instruction_content_cert
        self.trace_tail = trace_tail or []
        self.policy = policy or {}
        self.capabilities = capabilities or set()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_generators": sorted(self.active_generators),
            "instruction_content_cert": self.instruction_content_cert,
            "trace_tail": self.trace_tail,
            "policy": self.policy,
            "capabilities": sorted(self.capabilities),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GuardrailContext":
        return cls(
            active_generators=set(d.get("active_generators", [])),
            instruction_content_cert=d.get("instruction_content_cert"),
            trace_tail=d.get("trace_tail", []),
            policy=d.get("policy", {}),
            capabilities=set(d.get("capabilities", [])),
        )


# ============================================================================
# FAIL RECORD
# ============================================================================

def make_fail_record(
    move: str,
    fail_type: str,
    invariant_diff: Optional[Dict[str, Any]] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a FAIL_RECORD.v1 compliant failure record.

    Required fields: move, fail_type, invariant_diff
    """
    if fail_type not in GUARDRAIL_FAIL_TYPES:
        raise ValueError(f"Unknown fail_type: {fail_type}")

    return {
        "move": move,
        "fail_type": fail_type,
        "invariant_diff": invariant_diff or {},
        "detail": detail,
        "timestamp_utc": utc_now_iso(),
    }


# ============================================================================
# GUARDRAIL RESULT
# ============================================================================

class GuardrailResult:
    """
    Result of guardrail evaluation.

    Either ALLOW (ok=True) or DENY (ok=False with fail_record).
    """

    def __init__(
        self,
        ok: bool,
        result: str,
        fail_record: Optional[Dict[str, Any]] = None,
        checks: Optional[List[str]] = None,
    ):
        self.ok = ok
        self.result = result
        self.fail_record = fail_record
        self.checks = checks or []

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ok": self.ok,
            "result": self.result,
            "checks": self.checks,
        }
        if self.fail_record:
            d["fail_record"] = self.fail_record
        return d

    @staticmethod
    def allow(checks: Optional[List[str]] = None) -> "GuardrailResult":
        return GuardrailResult(ok=True, result="ALLOW", checks=checks)

    @staticmethod
    def deny(fail_record: Dict[str, Any], checks: Optional[List[str]] = None) -> "GuardrailResult":
        return GuardrailResult(ok=False, result="DENY", fail_record=fail_record, checks=checks)


# ============================================================================
# CORE GUARD FUNCTION
# ============================================================================

def guard(planned_move: str, ctx: GuardrailContext) -> GuardrailResult:
    """
    Evaluate whether a planned move should be allowed.

    Checks:
    1. Generator authorization (is the move an authorized generator?)
    2. Policy constraints (allow/deny lists)
    3. Instruction/content separation (if cert provided)
    4. Capability requirements (if policy specifies)

    Returns:
        GuardrailResult with ok=True (ALLOW) or ok=False (DENY + fail_record)
    """
    checks: List[str] = []

    # --- Check 1: Generator authorization ---
    # Extract generator name from move (e.g., "sigma(3)" -> "sigma")
    generator_name = planned_move.split("(")[0].strip().lower()

    if generator_name not in ctx.active_generators:
        # Check if it's in the full set but not active
        if generator_name in AUTHORIZED_GENERATORS:
            return GuardrailResult.deny(
                make_fail_record(
                    move=planned_move,
                    fail_type="UNAUTHORIZED_GENERATOR",
                    invariant_diff={"reason": "generator_not_active"},
                    detail=f"Generator '{generator_name}' is not active in current context",
                ),
                checks=checks + [f"generator_auth: FAIL ({generator_name} not active)"],
            )
        else:
            return GuardrailResult.deny(
                make_fail_record(
                    move=planned_move,
                    fail_type="UNAUTHORIZED_GENERATOR",
                    invariant_diff={"reason": "unknown_generator"},
                    detail=f"Generator '{generator_name}' is not in authorized set",
                ),
                checks=checks + [f"generator_auth: FAIL ({generator_name} unknown)"],
            )

    checks.append(f"generator_auth: OK ({generator_name})")

    # --- Check 2: Policy deny list ---
    deny_list = ctx.policy.get("deny", [])
    if planned_move in deny_list or generator_name in deny_list:
        return GuardrailResult.deny(
            make_fail_record(
                move=planned_move,
                fail_type="POLICY_CONSTRAINT_VIOLATION",
                invariant_diff={"policy_violated": "deny_list"},
                detail=f"Move '{planned_move}' is in policy deny list",
            ),
            checks=checks + ["policy_deny: FAIL (in deny list)"],
        )

    checks.append("policy_deny: OK")

    # --- Check 3: Policy allow list (if specified, move must be in it) ---
    allow_list = ctx.policy.get("allow")
    if allow_list is not None:
        if planned_move not in allow_list and generator_name not in allow_list:
            return GuardrailResult.deny(
                make_fail_record(
                    move=planned_move,
                    fail_type="POLICY_CONSTRAINT_VIOLATION",
                    invariant_diff={"policy_violated": "allow_list"},
                    detail=f"Move '{planned_move}' is not in policy allow list",
                ),
                checks=checks + ["policy_allow: FAIL (not in allow list)"],
            )
        checks.append("policy_allow: OK")

    # --- Check 4: Required capability ---
    required_capability = ctx.policy.get("required_capability")
    if required_capability and required_capability not in ctx.capabilities:
        return GuardrailResult.deny(
            make_fail_record(
                move=planned_move,
                fail_type="MISSING_CAPABILITY",
                invariant_diff={"required": required_capability, "granted": sorted(ctx.capabilities)},
                detail=f"Missing required capability: {required_capability}",
            ),
            checks=checks + [f"capability: FAIL (missing {required_capability})"],
        )

    if required_capability:
        checks.append(f"capability: OK ({required_capability})")

    # --- Check 5: Instruction/content separation ---
    if ctx.instruction_content_cert:
        cert = ctx.instruction_content_cert
        # Verify the cert has the expected structure
        if cert.get("schema_id") != "QA_INSTRUCTION_CONTENT_SEPARATION_CERT.v1":
            return GuardrailResult.deny(
                make_fail_record(
                    move=planned_move,
                    fail_type="INSTRUCTION_CONTENT_BOUNDARY_VIOLATION",
                    invariant_diff={"reason": "invalid_cert_schema"},
                    detail="instruction_content_cert has wrong schema_id",
                ),
                checks=checks + ["ic_separation: FAIL (invalid cert schema)"],
            )

        # Check that the planned move is in the instruction domain, not content domain
        content_domain = set(cert.get("content_domain", []))
        if generator_name in content_domain:
            return GuardrailResult.deny(
                make_fail_record(
                    move=planned_move,
                    fail_type="PROMOTION_FORBIDDEN",
                    invariant_diff={"reason": "content_promoted_to_instruction"},
                    detail=f"Generator '{generator_name}' is in content domain, cannot become instruction",
                ),
                checks=checks + ["ic_separation: FAIL (content promoted to instruction)"],
            )

        checks.append("ic_separation: OK")

    # --- Check 6: Trace continuity (if trace_tail provided) ---
    if ctx.trace_tail:
        last_entry = ctx.trace_tail[-1]
        # Check that last move didn't fail (or we're handling failure correctly)
        if last_entry.get("fail_type") and not ctx.policy.get("allow_after_failure"):
            return GuardrailResult.deny(
                make_fail_record(
                    move=planned_move,
                    fail_type="TRACE_INVARIANT_BROKEN",
                    invariant_diff={"previous_failure": last_entry.get("fail_type")},
                    detail="Previous move failed, cannot continue without explicit handling",
                ),
                checks=checks + ["trace_continuity: FAIL (previous failure unhandled)"],
            )

        checks.append("trace_continuity: OK")

    # All checks passed
    return GuardrailResult.allow(checks=checks)


# ============================================================================
# BATCH EVALUATION
# ============================================================================

def guard_batch(
    moves: List[str],
    ctx: GuardrailContext,
) -> List[GuardrailResult]:
    """Evaluate a batch of moves."""
    return [guard(move, ctx) for move in moves]


# ============================================================================
# FIXTURE VALIDATION
# ============================================================================

def validate_fixtures(fixtures_dir: str) -> Dict[str, Any]:
    """
    Validate golden fixtures in the given directory.

    Each fixture is a JSON file with:
    - planned_move: str
    - context: GuardrailContext dict
    - expected_result: "ALLOW" or "DENY"
    - expected_fail_type: (optional, for DENY cases)

    Returns validation results.
    """
    results = {"ok": True, "passed": 0, "failed": 0, "errors": []}

    for filename in sorted(os.listdir(fixtures_dir)):
        if not filename.endswith(".json"):
            continue

        fixture_path = os.path.join(fixtures_dir, filename)
        try:
            with open(fixture_path) as f:
                fixture = json.load(f)

            planned_move = fixture["planned_move"]
            ctx = GuardrailContext.from_dict(fixture.get("context", {}))
            expected_result = fixture["expected_result"]
            expected_fail_type = fixture.get("expected_fail_type")

            result = guard(planned_move, ctx)

            # Check result matches expectation
            if result.result != expected_result:
                results["ok"] = False
                results["failed"] += 1
                results["errors"].append(
                    f"{filename}: expected {expected_result}, got {result.result}"
                )
                continue

            # For DENY cases, check fail_type
            if expected_result == "DENY" and expected_fail_type:
                actual_fail_type = result.fail_record.get("fail_type") if result.fail_record else None
                if actual_fail_type != expected_fail_type:
                    results["ok"] = False
                    results["failed"] += 1
                    results["errors"].append(
                        f"{filename}: expected fail_type={expected_fail_type}, got {actual_fail_type}"
                    )
                    continue

            results["passed"] += 1

        except Exception as e:
            results["ok"] = False
            results["failed"] += 1
            results["errors"].append(f"{filename}: {e}")

    return results


# ============================================================================
# SELF-TESTS
# ============================================================================

def run_self_tests() -> Dict[str, Any]:
    """Run internal self-tests."""
    results = {"ok": True, "tests": [], "errors": []}

    def test(name: str, condition: bool, detail: str = ""):
        if condition:
            results["tests"].append(f"{name}: PASS")
        else:
            results["ok"] = False
            results["tests"].append(f"{name}: FAIL")
            results["errors"].append(f"{name}: {detail}")

    # Test 1: Basic ALLOW for authorized generator
    ctx = GuardrailContext()
    r = guard("sigma(1)", ctx)
    test("T1_sigma_allowed", r.ok and r.result == "ALLOW", f"got {r.result}")

    # Test 2: DENY for unknown generator
    r = guard("unknown_gen()", ctx)
    test("T2_unknown_denied", not r.ok and r.result == "DENY", f"got {r.result}")
    test("T2_fail_type", r.fail_record and r.fail_record["fail_type"] == "UNAUTHORIZED_GENERATOR",
         f"got {r.fail_record}")

    # Test 3: DENY for inactive generator
    ctx2 = GuardrailContext(active_generators={"sigma", "mu"})
    r = guard("lambda(2)", ctx2)
    test("T3_inactive_denied", not r.ok, f"got {r.result}")

    # Test 4: Policy deny list
    ctx3 = GuardrailContext(policy={"deny": ["sigma(bad)"]})
    r = guard("sigma(bad)", ctx3)
    test("T4_deny_list", not r.ok and r.fail_record["fail_type"] == "POLICY_CONSTRAINT_VIOLATION",
         f"got {r.result}")

    # Test 5: Policy allow list
    ctx4 = GuardrailContext(policy={"allow": ["sigma", "mu"]})
    r = guard("sigma(1)", ctx4)
    test("T5_allow_list_pass", r.ok, f"got {r.result}")
    r = guard("nu(1)", ctx4)
    test("T5_allow_list_fail", not r.ok, f"got {r.result}")

    # Test 6: Required capability
    ctx5 = GuardrailContext(
        policy={"required_capability": "ADMIN"},
        capabilities=set()
    )
    r = guard("sigma(1)", ctx5)
    test("T6_missing_capability", not r.ok and r.fail_record["fail_type"] == "MISSING_CAPABILITY",
         f"got {r.result}")

    ctx6 = GuardrailContext(
        policy={"required_capability": "ADMIN"},
        capabilities={"ADMIN"}
    )
    r = guard("sigma(1)", ctx6)
    test("T6_has_capability", r.ok, f"got {r.result}")

    # Test 7: Instruction/content separation
    ic_cert = {
        "schema_id": "QA_INSTRUCTION_CONTENT_SEPARATION_CERT.v1",
        "instruction_domain": ["sigma", "mu", "lambda", "nu"],
        "content_domain": ["user_input", "payload"],
    }
    ctx7 = GuardrailContext(instruction_content_cert=ic_cert)
    r = guard("sigma(1)", ctx7)
    test("T7_ic_pass", r.ok, f"got {r.result}")

    # Test 8: Content promoted to instruction (should fail)
    ic_cert_bad = {
        "schema_id": "QA_INSTRUCTION_CONTENT_SEPARATION_CERT.v1",
        "instruction_domain": ["mu", "lambda"],
        "content_domain": ["sigma"],  # sigma is content, not instruction
    }
    ctx8 = GuardrailContext(instruction_content_cert=ic_cert_bad)
    r = guard("sigma(1)", ctx8)
    test("T8_promotion_forbidden", not r.ok and r.fail_record["fail_type"] == "PROMOTION_FORBIDDEN",
         f"got {r.result}")

    # Test 9: Trace continuity
    ctx9 = GuardrailContext(
        trace_tail=[{"move": "sigma(1)", "fail_type": "OUT_OF_BOUNDS", "invariant_diff": {}}],
        policy={}
    )
    r = guard("sigma(2)", ctx9)
    test("T9_trace_failure_blocks", not r.ok and r.fail_record["fail_type"] == "TRACE_INVARIANT_BROKEN",
         f"got {r.result}")

    # Test 10: Trace continuity with allow_after_failure
    ctx10 = GuardrailContext(
        trace_tail=[{"move": "sigma(1)", "fail_type": "OUT_OF_BOUNDS", "invariant_diff": {}}],
        policy={"allow_after_failure": True}
    )
    r = guard("sigma(2)", ctx10)
    test("T10_allow_after_failure", r.ok, f"got {r.result}")

    # Test 11: Fail record structure
    fr = make_fail_record("test_move", "UNAUTHORIZED_GENERATOR", {"key": "val"}, "test detail")
    test("T11_fail_record_move", fr["move"] == "test_move", f"got {fr['move']}")
    test("T11_fail_record_type", fr["fail_type"] == "UNAUTHORIZED_GENERATOR", f"got {fr['fail_type']}")
    test("T11_fail_record_diff", fr["invariant_diff"] == {"key": "val"}, f"got {fr['invariant_diff']}")
    test("T11_fail_record_timestamp", "timestamp_utc" in fr, "missing timestamp")

    # Test 12: Invalid fail type raises
    try:
        make_fail_record("move", "INVALID_TYPE", {})
        test("T12_invalid_fail_type", False, "should have raised ValueError")
    except ValueError:
        test("T12_invalid_fail_type", True, "")

    # Test 13: Batch guard
    ctx = GuardrailContext()
    results_batch = guard_batch(["sigma(1)", "mu()", "unknown()"], ctx)
    test("T13_batch_count", len(results_batch) == 3, f"got {len(results_batch)}")
    test("T13_batch_allow", results_batch[0].ok and results_batch[1].ok, "first two should allow")
    test("T13_batch_deny", not results_batch[2].ok, "last should deny")

    # Test 14: GuardrailResult serialization
    r = GuardrailResult.allow(checks=["check1: OK"])
    d = r.to_dict()
    test("T14_result_dict_ok", d["ok"] is True, f"got {d['ok']}")
    test("T14_result_dict_result", d["result"] == "ALLOW", f"got {d['result']}")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QA Guardrail MVP")
    parser.add_argument("--validate", action="store_true", help="Output self-test results as JSON")
    parser.add_argument("--fixtures", action="store_true", help="Validate golden fixtures")
    parser.add_argument("--fixtures-dir", default=None, help="Directory containing fixtures")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.fixtures:
        fixtures_dir = args.fixtures_dir or os.path.join(base_dir, "fixtures")
        if os.path.isdir(fixtures_dir):
            result = validate_fixtures(fixtures_dir)
            print(json.dumps(result, indent=2))
            sys.exit(0 if result["ok"] else 1)
        else:
            print(f"Fixtures directory not found: {fixtures_dir}")
            sys.exit(1)

    results = run_self_tests()

    if args.validate:
        print(json.dumps(results, indent=2))
    else:
        print("QA Guardrail MVP - Self Tests")
        print("=" * 40)
        for test in results["tests"]:
            print(f"  {test}")
        if results["errors"]:
            print()
            print("Errors:")
            for err in results["errors"]:
                print(f"  - {err}")
        print()
        print(f"Result: {'PASS' if results['ok'] else 'FAIL'}")

    sys.exit(0 if results["ok"] else 1)
