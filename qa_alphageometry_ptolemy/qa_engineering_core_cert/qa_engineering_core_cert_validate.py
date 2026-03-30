#!/usr/bin/env python3
"""
qa_engineering_core_cert_validate.py

Validator for QA_ENGINEERING_CORE_CERT.v1  [family 121]

Family extension of QA_CORE_SPEC.v1.  Certifies that a classical engineering
system (state-space model, stability conditions, controllability claim) maps
validly to a QA specification, and that arithmetic obstructions are not
silently ignored by classical controllability analysis.

Checks
------
IH1  inherits_from == 'QA_CORE_SPEC.v1'
IH2  spec_scope == 'family_extension'
IH3  gate_policy_respected == [0,1,2,3,4,5]

EC1  all state_encoding entries: 1 <= b,e <= modulus
EC2  all transitions have a non-empty generator name
EC3  all failure_modes map to a QA canonical fail type
EC4  target_condition.orbit_family in {singularity, satellite, cosmos}
EC5  declared orbit_family for each state matches recomputed f(b,e) valuation
EC6  stability_claim.lyapunov_function is non-empty and mentions a QA invariant
EC7  stability_claim.orbit_contraction_factor < 1.0
EC8  equilibrium_state resolves to a state with orbit_family == singularity
EC9  reachability_witness is present when classical_controllability == full_rank
EC10 minimality_witness present when optimization_claim is present
EC11 obstruction_check.obstructed matches recomputed v_p(target_r) for inert primes

Usage
-----
    python qa_engineering_core_cert_validate.py --cert fixtures/engineering_core_pass_spring_mass.json
    python qa_engineering_core_cert_validate.py --cert fixtures/engineering_core_fail_arithmetic_obstruction.json
    python qa_engineering_core_cert_validate.py --cert fixtures/engineering_core_fail_invalid_encoding.json
    python qa_engineering_core_cert_validate.py --demo
    python qa_engineering_core_cert_validate.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repo root is on path so qa_orbit_rules is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qa_orbit_rules import orbit_family  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state"

SCHEMA_VERSION = "QA_ENGINEERING_CORE_CERT.v1"
KERNEL_VERSION = "QA_CORE_SPEC.v1"
REQUIRED_GATES = [0, 1, 2, 3, 4, 5]

QA_FAIL_TYPES: Set[str] = {
    "OUT_OF_BOUNDS", "PARITY", "PHASE_VIOLATION", "INVARIANT", "REDUCTION",
}

ORBIT_FAMILIES: Set[str] = {"singularity", "satellite", "cosmos"}

ENGINEERING_FAIL_TYPES: Set[str] = {
    "INVALID_KERNEL_REFERENCE",
    "SPEC_SCOPE_MISMATCH",
    "GATE_POLICY_INCOMPATIBLE",
    "STATE_ENCODING_INVALID",
    "TRANSITION_NOT_GENERATOR",
    "FAILURE_TAXONOMY_INCOMPLETE",
    "TARGET_NOT_ORBIT_FAMILY",
    "ORBIT_FAMILY_CLASSIFICATION_FAILURE",
    "LYAPUNOV_QA_MISMATCH",
    "CONTROLLABILITY_QA_MISMATCH",
    "ARITHMETIC_OBSTRUCTION_IGNORED",
}

# Inert primes by modulus (p inert in Z[phi])
INERT_PRIMES: Dict[int, List[int]] = {
    9:  [3],
    24: [3, 7],
}


# ---------------------------------------------------------------------------
# QA arithmetic helpers
# ---------------------------------------------------------------------------

def v_p(n: int, p: int) -> int:
    """p-adic valuation of n (number of times p divides |n|). Returns 0 for n==0."""
    if n == 0:
        return 0
    n = abs(n)
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return val


def qa_norm(b: int, e: int) -> int:
    """Q(sqrt5) norm: f(b,e) = b*b + b*e - e*e  (use multiplication, not **)"""
    return b * b + b * e - e * e


# orbit_family(b, e, m) imported from qa_orbit_rules — canonical implementation


def is_obstructed(target_r: int, inert_primes: List[int]) -> bool:
    """Return True if v_p(target_r) == 1 for any inert prime p."""
    return any(v_p(target_r, p) == 1 for p in inert_primes)


# ---------------------------------------------------------------------------
# Canonical JSON + hash
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class EngineeringCoreResult:
    def __init__(self, cert_id: str) -> None:
        self.cert_id = cert_id
        self.ok: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks_passed: int = 0
        self.checks_total: int = 0
        self.hash: str = ""

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def label(self) -> str:
        if not self.ok:
            return "FAIL"
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        return "PASS"

    def report(self) -> str:
        lines = [
            f"  cert_id  : {self.cert_id}",
            f"  result   : {self.label}",
            f"  hash     : {self.hash[:16]}...",
            f"  checks   : {self.checks_passed}/{self.checks_total}",
        ]
        if self.errors:
            lines.append("  errors:")
            for e in self.errors:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append("  warnings:")
            for w in self.warnings:
                lines.append(f"    ~ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_engineering_core_cert(cert: Dict[str, Any]) -> EngineeringCoreResult:
    """Validate QA_ENGINEERING_CORE_CERT.v1"""
    cid = cert.get("certificate_id", "unknown")
    out = EngineeringCoreResult(cid)
    out.hash = sha256_hex(canonical_json(cert))

    # Schema guard
    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail(f"Bad schema_version: {cert.get('schema_version')!r}; expected {SCHEMA_VERSION!r}")
        return out
    if cert.get("cert_type") != "engineering_core":
        out.fail(f"Expected cert_type='engineering_core', got {cert.get('cert_type')!r}")
        return out

    detected: Set[str] = set()
    compat = cert.get("core_kernel_compatibility", {})
    csys   = cert.get("classical_system", {})
    stab   = cert.get("stability_claim", {})
    ctrl   = cert.get("controllability_claim", {})
    obstr  = cert.get("obstruction_check", {})

    checks      = cert.get("validation_checks", [])
    fail_ledger = cert.get("fail_ledger", [])
    declared    = cert.get("result")

    out.checks_total  = len(checks)
    out.checks_passed = sum(1 for c in checks if c.get("passed"))

    # Validate fail_ledger types
    for entry in fail_ledger:
        ft = entry.get("fail_type")
        if ft not in ENGINEERING_FAIL_TYPES:
            out.fail(f"Unknown fail_type in fail_ledger: {ft!r}")

    # --- IH1: kernel reference ---
    if cert.get("inherits_from") != KERNEL_VERSION:
        detected.add("INVALID_KERNEL_REFERENCE")

    # --- IH2: spec_scope ---
    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")

    # --- IH3: gate_policy_respected ---
    if compat.get("gate_policy_respected") != REQUIRED_GATES:
        detected.add("GATE_POLICY_INCOMPATIBLE")

    modulus = csys.get("modulus", 9)
    inert   = INERT_PRIMES.get(modulus, [3])

    # Build label → (b, e, orbit_family) lookup
    state_map: Dict[str, Dict[str, Any]] = {}
    for se in csys.get("state_encoding", []):
        lbl = se.get("label", "")
        if lbl:
            state_map[lbl] = se

    # --- EC1: state encoding bounds ---
    for se in csys.get("state_encoding", []):
        b = se.get("b")
        e = se.get("e")
        lbl = se.get("label", "?")
        if not (isinstance(b, int) and isinstance(e, int) and 1 <= b <= modulus and 1 <= e <= modulus):
            detected.add("STATE_ENCODING_INVALID")
            break  # one bad encoding is enough to flag

    # --- EC2: transitions have generator names ---
    for tr in csys.get("transitions", []):
        g = tr.get("generator", "")
        if not isinstance(g, str) or not g.strip():
            detected.add("TRANSITION_NOT_GENERATOR")
            break

    # --- EC3: failure modes map to QA types ---
    for fm in csys.get("failure_modes", []):
        qt = fm.get("qa_fail_type", "")
        if qt not in QA_FAIL_TYPES:
            detected.add("FAILURE_TAXONOMY_INCOMPLETE")
            break

    # --- EC4: target orbit family valid ---
    tgt_family = csys.get("target_condition", {}).get("orbit_family", "")
    if tgt_family not in ORBIT_FAMILIES:
        detected.add("TARGET_NOT_ORBIT_FAMILY")

    # --- EC5: declared orbit_family matches recomputed f(b,e) ---
    if "STATE_ENCODING_INVALID" not in detected:
        for se in csys.get("state_encoding", []):
            b = se.get("b", 0)
            e = se.get("e", 0)
            declared_fam = se.get("orbit_family", "")
            computed_fam = orbit_family(b, e, modulus)
            if declared_fam != computed_fam:
                detected.add("ORBIT_FAMILY_CLASSIFICATION_FAILURE")
                break

    # --- EC6: lyapunov_function mentions a QA invariant ---
    lyap = stab.get("lyapunov_function", "")
    if not lyap or not any(kw in lyap.lower() for kw in ("f(b", "f(b,", " f ", "|i|", "i =", "i=", "b*b", "b*e", "norm", "invariant")):
        detected.add("LYAPUNOV_QA_MISMATCH")

    # --- EC7: orbit_contraction_factor < 1.0 ---
    rho = stab.get("orbit_contraction_factor")
    if not isinstance(rho, (int, float)) or rho >= 1.0:
        detected.add("LYAPUNOV_QA_MISMATCH")

    # --- EC8: equilibrium_state resolves to singularity ---
    eq_label = csys.get("equilibrium_state", "")
    eq_state = state_map.get(eq_label)
    if eq_state is None or eq_state.get("orbit_family") != "singularity":
        detected.add("LYAPUNOV_QA_MISMATCH")

    # --- EC9: reachability_witness present when full_rank ---
    classical_ctrl = ctrl.get("classical_controllability", "")
    rw = ctrl.get("reachability_witness")
    if classical_ctrl == "full_rank" and rw is None:
        detected.add("CONTROLLABILITY_QA_MISMATCH")

    # --- EC10: minimality_witness present when optimization_claim present ---
    opt = ctrl.get("optimization_claim")
    if opt is not None:
        mw = opt.get("minimality_witness")
        if mw is None:
            detected.add("CONTROLLABILITY_QA_MISMATCH")

    # --- EC11: arithmetic obstruction ---
    target_label = csys.get("target_condition", {}).get("label", "")
    target_state = state_map.get(target_label)
    if target_state is not None and "STATE_ENCODING_INVALID" not in detected:
        b_t = target_state.get("b", 1)
        e_t = target_state.get("e", 1)
        target_r = b_t * e_t
        # Cross-check declared target_r
        declared_r = obstr.get("target_r")
        if declared_r is not None and declared_r != target_r:
            detected.add("ARITHMETIC_OBSTRUCTION_IGNORED")
        elif is_obstructed(target_r, inert):
            # Recomputed obstruction = True
            if not obstr.get("obstructed", False):
                detected.add("ARITHMETIC_OBSTRUCTION_IGNORED")
        else:
            # Recomputed obstruction = False
            if obstr.get("obstructed", False):
                out.warn(f"obstruction_check.obstructed=True but v_p(r={target_r}) ≠ 1 for inert primes {inert} — verify cert")

    # --- Result consistency ---
    ledger_fail_types = {e.get("fail_type") for e in fail_ledger}
    ledger_has_fails = len(fail_ledger) > 0

    if declared == "PASS" and ledger_has_fails:
        out.fail("result=PASS but fail_ledger is non-empty")

    if declared == "FAIL" and not ledger_has_fails:
        out.warn("result=FAIL but fail_ledger is empty — consider documenting the failure")

    if detected:
        if declared == "PASS":
            out.fail(
                f"Recomputed engineering failures {sorted(detected)} but result=PASS — inconsistency"
            )
        else:
            missing = sorted(detected - ledger_fail_types)
            if missing:
                out.warn(
                    "Recomputed failures missing from fail_ledger: " + ", ".join(missing)
                )
    else:
        if declared == "FAIL":
            out.warn("Declared FAIL but validator recomputed no engineering failure — verify cert")

    return out


# ---------------------------------------------------------------------------
# File dispatch
# ---------------------------------------------------------------------------

def validate_file(path: str) -> EngineeringCoreResult:
    with open(path) as f:
        cert = json.load(f)
    return validate_engineering_core_cert(cert)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="QA Engineering Core certificate validator [family 121]"
    )
    parser.add_argument("--cert",      metavar="FILE", help="Validate a single cert file")
    parser.add_argument("--demo",      action="store_true", help="Run all fixtures, print results")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="Run all fixtures; emit JSON {ok, passed, failed} to stdout")
    args = parser.parse_args()

    here         = Path(__file__).parent
    fixtures_dir = here / "fixtures"

    results: List[Tuple[str, EngineeringCoreResult]] = []

    if args.cert:
        r = validate_file(args.cert)
        results.append((args.cert, r))

    if args.demo or args.self_test:
        for fp in sorted(fixtures_dir.glob("*.json")):
            r = validate_file(str(fp))
            results.append((fp.name, r))

    if not results:
        parser.print_help()
        return 0

    passed = sum(1 for _, r in results if r.ok)
    failed = sum(1 for _, r in results if not r.ok)

    if args.self_test:
        print(json.dumps({
            "ok": failed == 0,
            "passed": passed,
            "failed": failed,
            "total":  passed + failed,
            "details": [
                {"name": name, "result": r.label, "cert_type": "engineering_core"}
                for name, r in results
            ],
        }))
        return 0 if failed == 0 else 1

    print()
    for name, r in results:
        print(f"[{r.label}] {name}")
        print(r.report())
        print()

    print("=" * 60)
    print(f"Total: {passed+failed}  PASS: {passed}  FAIL: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
