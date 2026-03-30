#!/usr/bin/env python3
"""
qa_plan_control_compiler_validate.py

Validator for QA_PLAN_CONTROL_COMPILER_CERT.v1.

Proves the certifiable compilation relation between a QA planner witness
and a QA control witness over a shared generator algebra:

    CC1  source planner cert found in domain fixtures
    CC2  source cert hash matches declared hash
    CC3  target control cert found in domain fixtures
    CC4  target cert hash matches declared hash
    CC5  initial_pattern_class consistent: planner == control == claimed
    CC6  target_pattern_class consistent: planner == control == claimed
    CC7  path_length_k consistent: planner == control == claimed
    CC8  move_sequence consistent: planner == control == claimed
    CC9  final_orbit_family consistent: planner == control == claimed

Usage:
    python qa_plan_control_compiler_validate.py --cert fixtures/compiler_cert_pass_cymatics_hexagon.json
    python qa_plan_control_compiler_validate.py --cert fixtures/compiler_cert_fail_sequence_mismatch.json
    python qa_plan_control_compiler_validate.py --demo
    python qa_plan_control_compiler_validate.py --self-test
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical JSON + hashing (must match qa_cymatics_validate.py)
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cert_hash(cert: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json(cert))


# ---------------------------------------------------------------------------
# Domain fixture resolution
# ---------------------------------------------------------------------------

# Maps domain name to fixture directory path relative to qa_alphageometry_ptolemy/
DOMAIN_TO_FIXTURES: Dict[str, str] = {
    "cymatics": "qa_cymatics/fixtures",
    # extend for future domains: "rule30": "qa_rule30/certpacks/...", etc.
}


def _find_cert_by_id(cert_id: str, search_dir: Path) -> Optional[Dict[str, Any]]:
    """Search a fixtures directory for a cert whose certificate_id matches cert_id."""
    try:
        for fp in sorted(search_dir.glob("*.json")):
            try:
                with open(fp) as f:
                    cert = json.load(f)
                if cert.get("certificate_id") == cert_id:
                    return cert
            except Exception:
                pass
    except Exception:
        pass
    return None


def _find_cert_in_domain(
    cert_id: str, domain: str, repo_root: Path
) -> Optional[Dict[str, Any]]:
    """Find a cert by ID in the specified domain's fixtures directory."""
    rel = DOMAIN_TO_FIXTURES.get(domain)
    if rel is None:
        return None
    return _find_cert_by_id(cert_id, repo_root / rel)


# ---------------------------------------------------------------------------
# Cert-type-aware field extraction
# ---------------------------------------------------------------------------

def _extract_moves(cert: Dict[str, Any]) -> Optional[List[str]]:
    """Extract ordered move sequence from a planner or control cert."""
    ct = cert.get("cert_type")
    if ct == "cymatic_planner":
        return [s.get("move") for s in cert.get("plan_witness", {}).get("steps", [])]
    if ct == "cymatic_control":
        return [s.get("move") for s in cert.get("generator_sequence", [])]
    return None


def _extract_initial(cert: Dict[str, Any]) -> Optional[str]:
    ct = cert.get("cert_type")
    if ct == "cymatic_planner":
        return cert.get("planning_problem", {}).get("initial_pattern_class")
    if ct == "cymatic_control":
        return cert.get("initial_state", {}).get("pattern_class")
    return None


def _extract_target(cert: Dict[str, Any]) -> Optional[str]:
    ct = cert.get("cert_type")
    if ct == "cymatic_planner":
        return cert.get("planning_problem", {}).get("target_pattern_class")
    if ct == "cymatic_control":
        return cert.get("target_spec", {}).get("target_pattern_class")
    return None


def _extract_path_length(cert: Dict[str, Any]) -> Optional[int]:
    ct = cert.get("cert_type")
    if ct == "cymatic_planner":
        return cert.get("plan_witness", {}).get("path_length_k")
    if ct == "cymatic_control":
        return len(cert.get("generator_sequence", []))
    return None


def _extract_final_orbit(cert: Dict[str, Any]) -> Optional[str]:
    return cert.get("qa_mapping", {}).get("final_orbit_family")


# ---------------------------------------------------------------------------
# Failure algebra
# ---------------------------------------------------------------------------

KERNEL_VERSION = "QA_CORE_SPEC.v1"
REQUIRED_GATES = [0, 1, 2, 3, 4, 5]

COMPILER_FAIL_TYPES = frozenset([
    "INVALID_KERNEL_REFERENCE",
    "SPEC_SCOPE_MISMATCH",
    "GATE_POLICY_INCOMPATIBLE",
    "SOURCE_CERT_MISSING",
    "TARGET_CERT_MISSING",
    "GENERATOR_SEQUENCE_MISMATCH",
    "TARGET_INVARIANT_MISMATCH",
    "PATH_LENGTH_MISMATCH",
    "REPLAY_RESULT_MISMATCH",
    "COMPILATION_HASH_MISMATCH",
    "NORMALIZATION_RULE_MISMATCH",
])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class CompilerValidationResult:
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
# MAIN VALIDATOR
# ---------------------------------------------------------------------------

def validate_compiler_cert(cert: Dict[str, Any]) -> CompilerValidationResult:
    """Validate QA_PLAN_CONTROL_COMPILER_CERT.v1"""
    cid = cert.get("certificate_id", "unknown")
    out = CompilerValidationResult(cid)
    out.hash = cert_hash(cert)

    # Schema guard
    if cert.get("schema_version") != "QA_PLAN_CONTROL_COMPILER_CERT.v1":
        out.fail(f"Bad schema_version: {cert.get('schema_version')!r}")
        return out
    if cert.get("cert_type") != "plan_control_compiler":
        out.fail(f"Expected cert_type='plan_control_compiler', got {cert.get('cert_type')!r}")
        return out

    here       = Path(__file__).parent
    repo_root  = here.parent  # qa_alphageometry_ptolemy/
    detected: set = set()

    # --- IH1: kernel reference ---
    if cert.get("inherits_from") != KERNEL_VERSION:
        detected.add("INVALID_KERNEL_REFERENCE")

    # --- IH2: spec_scope ---
    if cert.get("spec_scope") != "family_extension":
        detected.add("SPEC_SCOPE_MISMATCH")

    # --- IH3: gate_policy_respected ---
    compat = cert.get("core_kernel_compatibility", {})
    if compat.get("gate_policy_respected") != REQUIRED_GATES:
        detected.add("GATE_POLICY_INCOMPATIBLE")

    src_id     = cert.get("source_planner_cert_id", "")
    src_hash   = cert.get("source_planner_cert_hash", "")
    src_domain = cert.get("source_domain", "")

    tgt_id     = cert.get("target_control_cert_id", "")
    tgt_hash   = cert.get("target_control_cert_hash", "")
    tgt_domain = cert.get("target_domain", "")

    claims     = cert.get("compilation_claims", {})
    claimed_initial  = claims.get("initial_pattern_class")
    claimed_target   = claims.get("target_pattern_class")
    claimed_length   = claims.get("path_length_k")
    claimed_moves    = claims.get("move_sequence", [])
    claimed_orbit    = claims.get("final_orbit_family")

    checks      = cert.get("validation_checks", [])
    fail_ledger = cert.get("fail_ledger", [])
    declared    = cert.get("result")

    ledger_fail_types = {e.get("fail_type") for e in fail_ledger}

    # Fail ledger type check
    for entry in fail_ledger:
        ft = entry.get("fail_type")
        if ft not in COMPILER_FAIL_TYPES:
            out.fail(f"Unknown fail_type in fail_ledger: {ft!r}")

    # --- CC1: find source cert ---
    src_cert = _find_cert_in_domain(src_id, src_domain, repo_root)
    if src_cert is None:
        detected.add("SOURCE_CERT_MISSING")
        src_moves = src_initial = src_target = src_length = src_orbit = None
    else:
        # CC2: hash check
        if cert_hash(src_cert) != src_hash:
            detected.add("COMPILATION_HASH_MISMATCH")
        src_moves   = _extract_moves(src_cert)
        src_initial = _extract_initial(src_cert)
        src_target  = _extract_target(src_cert)
        src_length  = _extract_path_length(src_cert)
        src_orbit   = _extract_final_orbit(src_cert)

    # --- CC3: find target cert ---
    tgt_cert = _find_cert_in_domain(tgt_id, tgt_domain, repo_root)
    if tgt_cert is None:
        detected.add("TARGET_CERT_MISSING")
        tgt_moves = tgt_initial = tgt_target = tgt_length = tgt_orbit = None
    else:
        # CC4: hash check
        if cert_hash(tgt_cert) != tgt_hash:
            detected.add("COMPILATION_HASH_MISMATCH")
        tgt_moves   = _extract_moves(tgt_cert)
        tgt_initial = _extract_initial(tgt_cert)
        tgt_target  = _extract_target(tgt_cert)
        tgt_length  = _extract_path_length(tgt_cert)
        tgt_orbit   = _extract_final_orbit(tgt_cert)

    # Only run semantic checks when both certs were found
    if src_cert is not None and tgt_cert is not None:
        # CC5: initial_pattern_class
        if not (src_initial == tgt_initial == claimed_initial):
            detected.add("TARGET_INVARIANT_MISMATCH")

        # CC6: target_pattern_class
        if not (src_target == tgt_target == claimed_target):
            detected.add("TARGET_INVARIANT_MISMATCH")

        # CC7: path_length_k
        if not (src_length == tgt_length == claimed_length):
            detected.add("PATH_LENGTH_MISMATCH")

        # CC8: move_sequence
        if not (src_moves == tgt_moves == claimed_moves):
            detected.add("GENERATOR_SEQUENCE_MISMATCH")

        # CC9: final_orbit_family
        if not (src_orbit == tgt_orbit == claimed_orbit):
            detected.add("REPLAY_RESULT_MISMATCH")

    # --- Validation checks accounting ---
    out.checks_total  = len(checks)
    out.checks_passed = sum(1 for c in checks if c.get("passed"))

    # --- Result consistency model ---
    ledger_has_fails = len(fail_ledger) > 0

    if declared == "PASS" and ledger_has_fails:
        out.fail("result=PASS but fail_ledger is non-empty")

    if declared == "FAIL" and not ledger_has_fails:
        out.warn("result=FAIL but fail_ledger is empty — consider documenting the failure")

    if detected:
        if declared == "PASS":
            out.fail(
                f"Recomputed compiler failures {sorted(detected)} but result=PASS — inconsistency"
            )
        else:
            missing = sorted(detected - ledger_fail_types)
            if missing:
                out.warn(
                    "Recomputed failures missing from fail_ledger: " + ", ".join(missing)
                )
    else:
        if declared == "FAIL":
            out.warn("Declared FAIL but validator recomputed no compiler failure — verify cert")

    return out


# ---------------------------------------------------------------------------
# File dispatch
# ---------------------------------------------------------------------------

def validate_file(path: str) -> CompilerValidationResult:
    with open(path) as f:
        cert = json.load(f)
    return validate_compiler_cert(cert)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="QA Plan-Control Compiler certificate validator")
    parser.add_argument("--cert",      metavar="FILE", help="Validate a plan_control_compiler cert file")
    parser.add_argument("--demo",      action="store_true", help="Run all fixtures and print results")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="Run all fixtures; emit JSON {ok, passed, failed} to stdout")
    args = parser.parse_args()

    here         = Path(__file__).parent
    fixtures_dir = here / "fixtures"

    results: List[Tuple[str, CompilerValidationResult]] = []

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
            "total": passed + failed,
            "details": [
                {"name": name, "result": r.label, "cert_type": "plan_control_compiler"}
                for name, r in results
            ],
        }))
        return 0 if failed == 0 else 1

    print()
    for name, r in results:
        print(f"[{r.label}] {name}")
        print(r.report())
        print()

    print(f"{'='*60}")
    print(f"Total: {passed+failed}  PASS: {passed}  FAIL: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
