#!/usr/bin/env python3
"""QA Area Quantization [108] validator — QA_AREA_QUANTIZATION_CERT.v1

Inheritance checks (kernel compatibility):
  IH1  inherits_from == 'QA_CORE_SPEC.v1'               → INVALID_KERNEL_REFERENCE
  IH2  spec_scope == 'family_extension'                  → SPEC_SCOPE_MISMATCH
  IH3  gate_policy_inherited == [0,1,2,3,4,5]            → GATE_POLICY_INCOMPATIBLE
  IH4  failure_algebra includes all kernel types         → FAILURE_ALGEBRA_BREAKS_KERNEL

Domain checks (area quantization):
  AQ1  quadrea_claim.spectrum matches computed set       → QUADREA_MISMATCH
  AQ2  forbidden_values == complement of spectrum        → FORBIDDEN_QUADREA_INCORRECT

Usage:
  python qa_area_quantization_validate.py --self-test
  python qa_area_quantization_validate.py --file fixtures/area_quant_pass_mod9.json
"""

import json
import sys
import argparse
from pathlib import Path


# ── kernel contract ───────────────────────────────────────────────────────────
KERNEL_VERSION = "QA_CORE_SPEC.v1"
REQUIRED_GATES = [0, 1, 2, 3, 4, 5]
KERNEL_FAILURE_TYPES = frozenset([
    "OUT_OF_BOUNDS", "PARITY", "INVARIANT_BREAK", "ZERO_DENOMINATOR"
])
REQUIRED_LOG_FIELDS = frozenset(["move", "fail_type", "invariant_diff"])

AREA_QUANT_FAIL_TYPES = frozenset([
    "INVALID_KERNEL_REFERENCE",
    "SPEC_SCOPE_MISMATCH",
    "GATE_POLICY_INCOMPATIBLE",
    "FAILURE_ALGEBRA_BREAKS_KERNEL",
    "QUADREA_MISMATCH",
    "FORBIDDEN_QUADREA_INCORRECT",
])


# ── quadrea spectrum computation ──────────────────────────────────────────────
def _compute_quadrea_spectrum(m: int) -> set:
    """Compute {f(b,e) mod m : b,e in 0..m-1} where f(b,e)=b^2+be-e^2."""
    return {(b * b + b * e - e * e) % m for b in range(m) for e in range(m)}


# ── output accumulator ────────────────────────────────────────────────────────
class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


# ── validator ─────────────────────────────────────────────────────────────────
def validate_area_quantization_cert(cert: dict) -> dict:
    out = _Out()
    detected_fails: set[str] = set()

    # ── schema_version / cert_type ──────────────────────────────────────────
    if cert.get("schema_version") != "QA_AREA_QUANTIZATION_CERT.v1":
        out.fail(f"schema_version must be 'QA_AREA_QUANTIZATION_CERT.v1', got {cert.get('schema_version')!r}")
    if cert.get("cert_type") != "qa_area_quantization":
        out.fail(f"cert_type must be 'qa_area_quantization', got {cert.get('cert_type')!r}")

    # ── required fields ─────────────────────────────────────────────────────
    for field in ["certificate_id", "inherits_from", "spec_scope",
                  "core_kernel_compatibility", "modulus", "quadrea_form",
                  "quadrea_claim", "generators", "invariants",
                  "failure_algebra", "logging", "validation",
                  "validation_checks", "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    if out.errors:
        return _reconcile(cert, out, detected_fails)

    # ── IH1: inherits_from ──────────────────────────────────────────────────
    if cert.get("inherits_from") != KERNEL_VERSION:
        detected_fails.add("INVALID_KERNEL_REFERENCE")

    # ── IH2: spec_scope ────────────────────────────────────────────────────
    if cert.get("spec_scope") != "family_extension":
        detected_fails.add("SPEC_SCOPE_MISMATCH")

    # ── IH3: gate_policy_inherited ─────────────────────────────────────────
    compat = cert.get("core_kernel_compatibility", {})
    gate_policy = compat.get("gate_policy_inherited", [])
    if gate_policy != REQUIRED_GATES:
        detected_fails.add("GATE_POLICY_INCOMPATIBLE")

    # ── IH4: failure algebra extends kernel ─────────────────────────────────
    fa_types = set(cert.get("failure_algebra", {}).get("types", []))
    if not KERNEL_FAILURE_TYPES.issubset(fa_types):
        detected_fails.add("FAILURE_ALGEBRA_BREAKS_KERNEL")

    # ── AQ1: quadrea spectrum matches computed ──────────────────────────────
    # Only run domain checks when kernel reference is valid (IH1 passes).
    # If kernel ref is wrong we can still run domain checks independently.
    modulus = cert.get("modulus")
    if isinstance(modulus, int) and modulus >= 2:
        actual_spectrum = _compute_quadrea_spectrum(modulus)
        claimed_spectrum = set(cert.get("quadrea_claim", {}).get("spectrum", []))
        if claimed_spectrum != actual_spectrum:
            detected_fails.add("QUADREA_MISMATCH")

        # ── AQ2: forbidden_values == complement ────────────────────────────
        all_values = set(range(modulus))
        expected_forbidden = all_values - actual_spectrum
        claimed_forbidden = set(cert.get("quadrea_claim", {}).get("forbidden_values", []))
        if claimed_forbidden != expected_forbidden:
            detected_fails.add("FORBIDDEN_QUADREA_INCORRECT")
    else:
        out.warn(f"modulus is invalid ({modulus!r}); skipping domain checks AQ1/AQ2")

    return _reconcile(cert, out, detected_fails)


def _reconcile(cert: dict, out: _Out, detected_fails: set) -> dict:
    declared_result = cert.get("result", "")
    declared_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {e.get("fail_type") for e in declared_ledger
                           if isinstance(e, dict)}

    for ft in declared_fail_types:
        if ft not in AREA_QUANT_FAIL_TYPES:
            out.warn(f"unrecognised fail_type in fail_ledger: {ft!r}")

    if declared_result == "PASS":
        if detected_fails:
            for ft in sorted(detected_fails):
                out.fail(f"cert declares PASS but detected: {ft}")
    elif declared_result == "FAIL":
        missing_from_ledger = detected_fails - declared_fail_types
        for ft in sorted(missing_from_ledger):
            out.warn(f"detected {ft} but not declared in fail_ledger")
        phantom = declared_fail_types - detected_fails
        for ft in sorted(phantom):
            out.warn(f"fail_ledger declares {ft} but validator did not detect it")
    else:
        out.fail(f"result must be 'PASS' or 'FAIL', got {declared_result!r}")

    ok = len(out.errors) == 0
    label = "PASS" if ok else "FAIL"
    if ok and out.warnings:
        label = "PASS_WITH_WARNINGS"

    return {
        "ok": ok,
        "label": label,
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": out.warnings,
        "detected_fails": sorted(detected_fails),
    }


# ── file entry point ──────────────────────────────────────────────────────────
def validate_file(path: Path) -> dict:
    with open(path) as f:
        cert = json.load(f)
    return validate_area_quantization_cert(cert)


# ── self-test ─────────────────────────────────────────────────────────────────
def self_test() -> dict:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = {
        "area_quant_pass_mod9.json": True,
        "area_quant_fail_wrong_kernel_ref.json": True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        r = validate_file(fpath)
        passed = r["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": r["label"],
            "errors": r["errors"],
            "warnings": r["warnings"],
        })

    return {"ok": all_ok, "results": results}


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="QA Area Quantization [108] validator")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test against fixture suite")
    parser.add_argument("--file", type=Path,
                        help="Validate a single cert file")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        result = validate_file(args.file)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
