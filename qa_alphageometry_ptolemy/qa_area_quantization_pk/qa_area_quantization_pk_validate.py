#!/usr/bin/env python3
"""
QA Inert Prime Area Quantization Certificate Validator
Family [111] — family_extension of QA_CORE_SPEC.v1 [107]

Checks: IH1-IH4 (inheritance), PK1-PK4 (inert-prime domain)

Theorem: for p inert in Z[phi] (Legendre(5,p)=-1) and modulus = p^k,
  Im(f) = {r in Z/p^k Z : v_p(r) != 1}
  Forbidden = {r : p|r but p^2 ∤ r}
"""

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

SCHEMA_VERSION = "QA_AREA_QUANTIZATION_PK_CERT.v1"
CERT_TYPE = "area_quantization_pk"
KERNEL_VERSION = "QA_CORE_SPEC.v1"
REQUIRED_SPEC_SCOPE = "family_extension"
REQUIRED_GATES = list(range(6))
KERNEL_FAILURE_TYPES = frozenset([
    "OUT_OF_BOUNDS", "PARITY", "INVARIANT_BREAK", "ZERO_DENOMINATOR"
])


class _Out:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def _legendre(a: int, p: int) -> int:
    """Legendre symbol (a/p) for odd prime p."""
    val = pow(a, (p - 1) // 2, p)
    return -1 if val == p - 1 else val


def _is_inert(p: int) -> bool:
    """True iff p is inert in Z[phi]: x^2+x-1 is irreducible mod p.
    Equivalent to Legendre(5, p) == -1 (discriminant of x^2+x-1 is 5)."""
    if p == 2 or p == 5:
        return False
    return _legendre(5, p) == -1


def _compute_spectrum(modulus: int) -> set[int]:
    """Exhaustively compute Im(f) where f(b,e) = b^2+be-e^2 mod modulus."""
    s: set[int] = set()
    for b in range(modulus):
        for e in range(modulus):
            s.add((b * b + b * e - e * e) % modulus)
    return s


def _theorem_forbidden(p: int, k: int) -> set[int]:
    """Predicted forbidden set: {r in {0..p^k - 1} : v_p(r) == 1}."""
    modulus = p ** k
    return {r for r in range(modulus) if r % p == 0 and r % (p * p) != 0}


def validate_pk_cert(cert: dict) -> dict:
    out = _Out()

    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail(f"schema_version must be '{SCHEMA_VERSION}'")
        return _build_result(cert, out, set())

    if cert.get("cert_type") != CERT_TYPE:
        out.fail(f"cert_type must be '{CERT_TYPE}'")
        return _build_result(cert, out, set())

    declared_result = cert.get("result", "")
    if declared_result not in ("PASS", "FAIL"):
        out.fail(f"result must be PASS or FAIL, got '{declared_result}'")
        return _build_result(cert, out, set())

    detected: set[str] = set()

    # IH1
    if cert.get("inherits_from") != KERNEL_VERSION:
        detected.add("INVALID_KERNEL_REFERENCE")

    # IH2
    if cert.get("spec_scope") != REQUIRED_SPEC_SCOPE:
        detected.add("SPEC_SCOPE_MISMATCH")

    # IH3: gate policy
    compat = cert.get("core_kernel_compatibility", {})
    gate_policy = compat.get("gate_policy_respected", [])
    if not all(g in gate_policy for g in REQUIRED_GATES):
        detected.add("GATE_POLICY_INCOMPATIBLE")

    # (IH4 placeholder — no explicit failure_algebra in this schema v1,
    # but included for future schema versions)

    # PK1: p is inert
    prime_p = cert.get("prime_p")
    if prime_p is None or not isinstance(prime_p, int) or prime_p < 2:
        detected.add("PRIME_NOT_INERT")
    else:
        if not _is_inert(prime_p):
            detected.add("PRIME_NOT_INERT")

    # PK2: modulus == p^k
    k = cert.get("k")
    modulus = cert.get("modulus")
    if prime_p and k and modulus:
        expected_modulus = prime_p ** k
        if modulus != expected_modulus:
            detected.add("MODULUS_MISMATCH")
    else:
        detected.add("MODULUS_MISMATCH")

    # PK3 + PK4: recompute and compare (only if prior checks passed)
    if "PRIME_NOT_INERT" not in detected and "MODULUS_MISMATCH" not in detected:
        actual_spectrum = _compute_spectrum(modulus)
        theorem_forb = _theorem_forbidden(prime_p, k)

        claim = cert.get("theorem_claim", {})
        claimed_spectrum = set(claim.get("spectrum", []))
        claimed_forbidden = set(claim.get("forbidden", []))

        # PK3: spectrum match
        if actual_spectrum != claimed_spectrum:
            detected.add("SPECTRUM_MISMATCH")

        # PK4: forbidden set matches theorem prediction
        actual_forbidden = set(range(modulus)) - actual_spectrum
        if claimed_forbidden != theorem_forb:
            detected.add("FORBIDDEN_SET_MISMATCH")
        # Also warn if the claim doesn't match the actual forbidden (independent check)
        if claimed_forbidden != actual_forbidden and "FORBIDDEN_SET_MISMATCH" not in detected:
            out.warn(
                f"claimed_forbidden={sorted(claimed_forbidden)} differs from "
                f"actual_forbidden={sorted(actual_forbidden)} though theorem prediction matches"
            )

    return _build_result(cert, out, detected)


def _build_result(cert: dict, out: _Out, detected: set[str]) -> dict:
    declared_result = cert.get("result", "")
    declared_fail_types = {
        e["fail_type"] for e in cert.get("fail_ledger", []) if "fail_type" in e
    }

    if declared_result == "PASS":
        if detected:
            out.fail(f"cert declares PASS but detected failures: {sorted(detected)}")
        elif declared_fail_types:
            out.fail(f"cert declares PASS but fail_ledger is non-empty: {sorted(declared_fail_types)}")
    elif declared_result == "FAIL":
        if not declared_fail_types:
            out.fail("cert declares FAIL but fail_ledger is empty")
        elif detected != declared_fail_types:
            out.fail(
                f"declared fail_ledger {sorted(declared_fail_types)} != "
                f"detected {sorted(detected)}"
            )

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
        "detected_fails": sorted(detected),
    }


def _self_test() -> dict:
    fixtures_dir = HERE / "fixtures"
    expected = {
        "pk_pass_p3_k2.json":          True,
        "pk_pass_p3_k3.json":          True,
        "pk_pass_p7_k2.json":          True,
        "pk_fail_wrong_forbidden.json": True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        with open(fpath) as f:
            cert = json.load(f)
        r = validate_pk_cert(cert)
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


def main():
    if "--self-test" in sys.argv:
        result = _self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        fpath = pathlib.Path(sys.argv[idx + 1])
        with open(fpath) as f:
            cert = json.load(f)
        r = validate_pk_cert(cert)
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)

    print("Usage: qa_area_quantization_pk_validate.py [--self-test] [--file <path>]")
    sys.exit(1)


if __name__ == "__main__":
    main()
