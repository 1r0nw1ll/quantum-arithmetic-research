#!/usr/bin/env python3
"""QA Directrix Divisibility Cert validator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_DIRECTRIX_DIVISIBILITY_CERT.v1"
CERT_TYPE = "qa_directrix_divisibility_cert"
THEOREM_STATUS = "PROVEN_STRUCTURAL_DIVISIBILITY_REDUCTION"
REQUIRED_OBLIGATIONS = {
    "modular_reduction_d_congruent_b",
    "cube_congruence",
    "prime_exponent_kernel3",
    "bounded_audit",
    "orbit_context_non_theorem",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Out:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.detected_fails: set[str] = set()

    def fail(self, code: str, msg: str) -> None:
        self.detected_fails.add(code)
        self.errors.append(f"{code}: {msg}")


def factor(n: int) -> dict[int, int]:
    remaining = n
    out: dict[int, int] = {}
    while remaining % 2 == 0:
        out[2] = out.get(2, 0) + 1
        remaining //= 2
    p = 3
    while p * p <= remaining:
        while remaining % p == 0:
            out[p] = out.get(p, 0) + 1
            remaining //= p
        p += 2
    if remaining > 1:
        out[remaining] = out.get(remaining, 0) + 1
    return out


def int_power(base: int, exponent: int) -> int:
    out = 1
    for _ in range(exponent):
        out *= base
    return out


def kernel3(e: int) -> int:
    out = 1
    for prime, exponent in factor(e).items():
        out *= int_power(prime, math.ceil(exponent / 3))
    return out


def directrix_integer(b: int, e: int) -> bool:
    d = b + e
    return (d * d * d) % e == 0


def reduced_integer(b: int, e: int) -> bool:
    return (b * b * b) % e == 0


def kernel_integer(b: int, e: int) -> bool:
    return b % kernel3(e) == 0


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def audit_window(b_max: int, e_max: int) -> dict[str, int]:
    support = 0
    reduction_mismatches = 0
    kernel_mismatches = 0
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            direct = directrix_integer(b, e)
            reduced = reduced_integer(b, e)
            kernel = kernel_integer(b, e)
            if direct:
                support += 1
            if direct != reduced:
                reduction_mismatches += 1
            if reduced != kernel:
                kernel_mismatches += 1
    return {
        "pair_count": b_max * e_max,
        "directrix_support": support,
        "reduction_mismatch_count": reduction_mismatches,
        "kernel_mismatch_count": kernel_mismatches,
    }


def validate_cert(cert: dict[str, Any]) -> dict[str, Any]:
    out = Out()
    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail("SCHEMA_VERSION", f"schema_version must be {SCHEMA_VERSION}")
    if cert.get("cert_type") != CERT_TYPE:
        out.fail("CERT_TYPE", f"cert_type must be {CERT_TYPE}")
    if cert.get("theorem_status") != THEOREM_STATUS:
        out.fail("THEOREM_STATUS", f"theorem_status must be {THEOREM_STATUS}")

    required = [
        "certificate_id",
        "theorem_statement",
        "proof_obligations",
        "witnesses",
        "bounded_audit",
        "orbit_context",
        "non_claims",
        "result",
        "fail_ledger",
    ]
    for field in required:
        if field not in cert:
            out.fail("MISSING_FIELD", f"missing required field {field}")
    if out.errors:
        return reconcile(cert, out)

    obligations = cert.get("proof_obligations")
    if not isinstance(obligations, list):
        out.fail("PROOF_OBLIGATIONS", "proof_obligations must be a list")
    else:
        missing = REQUIRED_OBLIGATIONS - set(obligations)
        if missing:
            out.fail("PROOF_OBLIGATIONS", f"missing proof obligations {sorted(missing)}")

    statement = cert.get("theorem_statement", {})
    if not isinstance(statement, dict):
        out.fail("THEOREM_STATEMENT", "theorem_statement must be an object")
    elif statement.get("claims_orbit_lift_as_theorem") is True:
        out.fail("ORBIT_OVERCLAIM", "orbit lift must remain empirical context")

    witnesses = cert.get("witnesses")
    if not isinstance(witnesses, list) or not witnesses:
        out.fail("WITNESSES", "witnesses must be a nonempty list")
    else:
        for idx, row in enumerate(witnesses):
            if not isinstance(row, dict):
                out.fail("WITNESS_ROW", f"witness {idx} must be an object")
                continue
            if not _is_pos_int(row.get("b")) or not _is_pos_int(row.get("e")):
                out.fail("WITNESS_ROW", f"witness {idx} needs positive integer b,e")
                continue
            b = row["b"]
            e = row["e"]
            d = b + e
            k3 = kernel3(e)
            if row.get("d") != d:
                out.fail("D_WITNESS", f"witness {idx} d mismatch")
            if row.get("kernel3_e") != k3:
                out.fail("KERNEL3_WITNESS", f"witness {idx} kernel3 mismatch")
            expected = directrix_integer(b, e)
            if row.get("directrix_integer") is not expected:
                out.fail("DIRECTRIX_WITNESS", f"witness {idx} directrix truth mismatch")
            if expected != reduced_integer(b, e):
                out.fail("REDUCTION_IDENTITY", f"witness {idx} e|d*d*d != e|b*b*b")
            if expected != kernel_integer(b, e):
                out.fail("KERNEL3_IDENTITY", f"witness {idx} kernel classifier mismatch")

    audit = cert.get("bounded_audit")
    if not isinstance(audit, dict):
        out.fail("BOUNDED_AUDIT", "bounded_audit must be an object")
    else:
        b_max = audit.get("b_max")
        e_max = audit.get("e_max")
        if not _is_pos_int(b_max) or not _is_pos_int(e_max):
            out.fail("BOUNDED_AUDIT", "b_max/e_max must be positive integers")
        else:
            recomputed = audit_window(b_max, e_max)
            for key, value in recomputed.items():
                if audit.get(key) != value:
                    out.fail("BOUNDED_AUDIT", f"{key} does not recompute")
            if recomputed["reduction_mismatch_count"] != 0:
                out.fail("REDUCTION_IDENTITY", "bounded audit found reduction mismatches")
            if recomputed["kernel_mismatch_count"] != 0:
                out.fail("KERNEL3_IDENTITY", "bounded audit found kernel mismatches")

    orbit_context = cert.get("orbit_context")
    if not isinstance(orbit_context, dict):
        out.fail("ORBIT_CONTEXT", "orbit_context must be an object")
    else:
        if orbit_context.get("included_as_theorem") is True:
            out.fail("ORBIT_OVERCLAIM", "orbit context cannot be included as theorem")
        for key in ("qa_orbit_family9_lift", "e_only_lift", "b_only_lift"):
            if not isinstance(orbit_context.get(key), (int, float)):
                out.fail("ORBIT_CONTEXT", f"orbit_context.{key} must be numeric")

    non_claims = cert.get("non_claims")
    if not isinstance(non_claims, list) or "conic_geometry_explanation" not in non_claims:
        out.fail("NON_CLAIMS", "non_claims must include conic_geometry_explanation")

    return reconcile(cert, out)


def reconcile(cert: dict[str, Any], out: Out) -> dict[str, Any]:
    declared_result = cert.get("result")
    declared_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {
        item.get("fail_type")
        for item in declared_ledger
        if isinstance(item, dict) and isinstance(item.get("fail_type"), str)
    }
    warnings = list(out.warnings)
    if declared_result == "FAIL":
        for code in sorted(out.detected_fails - declared_fail_types):
            warnings.append(f"detected {code} but fail_ledger does not declare it")
    elif declared_result != "PASS":
        out.fail("BAD_RESULT", "result must be PASS or FAIL")
    ok = declared_result == "PASS" and not out.errors
    return {
        "ok": ok,
        "label": "PASS" if ok else "FAIL",
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": warnings,
        "detected_fails": sorted(out.detected_fails),
    }


def validate_file(path: Path) -> dict[str, Any]:
    return validate_cert(json.loads(path.read_text()))


def self_test() -> dict[str, Any]:
    fixtures = Path(__file__).parent / "fixtures"
    expected = {
        "directrix_pass_divisibility.json": True,
        "directrix_fail_bad_kernel.json": False,
        "directrix_fail_orbit_overclaim.json": False,
    }
    results = []
    ok = True
    for name, expected_ok in expected.items():
        result = validate_file(fixtures / name)
        matched = result["ok"] == expected_ok
        ok = ok and matched
        results.append({
            "fixture": name,
            "ok": matched,
            "validator_ok": result["ok"],
            "detected_fails": result["detected_fails"],
            "errors": result["errors"],
        })
    return {"ok": ok, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(canonical_json(self_test()))
        return
    if args.file:
        print(canonical_json(validate_file(args.file)))
        return
    parser.error("use --self-test or --file")


if __name__ == "__main__":
    main()
