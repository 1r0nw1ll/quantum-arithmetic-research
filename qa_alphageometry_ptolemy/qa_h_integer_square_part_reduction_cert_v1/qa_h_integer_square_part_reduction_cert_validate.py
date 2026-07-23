#!/usr/bin/env python3
"""QA h_integer Square-Part Reduction Cert validator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_H_INTEGER_SQUARE_PART_REDUCTION_CERT.v1"
CERT_TYPE = "qa_h_integer_square_part_reduction_cert"
THEOREM_STATUS = "PROVEN_STRUCTURAL_SQUARE_PART_REDUCTION"
REQUIRED_OBLIGATIONS = {
    "qa_reduction_h_to_F_square",
    "gcd_square_part_decomposition",
    "coprime_product_square",
    "bounded_audit",
    "parametrization_strength_boundary",
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


def is_square(n: int) -> bool:
    root = math.isqrt(n)
    return root * root == n


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    F = a * b
    return {"b": b, "e": e, "d": d, "a": a, "F": F}


def h_integer(b: int, e: int) -> bool:
    return is_square(qa_values(b, e)["F"])


def reduced_square_condition(b: int, e: int) -> bool:
    row = qa_values(b, e)
    divisor = math.gcd(row["a"], row["b"])
    return is_square(row["a"] // divisor) and is_square(row["b"] // divisor)


def generated_witness(g: int, r: int, s: int) -> tuple[int, int, int, int, int] | None:
    b = g * r * r
    a = g * s * s
    numerator = a - b
    if numerator <= 0 or numerator % 2 != 0:
        return None
    e = numerator // 2
    d = b + e
    sqrt_F = g * r * s
    return b, e, d, a, sqrt_F


def audit_window(b_max: int, e_max: int) -> dict[str, int]:
    support = 0
    reduction_mismatches = 0
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            target = h_integer(b, e)
            reduced = reduced_square_condition(b, e)
            if target:
                support += 1
            if target != reduced:
                reduction_mismatches += 1
    return {
        "pair_count": b_max * e_max,
        "support": support,
        "reduction_mismatch_count": reduction_mismatches,
    }


def _is_pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


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
    else:
        if statement.get("claims_complete_geometry_parametrization") is True:
            out.fail("PARAMETRIZATION_OVERCLAIM", "h_integer cert is a structural reduction, not a complete geometry parametrization theorem")

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
            values = qa_values(b, e)
            divisor = math.gcd(values["a"], values["b"])
            a_reduced = values["a"] // divisor
            b_reduced = values["b"] // divisor
            if row.get("d") != values["d"]:
                out.fail("D_WITNESS", f"witness {idx} d mismatch")
            if row.get("a") != values["a"]:
                out.fail("A_WITNESS", f"witness {idx} a mismatch")
            if row.get("F") != values["F"]:
                out.fail("F_WITNESS", f"witness {idx} F mismatch")
            if row.get("gcd_a_b") != divisor:
                out.fail("GCD_WITNESS", f"witness {idx} gcd(a,b) mismatch")
            if row.get("a_reduced") != a_reduced or row.get("b_reduced") != b_reduced:
                out.fail("REDUCED_PART_WITNESS", f"witness {idx} reduced square parts mismatch")
            expected = h_integer(b, e)
            if row.get("h_integer") is not expected:
                out.fail("H_INTEGER_WITNESS", f"witness {idx} h_integer truth mismatch")
            if expected != reduced_square_condition(b, e):
                out.fail("SQUARE_PART_IDENTITY", f"witness {idx} square-part reduction mismatch")
            generated = row.get("generated_parameters")
            if generated is not None:
                if not isinstance(generated, dict):
                    out.fail("GENERATED_WITNESS", f"witness {idx} generated_parameters must be object")
                elif any(not _is_pos_int(generated.get(f)) for f in ("g", "r", "s")):
                    out.fail("GENERATED_WITNESS", f"witness {idx} generated g,r,s must be positive integers")
                else:
                    gen = generated_witness(generated["g"], generated["r"], generated["s"])
                    if gen is None:
                        out.fail("GENERATED_WITNESS", f"witness {idx} generated parameters do not yield positive integer e")
                    elif gen != (b, e, values["d"], values["a"], row.get("sqrt_F")):
                        out.fail("GENERATED_WITNESS", f"witness {idx} generated parameters mismatch")

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
                out.fail("SQUARE_PART_IDENTITY", "bounded audit found reduction mismatches")

    non_claims = cert.get("non_claims")
    if not isinstance(non_claims, list) or "complete_geometry_parametrization" not in non_claims:
        out.fail("NON_CLAIMS", "non_claims must include complete_geometry_parametrization")

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
        "hint_pass_reduction.json": True,
        "hint_fail_bad_reduced_parts.json": False,
        "hint_fail_parametrization_overclaim.json": False,
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
