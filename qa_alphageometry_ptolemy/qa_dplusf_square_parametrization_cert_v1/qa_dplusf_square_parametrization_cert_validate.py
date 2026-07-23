#!/usr/bin/env python3
"""QA D_plus_F Square Parametrization Cert validator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_DPLUSF_SQUARE_PARAMETRIZATION_CERT.v1"
CERT_TYPE = "qa_dplusf_square_parametrization_cert"
THEOREM_STATUS = "PROVEN_BY_RATIONAL_CONIC_PARAMETRIZATION"
REQUIRED_OBLIGATIONS = {
    "qa_reduction_D_plus_F",
    "rational_conic_parametrization",
    "positive_branch",
    "forward_identity",
    "bounded_audit",
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


def dplusf(b: int, e: int) -> int:
    d = b + e
    a = b + 2 * e
    return d * d + a * b


def param_row(t: int, m: int, n: int) -> tuple[int, int, int]:
    b = t * 2 * m * n
    e = t * (m * m - 4 * m * n + 2 * n * n)
    k_raw = m * m - 2 * n * n
    k = t * abs(k_raw)
    return b, e, k


def brute_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            value = dplusf(b, e)
            root = math.isqrt(value)
            if root * root == value:
                out[(b, e)] = root
    return out


def param_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for m in range(1, b_max + 1):
        for n in range(1, b_max + 1):
            denom = 2 * m * n
            if denom > b_max:
                continue
            for t in range(1, b_max // denom + 1):
                b, e, k = param_row(t, m, n)
                if b < 1 or e < 1 or b > b_max or e > e_max:
                    continue
                if dplusf(b, e) != k * k:
                    continue
                prior = out.get((b, e))
                if prior is None or k < prior:
                    out[(b, e)] = k
    return out


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
    elif statement.get("claims_empirical_orbit_lift_as_theorem") is True:
        out.fail("ORBIT_OVERCLAIM", "empirical orbit lift cannot be claimed inside this theorem cert")

    witnesses = cert.get("witnesses")
    if not isinstance(witnesses, list) or not witnesses:
        out.fail("WITNESSES", "witnesses must be a nonempty list")
    else:
        for idx, row in enumerate(witnesses):
            if not isinstance(row, dict):
                out.fail("WITNESS_ROW", f"witness {idx} must be an object")
                continue
            fields = ("b", "e", "t", "m", "n", "sqrt_D_plus_F")
            if any(not _is_pos_int(row.get(f)) for f in fields):
                out.fail("WITNESS_ROW", f"witness {idx} must contain positive integer {fields}")
                continue
            b, e, k = param_row(row["t"], row["m"], row["n"])
            if b != row["b"] or e != row["e"]:
                out.fail("PARAM_WITNESS", f"witness {idx} does not match parametrization")
            if e <= 0:
                out.fail("POSITIVE_BRANCH", f"witness {idx} has nonpositive e")
            if k != row["sqrt_D_plus_F"]:
                out.fail("SQRT_WITNESS", f"witness {idx} sqrt mismatch")
            if dplusf(row["b"], row["e"]) != row["sqrt_D_plus_F"] * row["sqrt_D_plus_F"]:
                out.fail("DPLUSF_IDENTITY", f"witness {idx} D+F is not declared square")

    audit = cert.get("bounded_audit")
    if not isinstance(audit, dict):
        out.fail("BOUNDED_AUDIT", "bounded_audit must be an object")
    else:
        b_max = audit.get("b_max")
        e_max = audit.get("e_max")
        if not _is_pos_int(b_max) or not _is_pos_int(e_max):
            out.fail("BOUNDED_AUDIT", "b_max/e_max must be positive integers")
        else:
            brute = brute_solutions(b_max, e_max)
            param = param_solutions(b_max, e_max)
            misses = [key for key in brute if key not in param]
            if audit.get("brute_solution_count") != len(brute):
                out.fail("BOUNDED_AUDIT", "brute_solution_count does not recompute")
            if audit.get("param_hit_count") != len(brute) - len(misses):
                out.fail("BOUNDED_AUDIT", "param_hit_count does not recompute")
            if audit.get("miss_count") != len(misses):
                out.fail("BOUNDED_AUDIT", "miss_count does not recompute")
            if misses:
                out.fail("PARAM_EXHAUSTIVENESS", f"bounded audit found misses: {misses[:5]}")

    non_claims = cert.get("non_claims")
    if not isinstance(non_claims, list) or "prime_prediction_or_factorization_shortcut" not in non_claims:
        out.fail("NON_CLAIMS", "non_claims must include prime_prediction_or_factorization_shortcut")

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
        "dpf_pass_parametrization.json": True,
        "dpf_fail_bad_witness.json": False,
        "dpf_fail_orbit_overclaim.json": False,
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
