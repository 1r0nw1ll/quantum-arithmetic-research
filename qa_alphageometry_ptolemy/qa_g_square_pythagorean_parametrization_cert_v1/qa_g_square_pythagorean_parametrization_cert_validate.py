#!/usr/bin/env python3
"""QA G Square Pythagorean Parametrization Cert validator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_G_SQUARE_PYTHAGOREAN_PARAMETRIZATION_CERT.v1"
CERT_TYPE = "qa_g_square_pythagorean_parametrization_cert"
THEOREM_STATUS = "PROVEN_BY_EUCLID_PYTHAGOREAN_PARAMETRIZATION"
REQUIRED_OBLIGATIONS = {
    "qa_reduction_G",
    "euclid_pythagorean_parametrization",
    "branch_filter_d_gt_e",
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


def g_value(b: int, e: int) -> int:
    d = b + e
    return d * d + e * e


def pythagorean_witness(t: int, m: int, n: int, branch: str) -> tuple[int, int, int, int]:
    odd_leg = t * (m * m - n * n)
    even_leg = t * 2 * m * n
    k = t * (m * m + n * n)
    if branch == "odd_as_d":
        d = odd_leg
        e = even_leg
    elif branch == "even_as_d":
        d = even_leg
        e = odd_leg
    else:
        raise ValueError("branch must be odd_as_d or even_as_d")
    b = d - e
    return b, e, d, k


def brute_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for b in range(1, b_max + 1):
        for e in range(1, e_max + 1):
            value = g_value(b, e)
            root = math.isqrt(value)
            if root * root == value:
                out[(b, e)] = root
    return out


def param_solutions(b_max: int, e_max: int) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    d_max = b_max + e_max
    for m in range(2, d_max + 1):
        for n in range(1, m):
            odd_leg = m * m - n * n
            even_leg = 2 * m * n
            if max(odd_leg, even_leg) > d_max:
                continue
            for t in range(1, d_max // max(odd_leg, even_leg) + 1):
                for branch in ("odd_as_d", "even_as_d"):
                    b, e, _d, k = pythagorean_witness(t, m, n, branch)
                    if b < 1 or e < 1 or b > b_max or e > e_max:
                        continue
                    if g_value(b, e) != k * k:
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
            fields = ("b", "e", "d", "t", "m", "n", "sqrt_G")
            if any(not _is_pos_int(row.get(f)) for f in fields):
                out.fail("WITNESS_ROW", f"witness {idx} must contain positive integer {fields}")
                continue
            branch = row.get("branch")
            if branch not in {"odd_as_d", "even_as_d"}:
                out.fail("BRANCH_WITNESS", f"witness {idx} branch must be odd_as_d or even_as_d")
                continue
            b, e, d, k = pythagorean_witness(row["t"], row["m"], row["n"], branch)
            if b != row["b"] or e != row["e"] or d != row["d"]:
                out.fail("PARAM_WITNESS", f"witness {idx} does not match Euclid parametrization")
            if d <= e:
                out.fail("BRANCH_FILTER", f"witness {idx} does not satisfy d>e")
            if k != row["sqrt_G"]:
                out.fail("SQRT_WITNESS", f"witness {idx} sqrt mismatch")
            if g_value(row["b"], row["e"]) != row["sqrt_G"] * row["sqrt_G"]:
                out.fail("G_IDENTITY", f"witness {idx} G is not declared square")

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
        "gsq_pass_parametrization.json": True,
        "gsq_fail_bad_witness.json": False,
        "gsq_fail_orbit_overclaim.json": False,
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
