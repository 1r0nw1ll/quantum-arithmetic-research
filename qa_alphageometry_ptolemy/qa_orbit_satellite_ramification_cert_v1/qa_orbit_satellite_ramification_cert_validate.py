#!/usr/bin/env python3
"""QA Orbit Satellite Ramification Cert validator.

Primary source: Wall, D.D. (1960). Fibonacci Series Modulo m. American
Mathematical Monthly, 67(6), 525-532. Ireland, K. & Rosen, M. (1990).
A Classical Introduction to Modern Number Theory, 2nd ed., Springer,
ISBN 978-0-387-97329-6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_ORBIT_SATELLITE_RAMIFICATION_CERT.v1"
CERT_TYPE = "qa_orbit_satellite_ramification_cert"
THEOREM_STATUS = "PROVEN_BY_DISCRIMINANT_5_RAMIFICATION"
REQUIRED_OBLIGATIONS = {
    "qa_step_fibonacci_conjugacy",
    "mod3_irreducible_period8",
    "mod5_jordan_eigenspace_period4",
    "crt_lcm_composition",
    "shortcut_miss_count_bounded_audit",
    "known_moduli_shortcut_exact",
}
TESTED_MODULI = (15, 30, 45, 60, 75)
KNOWN_MODULI = (9, 24)
MOD5_EIGENSPACE = frozenset({(1, 3), (2, 1), (3, 4), (4, 2)})


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


def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    return e, ((b + e - 1) % m) + 1


def orbit_period(b: int, e: int, m: int) -> int:
    cb, ce = b, e
    seen: set[tuple[int, int]] = set()
    for _ in range(m * m + 1):
        if (cb, ce) in seen:
            break
        seen.add((cb, ce))
        cb, ce = qa_step(cb, ce, m)
    return len(seen)


def orbit_family(b: int, e: int, m: int) -> str:
    period = orbit_period(b, e, m)
    if period == 1:
        return "singularity"
    if period == 8:
        return "satellite"
    return "cosmos"


def orbit_family_divisor_shortcut(b: int, e: int, m: int) -> str:
    sat_divisor = m // 3
    if b == m and e == m:
        return "singularity"
    if sat_divisor > 0 and b % sat_divisor == 0 and e % sat_divisor == 0:
        return "satellite"
    return "cosmos"


def prime_power_part(m: int, p: int) -> int:
    part = 1
    while m % p == 0:
        part *= p
        m //= p
    return part


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
        "mod3_witnesses",
        "mod5_witnesses",
        "crt_examples",
        "bounded_audit",
        "known_moduli_check",
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
        if statement.get("claims_empirical_orbit_lift_as_theorem") is True:
            out.fail("ORBIT_OVERCLAIM", "empirical orbit lift cannot be claimed inside this theorem cert")
        if statement.get("claims_general_hensel_lifting_proof") is True:
            out.fail(
                "HENSEL_GENERALIZATION_OVERCLAIM",
                "arbitrary-exponent prime-power lifting is verified computationally here, "
                "not proven by a general p-adic/Hensel argument",
            )
        if statement.get("claims_replaces_canonical_orbit_period") is True:
            out.fail(
                "CLASSIFIER_REPLACEMENT_OVERCLAIM",
                "this cert explains orbit_period/orbit_family, it does not replace the "
                "canonical simulation-based classifier",
            )

    # qa_step_fibonacci_conjugacy: qa_step(b,e,m) == (y, x+y mod m) with 0 relabeled m.
    conj_rows = cert.get("qa_step_fibonacci_conjugacy_samples")
    if not isinstance(conj_rows, list) or not conj_rows:
        out.fail("FIBONACCI_CONJUGACY", "qa_step_fibonacci_conjugacy_samples must be a nonempty list")
    else:
        for idx, row in enumerate(conj_rows):
            if not isinstance(row, dict):
                out.fail("FIBONACCI_CONJUGACY", f"sample {idx} must be an object")
                continue
            b, e, m = row.get("b"), row.get("e"), row.get("m")
            if not (_is_pos_int(b) and _is_pos_int(e) and _is_pos_int(m)) or not (1 <= b <= m and 1 <= e <= m):
                out.fail("FIBONACCI_CONJUGACY", f"sample {idx} must have 1<=b,e<=m")
                continue
            nb, ne = qa_step(b, e, m)
            x, y = b % m, e % m
            nx, ny = y, (x + y) % m
            nb2 = m if nx == 0 else nx
            ne2 = m if ny == 0 else ny
            if (nb, ne) != (nb2, ne2):
                out.fail("FIBONACCI_CONJUGACY", f"sample {idx} qa_step does not match Fibonacci-matrix conjugacy")

    # mod3_witnesses: every declared (b,e,m=3) matches recomputed orbit_period; and the
    # full 3x3 grid is exactly {1 fixed point, 8 period-8 points}.
    mod3 = cert.get("mod3_witnesses")
    if not isinstance(mod3, list) or not mod3:
        out.fail("MOD3_WITNESS", "mod3_witnesses must be a nonempty list")
    else:
        for idx, row in enumerate(mod3):
            if not isinstance(row, dict):
                out.fail("MOD3_WITNESS", f"mod3 witness {idx} must be an object")
                continue
            b, e, period = row.get("b"), row.get("e"), row.get("period")
            if not (_is_pos_int(b) and _is_pos_int(e) and 1 <= b <= 3 and 1 <= e <= 3):
                out.fail("MOD3_WITNESS", f"mod3 witness {idx} must have 1<=b,e<=3")
                continue
            if orbit_period(b, e, 3) != period:
                out.fail("MOD3_WITNESS", f"mod3 witness {idx} period does not recompute")
    period_counts_3 = {}
    for b in range(1, 4):
        for e in range(1, 4):
            p = orbit_period(b, e, 3)
            period_counts_3[p] = period_counts_3.get(p, 0) + 1
    if period_counts_3 != {1: 1, 8: 8}:
        out.fail("MOD3_IRREDUCIBLE_PERIOD8", f"expected {{1:1,8:8}} over (Z/3Z)^2, got {period_counts_3}")

    # mod5_witnesses: every declared (b,e,m=5) matches recomputed orbit_period; and the
    # full 5x5 grid is exactly {1 fixed point, 4 eigenspace points (period 4), 20 (period 20)}.
    mod5 = cert.get("mod5_witnesses")
    if not isinstance(mod5, list) or not mod5:
        out.fail("MOD5_WITNESS", "mod5_witnesses must be a nonempty list")
    else:
        for idx, row in enumerate(mod5):
            if not isinstance(row, dict):
                out.fail("MOD5_WITNESS", f"mod5 witness {idx} must be an object")
                continue
            b, e, period = row.get("b"), row.get("e"), row.get("period")
            if not (_is_pos_int(b) and _is_pos_int(e) and 1 <= b <= 5 and 1 <= e <= 5):
                out.fail("MOD5_WITNESS", f"mod5 witness {idx} must have 1<=b,e<=5")
                continue
            if orbit_period(b, e, 5) != period:
                out.fail("MOD5_WITNESS", f"mod5 witness {idx} period does not recompute")
    period_counts_5 = {}
    eigenspace_periods = set()
    for b in range(1, 6):
        for e in range(1, 6):
            p = orbit_period(b, e, 5)
            period_counts_5[p] = period_counts_5.get(p, 0) + 1
            if (b % 5, e % 5) in MOD5_EIGENSPACE:
                eigenspace_periods.add(p)
    if period_counts_5 != {1: 1, 4: 4, 20: 20}:
        out.fail("MOD5_JORDAN_EIGENSPACE_PERIOD4", f"expected {{1:1,4:4,20:20}} over (Z/5Z)^2, got {period_counts_5}")
    if eigenspace_periods != {4}:
        out.fail("MOD5_JORDAN_EIGENSPACE_PERIOD4", f"declared eigenspace must have period exactly 4, got {eigenspace_periods}")

    # crt_examples: illustrative shortcut-missed satellites, each independently recomputed
    # and each explained by (mod-3-part period 8) AND (mod-5-part period 4).
    crt_examples = cert.get("crt_examples")
    if not isinstance(crt_examples, list) or not crt_examples:
        out.fail("CRT_EXAMPLE", "crt_examples must be a nonempty list")
    else:
        for idx, row in enumerate(crt_examples):
            if not isinstance(row, dict):
                out.fail("CRT_EXAMPLE", f"crt example {idx} must be an object")
                continue
            m, b, e = row.get("m"), row.get("b"), row.get("e")
            if not (_is_pos_int(m) and _is_pos_int(b) and _is_pos_int(e) and 1 <= b <= m and 1 <= e <= m):
                out.fail("CRT_EXAMPLE", f"crt example {idx} must have 1<=b,e<=m")
                continue
            canonical = orbit_family(b, e, m)
            shortcut = orbit_family_divisor_shortcut(b, e, m)
            if canonical != "satellite" or shortcut == "satellite":
                out.fail("CRT_EXAMPLE", f"crt example {idx} is not a shortcut-missed satellite")
                continue
            p3 = prime_power_part(m, 3)
            p5 = prime_power_part(m, 5)
            b3 = b % p3 or p3
            e3 = e % p3 or p3
            b5 = b % p5 or p5
            e5 = e % p5 or p5
            per3 = orbit_period(b3, e3, p3)
            per5 = orbit_period(b5, e5, p5)
            if per3 != 8 or per5 != 4:
                out.fail("CRT_EXAMPLE", f"crt example {idx} does not match (mod3-part period8, mod5-part period4)")

    # bounded_audit: full recomputation of the shortcut miss count for every tested modulus,
    # plus verification that EVERY missed satellite matches the CRT mechanism (not just count).
    audit = cert.get("bounded_audit")
    if not isinstance(audit, dict):
        out.fail("BOUNDED_AUDIT", "bounded_audit must be an object")
    else:
        per_modulus = audit.get("per_modulus")
        if not isinstance(per_modulus, list) or len(per_modulus) != len(TESTED_MODULI):
            out.fail("BOUNDED_AUDIT", f"per_modulus must list all {len(TESTED_MODULI)} tested moduli")
        else:
            declared_moduli = sorted(row.get("m") for row in per_modulus if isinstance(row, dict))
            if declared_moduli != sorted(TESTED_MODULI):
                out.fail("BOUNDED_AUDIT", f"per_modulus must cover exactly {sorted(TESTED_MODULI)}")
            for row in per_modulus:
                if not isinstance(row, dict):
                    out.fail("BOUNDED_AUDIT", "per_modulus row must be an object")
                    continue
                m = row.get("m")
                if m not in TESTED_MODULI:
                    continue
                misses = []
                total_satellite = 0
                for b in range(1, m + 1):
                    for e in range(1, m + 1):
                        if orbit_family(b, e, m) == "satellite":
                            total_satellite += 1
                            if orbit_family_divisor_shortcut(b, e, m) != "satellite":
                                misses.append((b, e))
                if row.get("satellite_count") != total_satellite:
                    out.fail("BOUNDED_AUDIT", f"m={m} satellite_count does not recompute")
                if row.get("shortcut_miss_count") != len(misses):
                    out.fail("BOUNDED_AUDIT", f"m={m} shortcut_miss_count does not recompute")
                if len(misses) != 32:
                    out.fail("BOUNDED_AUDIT", f"m={m} expected exactly 32 shortcut misses, got {len(misses)}")
                p3 = prime_power_part(m, 3)
                p5 = prime_power_part(m, 5)
                for (b, e) in misses:
                    b3 = b % p3 or p3
                    e3 = e % p3 or p3
                    b5 = b % p5 or p5
                    e5 = e % p5 or p5
                    if orbit_period(b3, e3, p3) != 8 or orbit_period(b5, e5, p5) != 4:
                        out.fail("BOUNDED_AUDIT", f"m={m} a shortcut miss did not match the CRT mechanism")
                        break

    # known_moduli_check: shortcut must be EXACT (0 misses) on m in {9,24} (no factor of 5).
    known = cert.get("known_moduli_check")
    if not isinstance(known, dict):
        out.fail("KNOWN_MODULI_CHECK", "known_moduli_check must be an object")
    else:
        rows = known.get("per_modulus")
        if not isinstance(rows, list) or len(rows) != len(KNOWN_MODULI):
            out.fail("KNOWN_MODULI_CHECK", f"must list all {len(KNOWN_MODULI)} known moduli")
        else:
            for row in rows:
                if not isinstance(row, dict):
                    out.fail("KNOWN_MODULI_CHECK", "per_modulus row must be an object")
                    continue
                m = row.get("m")
                if m not in KNOWN_MODULI:
                    out.fail("KNOWN_MODULI_CHECK", f"unexpected modulus {m}")
                    continue
                misses = sum(
                    1
                    for b in range(1, m + 1)
                    for e in range(1, m + 1)
                    if orbit_family(b, e, m) == "satellite" and orbit_family_divisor_shortcut(b, e, m) != "satellite"
                )
                if row.get("shortcut_miss_count") != misses or misses != 0:
                    out.fail("KNOWN_MODULI_CHECK", f"m={m} shortcut must be exact (0 misses), recomputed {misses}")

    non_claims = cert.get("non_claims")
    required_non_claims = {
        "prime_prediction_or_factorization_shortcut",
        "general_prime_power_hensel_lifting_proof_for_arbitrary_exponents",
        "replacement_for_canonical_orbit_period_simulation",
    }
    if not isinstance(non_claims, list) or not required_non_claims.issubset(set(non_claims)):
        out.fail("NON_CLAIMS", f"non_claims must include {sorted(required_non_claims)}")

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
        "orbit_pass_ramification.json": True,
        "orbit_fail_bad_witness.json": False,
        "orbit_fail_hensel_overclaim.json": False,
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
