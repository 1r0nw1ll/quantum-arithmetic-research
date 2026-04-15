#!/usr/bin/env python3
"""
qa_g_equivariant_cnn_structural_cert_v1.py  [family 247]
"""

QA_COMPLIANCE = "cert_validator - pure-algebra G-equivariant CNN structural cert; integer state space; closed-form residue arithmetic; no float state; no gradients; no non-stdlib imports"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_G_EQUIVARIANT_CNN_STRUCTURAL_CERT.v1"

EXPECTED_C4_TABLE = [
    {
        "paper_equation": "Eq. 10",
        "paper_role": "lifting",
        "qa_role": "observer IN",
        "note": "first-layer image -> group-indexed feature map",
    },
    {
        "paper_equation": "Eq. 11",
        "paper_role": "G-correlation",
        "qa_role": "QA-layer resonance",
        "note": "group-indexed coupling on the discrete rotation index",
    },
    {
        "paper_equation": "§6.3",
        "paper_role": "coset pooling",
        "qa_role": "observer OUT",
        "note": "quotient projection to an invariant output",
    },
]

EXPECTED_C3_FAMILY_COUNTS = {
    "singularity": 9,
    "satellite": 18,
    "cosmos": 54,
}

EXPECTED_C3_PERIOD_COUNTS = {
    "1": 9,
    "3": 18,
    "9": 54,
}


def qa_step(b1, b2, n):
    return ((b1 + b2 - 1) % n) + 1


def phi_from_rule(rule, b, n):
    if rule == "mod_n_residue":
        return b % n
    if rule == "minus_one_no_wrap":
        return b - 1
    raise ValueError(f"unsupported phi_rule: {rule!r}")


def phi_inverse_from_rule(rule, r, n):
    if rule == "mod_n_residue":
        return n if r == 0 else r
    if rule == "minus_one_no_wrap":
        return r + 1
    raise ValueError(f"unsupported phi_rule: {rule!r}")


def _orbit_period_1d(b_gen, n):
    step = (b_gen - 1) % n
    if step == 0:
        return 1
    return n // gcd(n, step)


def _orbit_family_from_period(period):
    if period == 1:
        return "singularity"
    if period == 3:
        return "satellite"
    return "cosmos"


def check_c1_bijection(fixture):
    n = int(fixture.get("n", 0))
    rule = fixture.get("phi_rule", "")
    images = [phi_from_rule(rule, b, n) for b in range(1, n + 1)]
    expected_images = list(range(0, n))
    image_set = sorted(set(images))
    inverse_ok = all(
        phi_from_rule(rule, phi_inverse_from_rule(rule, r, n), n) == r
        for r in range(0, n)
    )
    roundtrip_ok = all(
        phi_inverse_from_rule(rule, phi_from_rule(rule, b, n), n) == b
        for b in range(1, n + 1)
    )
    declared_inverse = fixture.get("phi_inverse_rule", "")
    declared_inverse_ok = declared_inverse in {"n_if_zero_else_r", "r_plus_1"}
    return {
        "passed": (
            n > 0
            and len(images) == n
            and image_set == expected_images
            and inverse_ok
            and roundtrip_ok
            and declared_inverse_ok
        ),
        "n": n,
        "image_set": image_set,
        "expected_images": expected_images,
        "roundtrip_ok": roundtrip_ok,
        "inverse_ok": inverse_ok,
    }


def check_c2_composition(fixture):
    n = int(fixture.get("n", 0))
    rule = fixture.get("phi_rule", "")
    pair_count = 0
    exceptions = []
    for b1 in range(1, n + 1):
        for b2 in range(1, n + 1):
            pair_count += 1
            lhs = phi_from_rule(rule, qa_step(b1, b2, n), n)
            rhs = (phi_from_rule(rule, b1, n) + phi_from_rule(rule, b2, n)) % n
            if lhs != rhs:
                exceptions.append(
                    {
                        "b1": b1,
                        "b2": b2,
                        "lhs": lhs,
                        "rhs": rhs,
                    }
                )
    declared = fixture.get("composition", {})
    declared_pair_count = declared.get("pair_count")
    declared_exception_count = declared.get("exception_count")
    declared_witnesses = declared.get("witnesses", [])
    witness_ok = True
    for witness in declared_witnesses:
        b1 = int(witness.get("b1", 0))
        b2 = int(witness.get("b2", 0))
        lhs = phi_from_rule(rule, qa_step(b1, b2, n), n)
        rhs = (phi_from_rule(rule, b1, n) + phi_from_rule(rule, b2, n)) % n
        if witness.get("lhs") != lhs or witness.get("rhs") != rhs:
            witness_ok = False
            break
    return {
        "passed": (
            pair_count == n * n
            and declared_pair_count == pair_count
            and declared_exception_count == len(exceptions)
            and witness_ok
            and len(exceptions) == 0
        ),
        "pair_count": pair_count,
        "exception_count": len(exceptions),
        "first_exception": exceptions[0] if exceptions else None,
        "witness_ok": witness_ok,
    }


def check_c3_orbit_partition(fixture):
    n = int(fixture.get("n", 0))
    if n != 9:
        return {
            "passed": True,
            "skipped": True,
            "reason": "C3 is only asserted for n=9",
        }

    family_counts = {"singularity": 0, "satellite": 0, "cosmos": 0}
    period_counts = {}
    for b0 in range(1, n + 1):
        for b_gen in range(1, n + 1):
            period = _orbit_period_1d(b_gen, n)
            fam = _orbit_family_from_period(period)
            family_counts[fam] += 1
            period_key = str(period)
            period_counts[period_key] = period_counts.get(period_key, 0) + 1

    declared = fixture.get("orbit_partition", {})
    declared_family_counts = declared.get("family_counts", {})
    declared_period_counts = declared.get("period_counts", {})
    declared_exception_count = declared.get("exception_count")
    witnesses = declared.get("witnesses", [])
    witness_ok = True
    for witness in witnesses:
        b_gen = int(witness.get("b_gen", 0))
        period = _orbit_period_1d(b_gen, n)
        fam = _orbit_family_from_period(period)
        if witness.get("orbit_length") != period or witness.get("family") != fam:
            witness_ok = False
            break

    return {
        "passed": (
            family_counts == EXPECTED_C3_FAMILY_COUNTS
            and period_counts == EXPECTED_C3_PERIOD_COUNTS
            and declared_family_counts == family_counts
            and declared_period_counts == period_counts
            and declared_exception_count == 0
            and witness_ok
        ),
        "family_counts": family_counts,
        "period_counts": period_counts,
        "exception_count": 0,
        "witness_ok": witness_ok,
    }


def check_c4_correspondence(fixture):
    table = fixture.get("equation_correspondence", [])
    return {
        "passed": table == EXPECTED_C4_TABLE,
        "table_len": len(table) if isinstance(table, list) else None,
    }


def _validate_checks(fixture):
    checks = {}
    checks["SCHEMA"] = fixture.get("schema_version") == SCHEMA_VERSION
    c1 = check_c1_bijection(fixture)
    c2 = check_c2_composition(fixture)
    c3 = check_c3_orbit_partition(fixture)
    c4 = check_c4_correspondence(fixture)
    checks["C1"] = c1["passed"]
    checks["C2"] = c2["passed"]
    checks["C3"] = c3["passed"]
    checks["C4"] = c4["passed"]
    src = str(fixture.get("source_attribution", ""))
    checks["SRC"] = ("Cohen" in src) and ("Welling" in src) and ("1602.07576" in src)
    checks["F"] = isinstance(fixture.get("fail_ledger"), list)
    return checks, {"C1": c1, "C2": c2, "C3": c3, "C4": c4}


def validate_fixture(path):
    with open(path, encoding="utf-8") as f:
        fixture = json.load(f)
    checks, details = _validate_checks(fixture)
    expected = fixture.get("result", "PASS")
    actual = "PASS" if all(checks.values()) else "FAIL"
    return {
        "ok": actual == expected,
        "expected": expected,
        "actual": actual,
        "checks": checks,
        "details": details,
    }


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {fp.name: validate_fixture(fp) for fp in sorted(fdir.glob("*.json"))}
    ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": ok, "checks": results}, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    if len(sys.argv) > 1:
        result = validate_fixture(sys.argv[1])
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    print("Usage: python qa_g_equivariant_cnn_structural_cert_v1.py [--self-test | fixture.json]")
    sys.exit(1)
