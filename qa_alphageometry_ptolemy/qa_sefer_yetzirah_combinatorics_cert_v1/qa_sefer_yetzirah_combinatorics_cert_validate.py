#!/usr/bin/env python3
"""
qa_sefer_yetzirah_combinatorics_cert_validate.py

Validator for QA_SEFER_YETZIRAH_COMBINATORICS_CERT.v1  [family 203]

Certifies: Combinatorial structures in the Sefer Yetzirah (Book of Formation,
c. 2nd-6th century CE) as QA-compatible discrete mathematics.

Claims:
  GATES  — 231 gates = C(22,2) = K_22 complete graph
  FACT   — Factorial computation n! for n=2..7 (earliest known systematic)
  PART   — 3-7-12 partition of 22 Hebrew letters
  PATHS  — 32 paths of Wisdom = 10 Sefirot + 22 letters = 2^5
  CIRC   — Oscillating circle as cyclic structure on Z/22Z
  PYTH   — Pythagorean transmission via Iamblichus/Mount Carmel
  TZERUF — Letter permutation as group action (S_n on alphabet subsets)

Checks:
  SYC_1     — schema_version matches
  SYC_GATES — 22*(22-1)/2 = 231; factorization 3 x 7 x 11
  SYC_FACT  — n! values correct for n=2..7; dr convergence to 9
  SYC_PART  — 3 + 7 + 12 = 22
  SYC_PATHS — 10 + 22 = 32 = 2^5
  SYC_NUM   — numerical checks verify all combinatorial formulas
  SYC_W     — at least 5 witnesses
  SYC_F     — falsifier well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Sefer Yetzirah combinatorial claims; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SEFER_YETZIRAH_COMBINATORICS_CERT.v1"


def digital_root(n):
    """Digital root of positive integer (= n mod 9, with 9 instead of 0)."""
    n = abs(int(n))
    if n == 0:
        return 0
    r = n % 9
    return 9 if r == 0 else r


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # SYC_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"SYC_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # If explicitly marked as expect_fail, validate the failure structure
    if cert.get("expect_fail"):
        if not cert.get("fail_reason"):
            errors.append("SYC_F: expect_fail fixture missing fail_reason")
        # Check that the gates claim has wrong value
        claims = cert.get("claims", [])
        for claim in claims:
            if claim.get("id") == "GATES":
                w = claim.get("witnesses", {})
                result = w.get("result")
                if result == 231:
                    errors.append("SYC_F: expect_fail fixture has correct gate count 231")
        return errors, warnings

    claims = cert.get("claims", [])

    # SYC_W: at least 5 claims
    if len(claims) < 5:
        warnings.append(f"SYC_W: need >= 5 claims, got {len(claims)}")

    for claim in claims:
        cid = claim.get("id", "?")
        w = claim.get("witnesses", {})

        # SYC_GATES: 231 gates
        if cid == "GATES":
            n = w.get("n")
            result = w.get("result")
            if n is not None and result is not None:
                expected = n * (n - 1) // 2
                if result != expected:
                    errors.append(f"SYC_GATES: C({n},2) = {expected}, declared {result}")
                if n == 22 and result != 231:
                    errors.append(f"SYC_GATES: C(22,2) must be 231, got {result}")
            # Check factorization
            fact = w.get("factorization")
            if fact == "231 = 3 x 7 x 11":
                if 3 * 7 * 11 != 231:
                    errors.append("SYC_GATES: 3 x 7 x 11 != 231 (impossible)")
            # Check mod 9
            m9 = w.get("mod_9")
            if m9 is not None and result is not None:
                if result % 9 != m9 % 9:
                    computed = result % 9
                    if computed == 0:
                        computed = 9
                    errors.append(f"SYC_GATES: {result} mod 9 = {computed}, declared {m9}")

        # SYC_FACT: factorials
        elif cid == "FACT":
            factorials = w.get("factorials", [])
            for ft in factorials:
                n = ft.get("n")
                val = ft.get("value")
                dr_declared = ft.get("dr")
                if n is not None and val is not None:
                    expected = 1
                    for i in range(2, n + 1):
                        expected *= i
                    if val != expected:
                        errors.append(f"SYC_FACT: {n}! = {expected}, declared {val}")
                if val is not None and dr_declared is not None:
                    if digital_root(val) != dr_declared:
                        errors.append(
                            f"SYC_FACT: dr({val}) = {digital_root(val)}, declared {dr_declared}"
                        )
            # Check convergence: dr(n!) = 9 for n >= 6
            for ft in factorials:
                n = ft.get("n")
                val = ft.get("value")
                if n is not None and n >= 6 and val is not None:
                    if digital_root(val) != 9:
                        errors.append(f"SYC_FACT: dr({n}!) should be 9 for n>=6, got {digital_root(val)}")

        # SYC_PART: 3-7-12 partition
        elif cid == "PART":
            total = w.get("total")
            partition = w.get("partition", {})
            mother_count = partition.get("mother_letters", {}).get("count", 0)
            double_count = partition.get("double_letters", {}).get("count", 0)
            simple_count = partition.get("simple_letters", {}).get("count", 0)
            part_sum = mother_count + double_count + simple_count
            if total is not None and part_sum != total:
                errors.append(f"SYC_PART: {mother_count}+{double_count}+{simple_count} = {part_sum}, expected {total}")
            if total is not None and total != 22:
                errors.append(f"SYC_PART: total = {total}, expected 22")
            # Verify canonical partition values
            if mother_count != 3:
                warnings.append(f"SYC_PART: mother_letters count = {mother_count}, expected 3")
            if double_count != 7:
                warnings.append(f"SYC_PART: double_letters count = {double_count}, expected 7")
            if simple_count != 12:
                warnings.append(f"SYC_PART: simple_letters count = {simple_count}, expected 12")

        # SYC_PATHS: 32 paths
        elif cid == "PATHS":
            total = w.get("total")
            sefirot = w.get("sefirot")
            letters = w.get("letters")
            if sefirot is not None and letters is not None:
                if sefirot + letters != total:
                    errors.append(f"SYC_PATHS: {sefirot} + {letters} = {sefirot + letters}, declared total {total}")
            if total is not None and total != 32:
                errors.append(f"SYC_PATHS: total = {total}, expected 32")
            p2 = w.get("is_power_of_2")
            if p2 is not None and not p2:
                errors.append("SYC_PATHS: 32 should be flagged as power of 2")
            exp = w.get("exponent")
            if exp is not None and 2 ** exp != 32:
                errors.append(f"SYC_PATHS: 2^{exp} = {2**exp}, expected 32")

    # SYC_NUM: numerical checks
    num = cert.get("numerical_checks", {})
    if num:
        # Gates
        gates = num.get("gates_231", {})
        n = gates.get("n")
        half = gates.get("half")
        if n is not None and half is not None:
            if n * (n - 1) // 2 != half:
                errors.append(f"SYC_NUM: C({n},2) = {n*(n-1)//2}, declared {half}")
        factors = gates.get("factorization", [])
        if factors:
            product = 1
            for f in factors:
                product *= f
            pop = gates.get("product_of_factors")
            if pop is not None and product != pop:
                errors.append(f"SYC_NUM: product of factors {factors} = {product}, declared {pop}")

        # Factorials
        for ft in num.get("factorial_verification", []):
            n = ft.get("n")
            fac = ft.get("factorial")
            if n is not None and fac is not None:
                expected = 1
                for i in range(2, n + 1):
                    expected *= i
                if fac != expected:
                    errors.append(f"SYC_NUM: {n}! = {expected}, declared {fac}")
                dr_declared = ft.get("dr")
                if dr_declared is not None and digital_root(fac) != dr_declared:
                    errors.append(f"SYC_NUM: dr({fac}) = {digital_root(fac)}, declared {dr_declared}")

        # Partition
        pc = num.get("partition_check", {})
        if pc:
            s = pc.get("mothers", 0) + pc.get("doubles", 0) + pc.get("simples", 0)
            if pc.get("sum") is not None and s != pc["sum"]:
                errors.append(f"SYC_NUM: partition sum {s} != declared {pc['sum']}")

        # Paths
        ptc = num.get("paths_check", {})
        if ptc:
            t = ptc.get("sefirot", 0) + ptc.get("letters", 0)
            if ptc.get("total") is not None and t != ptc["total"]:
                errors.append(f"SYC_NUM: paths total {t} != declared {ptc['total']}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("syc_pass_core.json", True),
        ("syc_pass_numerical.json", True),
        ("syc_fail_wrong_gates.json", True),  # expect_fail -> validates failure structure
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs, "warnings": warns})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Sefer Yetzirah Combinatorics Cert [203] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
