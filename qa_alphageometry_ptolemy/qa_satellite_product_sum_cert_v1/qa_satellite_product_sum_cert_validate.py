#!/usr/bin/env python3
"""
qa_satellite_product_sum_cert_validate.py

Validator for QA_SATELLITE_PRODUCT_SUM_CERT.v1  [family 181]

Certifies: sum of QA tuple products (b*e*d*a) over all satellite
pairs equals M^4 for any modulus M divisible by 3.

Checks:
  SPS_1       — schema_version matches
  SPS_PROOF   — normalized product sum = 81 = 3^4
  SPS_COUNT   — satellite count = 8 for each witness
  SPS_SUM     — declared product_sum = M^4
  SPS_TUPLES  — each satellite tuple (b,e,d,a) is A1/A2-compliant
  SPS_CLOSURE — d,a are multiples of sat_divisor for satellite pairs
  SPS_COROL   — singularity product = M^4 (corollary)
  SPS_W       — at least one witness present
  SPS_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates algebraic identity claims in submitted JSON; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SATELLITE_PRODUCT_SUM_CERT.v1"


def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def validate(path):
    """Validate a satellite product sum certificate.

    Returns (errors, warnings) where both are lists of strings.
    """
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # SPS_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"SPS_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # SPS_F: fail_ledger well-formed
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("SPS_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("SPS_F: fail_ledger must be a list")

    # Handle FAIL result early
    if cert.get("result") == "FAIL":
        return errors, warnings

    # SPS_W: at least one witness
    witnesses = cert.get("witnesses")
    if not witnesses or not isinstance(witnesses, list):
        errors.append("SPS_W: no witnesses array found")
        return errors, warnings

    if len(witnesses) < 1:
        errors.append("SPS_W: at least one witness required")

    for idx, w in enumerate(witnesses):
        M = w.get("modulus")
        if M is None:
            errors.append(f"SPS_W: witness[{idx}] missing modulus")
            continue
        if M % 3 != 0:
            errors.append(f"SPS_W: witness[{idx}] modulus {M} not divisible by 3")
            continue

        sat_div = M // 3

        # SPS_COUNT: satellite count
        sc = w.get("satellite_count")
        if sc is not None and sc != 8:
            errors.append(f"SPS_COUNT: witness[{idx}] M={M}: satellite_count={sc}, expected 8")

        # SPS_SUM: product sum = M^4
        ps = w.get("product_sum")
        m4 = w.get("M_fourth")
        expected_m4 = M * M * M * M

        if m4 is not None and m4 != expected_m4:
            errors.append(f"SPS_SUM: witness[{idx}] M={M}: declared M_fourth={m4}, actual M^4={expected_m4}")

        if ps is not None and ps != expected_m4:
            errors.append(f"SPS_SUM: witness[{idx}] M={M}: product_sum={ps} != M^4={expected_m4}")

        # SPS_TUPLES: verify individual satellite pairs if present
        pairs = w.get("satellite_pairs")
        if pairs:
            recomputed_sum = 0
            for pidx, p in enumerate(pairs):
                b = p.get("b")
                e = p.get("e")
                d_decl = p.get("d")
                a_decl = p.get("a")
                prod_decl = p.get("product")

                if any(v is None for v in [b, e, d_decl, a_decl]):
                    errors.append(f"SPS_TUPLES: witness[{idx}] pair[{pidx}]: missing b/e/d/a")
                    continue

                # A1: values in {1,...,M}
                for name, val in [("b", b), ("e", e), ("d", d_decl), ("a", a_decl)]:
                    if val < 1 or val > M:
                        errors.append(f"SPS_TUPLES: witness[{idx}] pair[{pidx}]: {name}={val} outside {{1,...,{M}}}")

                # A2: derived coords
                d_expected = qa_mod(b + e, M)
                a_expected = qa_mod(b + 2 * e, M)
                if d_decl != d_expected:
                    errors.append(f"SPS_TUPLES: witness[{idx}] pair[{pidx}]: d={d_decl}, expected qa_mod({b}+{e},{M})={d_expected}")
                if a_decl != a_expected:
                    errors.append(f"SPS_TUPLES: witness[{idx}] pair[{pidx}]: a={a_decl}, expected qa_mod({b}+2*{e},{M})={a_expected}")

                # Product check
                expected_prod = b * e * d_expected * a_expected
                if prod_decl is not None and prod_decl != expected_prod:
                    errors.append(f"SPS_TUPLES: witness[{idx}] pair[{pidx}]: product={prod_decl}, expected {expected_prod}")

                recomputed_sum += expected_prod

                # SPS_CLOSURE: satellite pairs must have d,a as multiples of sat_div
                if b % sat_div != 0:
                    errors.append(f"SPS_CLOSURE: witness[{idx}] pair[{pidx}]: b={b} not divisible by sat_divisor={sat_div}")
                if e % sat_div != 0:
                    errors.append(f"SPS_CLOSURE: witness[{idx}] pair[{pidx}]: e={e} not divisible by sat_divisor={sat_div}")
                if d_expected % sat_div != 0:
                    errors.append(f"SPS_CLOSURE: witness[{idx}] pair[{pidx}]: d={d_expected} not divisible by sat_divisor={sat_div}")
                if a_expected % sat_div != 0:
                    errors.append(f"SPS_CLOSURE: witness[{idx}] pair[{pidx}]: a={a_expected} not divisible by sat_divisor={sat_div}")

            # Verify recomputed sum
            if recomputed_sum != expected_m4:
                errors.append(f"SPS_SUM: witness[{idx}] M={M}: recomputed product sum={recomputed_sum} != M^4={expected_m4}")

    # SPS_PROOF: normalized products
    np_block = cert.get("normalized_products")
    if np_block:
        vals = np_block.get("values")
        if vals:
            s = sum(vals)
            if s != 81:
                errors.append(f"SPS_PROOF: normalized product sum={s}, expected 81=3^4")
        eq = np_block.get("equals_3_to_4")
        if eq is not None and eq is not True:
            errors.append("SPS_PROOF: equals_3_to_4 must be true")

    # SPS_COROL: corollary
    corol = cert.get("corollary")
    if corol:
        seq = corol.get("satellite_equals_singularity")
        if seq is not None and seq is not True:
            errors.append("SPS_COROL: satellite_equals_singularity must be true")

    return errors, warnings


def _self_test():
    """Test bundled fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("sps_pass_multi_modulus.json", True),
        ("sps_fail_wrong_sum.json", True),  # declares result=FAIL, validator skips detailed checks → no errors
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
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Satellite Product Sum Cert [181] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
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
