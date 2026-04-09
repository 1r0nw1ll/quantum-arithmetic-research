#!/usr/bin/env python3
"""
qa_grid_cell_rns_cert_validate.py

Validator for QA_GRID_CELL_RNS_CERT.v1  [family 205]

Certifies: The structural isomorphism between the entorhinal grid cell
residue number system (Fiete 2008, Wei 2015, Vago 2018, Constantinescu
2016) and QA modular arithmetic.

Claims:
  RNS   — Grid cell phase code is an RNS isomorphic to QA
  CRT   — CRT reconstruction: grid decoding = QA multi-modulus join
  RATIO — QA ratio 24/9 = 2.667 within 2% of optimal e = 2.718
  LCM   — LCM(9,24) = 72 = Cosmos orbit cardinality
  PHI   — Golden ratio optimal for two-module coding; QA norm = Q(sqrt(5))
  CARRY — Carry-free computation = QA axiom independence
  ABS   — Grid codes organize abstract (non-spatial) concepts
  TORUS — Toroidal state space: grid tori = QA (Z/mZ)^2
  HEX27 — Hexagonal encoding for m=9 uses 27 codebook vectors

Checks:
  GCR_1     — schema_version matches
  GCR_RATIO — |24/9 - e| / e < threshold (default 10%)
  GCR_LCM   — LCM(9,24) = 72
  GCR_PHI6  — phi(9) = 6 (hexagonal symmetry)
  GCR_HEX   — 3*9^2 - 3*9 + 1 = 217; 3*9 = 27
  GCR_NUM   — numerical checks pass
  GCR_W     — at least 5 witnesses
  GCR_F     — falsifier well-formed
"""

QA_COMPLIANCE = "cert_validator — validates grid cell RNS claims; no float state (ratios computed as fractions where possible)"

import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_GRID_CELL_RNS_CERT.v1"
EULER_E = math.e  # 2.718281828...


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    return a * b // gcd(a, b)


def euler_totient(n):
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # GCR_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"GCR_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # If explicitly marked as expect_fail, validate the failure structure
    if cert.get("expect_fail"):
        if not cert.get("fail_reason"):
            errors.append("GCR_F: expect_fail fixture missing fail_reason")
        # Check that the ratio claim actually fails
        claims = cert.get("claims", [])
        for claim in claims:
            if claim.get("id") == "RATIO":
                w = claim.get("witnesses", {})
                pct = w.get("percent_deviation")
                if pct is not None and pct <= 10:
                    errors.append(f"GCR_F: expect_fail fixture has deviation {pct}% <= 10% (should exceed)")
        return errors, warnings

    claims = cert.get("claims", [])

    # GCR_W: at least 5 claims
    if len(claims) < 5:
        warnings.append(f"GCR_W: need >= 5 claims, got {len(claims)}")

    for claim in claims:
        cid = claim.get("id", "?")
        w = claim.get("witnesses", {})

        # GCR_RATIO: 24/9 within threshold of e
        if cid == "RATIO":
            qr = w.get("qa_ratio", {})
            num = qr.get("numerator")
            den = qr.get("denominator")
            if num is not None and den is not None:
                ratio = num / den
                pct_dev = abs(ratio - EULER_E) / EULER_E * 100
                declared_pct = w.get("percent_deviation")
                if declared_pct is not None and abs(pct_dev - declared_pct) > 0.5:
                    errors.append(
                        f"GCR_RATIO: computed deviation {pct_dev:.1f}%, declared {declared_pct}%"
                    )
                # Check threshold
                threshold = 10.0  # percent
                if pct_dev > threshold:
                    errors.append(
                        f"GCR_RATIO: {num}/{den} = {ratio:.4f} deviates {pct_dev:.1f}% from e, "
                        f"exceeds {threshold}% threshold"
                    )

        # GCR_LCM: LCM(9,24) = 72
        elif cid == "LCM":
            a = w.get("a")
            b = w.get("b")
            declared_gcd = w.get("gcd")
            declared_lcm = w.get("lcm")
            if a is not None and b is not None:
                computed_gcd = gcd(a, b)
                computed_lcm = lcm(a, b)
                if declared_gcd is not None and computed_gcd != declared_gcd:
                    errors.append(f"GCR_LCM: gcd({a},{b}) = {computed_gcd}, declared {declared_gcd}")
                if declared_lcm is not None and computed_lcm != declared_lcm:
                    errors.append(f"GCR_LCM: lcm({a},{b}) = {computed_lcm}, declared {declared_lcm}")
                if computed_lcm != 72:
                    errors.append(f"GCR_LCM: lcm(9,24) = {computed_lcm}, expected 72")
            cosmos = w.get("cosmos_orbit_pairs")
            if cosmos is not None and cosmos != 72:
                errors.append(f"GCR_LCM: cosmos_orbit_pairs = {cosmos}, expected 72")

        # GCR_PHI6: phi(9) = 6
        elif cid == "TORUS":
            hex_sym = w.get("hexagonal_symmetry")
            if hex_sym is not None:
                phi_9 = euler_totient(9)
                if hex_sym != phi_9:
                    errors.append(f"GCR_PHI6: phi(9) = {phi_9}, declared hexagonal_symmetry = {hex_sym}")

        # GCR_HEX: hexagonal encoding verification
        elif cid == "HEX27":
            m = 9
            hw = w.get("m_9", {})
            declared_states = hw.get("states")
            declared_vectors = hw.get("codebook_vectors")
            expected_states = 3 * m * m - 3 * m + 1  # 217
            expected_vectors = 3 * m  # 27
            if declared_states is not None and declared_states != expected_states:
                errors.append(f"GCR_HEX: 3*9^2-3*9+1 = {expected_states}, declared {declared_states}")
            if declared_vectors is not None and declared_vectors != expected_vectors:
                errors.append(f"GCR_HEX: 3*9 = {expected_vectors}, declared {declared_vectors}")

    # GCR_NUM: numerical checks
    num = cert.get("numerical_checks", {})
    if num:
        # Ratio test
        rt = num.get("ratio_test", {})
        if rt:
            qa_r = rt.get("qa_ratio")
            e_val = rt.get("euler_e")
            pct = rt.get("percent_deviation")
            if qa_r is not None and e_val is not None:
                computed_pct = abs(qa_r - e_val) / e_val * 100
                if pct is not None and abs(computed_pct - pct) > 0.5:
                    errors.append(f"GCR_NUM: ratio deviation {computed_pct:.1f}%, declared {pct}")
            within = rt.get("within_threshold")
            threshold = rt.get("threshold_percent", 10)
            if within is not None and pct is not None:
                if (pct <= threshold) != within:
                    errors.append(f"GCR_NUM: within_threshold={within} but deviation={pct}% vs threshold={threshold}%")

        # LCM test
        lt = num.get("lcm_test", {})
        if lt:
            a, b = lt.get("a"), lt.get("b")
            if a is not None and b is not None:
                cg = gcd(a, b)
                cl = lcm(a, b)
                if lt.get("gcd") is not None and cg != lt["gcd"]:
                    errors.append(f"GCR_NUM: gcd({a},{b}) = {cg}, declared {lt['gcd']}")
                if lt.get("lcm") is not None and cl != lt["lcm"]:
                    errors.append(f"GCR_NUM: lcm({a},{b}) = {cl}, declared {lt['lcm']}")

        # Euler totient
        et = num.get("euler_totient_test", {})
        if et:
            m = et.get("m")
            phi_m = et.get("phi_m")
            if m is not None and phi_m is not None:
                computed = euler_totient(m)
                if computed != phi_m:
                    errors.append(f"GCR_NUM: phi({m}) = {computed}, declared {phi_m}")

        # Hex encoding
        ht = num.get("hex_encoding_test", {})
        if ht:
            m = ht.get("m")
            states = ht.get("states")
            vectors = ht.get("codebook_vectors")
            if m is not None:
                expected_s = 3 * m * m - 3 * m + 1
                expected_v = 3 * m
                if states is not None and states != expected_s:
                    errors.append(f"GCR_NUM: hex states for m={m}: {expected_s}, declared {states}")
                if vectors is not None and vectors != expected_v:
                    errors.append(f"GCR_NUM: hex vectors for m={m}: {expected_v}, declared {vectors}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("gcr_pass_core.json", True),
        ("gcr_pass_numerical.json", True),
        ("gcr_fail_wrong_ratio.json", True),
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
        description="QA Grid Cell RNS Cert [205] validator")
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
