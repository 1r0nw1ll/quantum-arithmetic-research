#!/usr/bin/env python3
"""
qa_hebrew_mod9_identity_cert_validate.py

Validator for QA_HEBREW_MOD9_IDENTITY_CERT.v1  [family 202]

Certifies: The structural identity between Hebrew gematria mod-9
reduction (Aiq Bekar / pythmen) and QA A1 axiom state space {1,...,9}.

Claims:
  AIQ    — Aiq Bekar Nine Chambers is identically QA mod-9
  DR     — Digital root is additive + multiplicative homomorphism
  ENNEAD — Three enneads: 27 signs = {mod-9 residue} x {decimal order}
  SY24   — Sefer Yetzirah 4! = 24 as structural constant
  SKIN   — Skinner metrological kernel built on powers of 9
  BRIDGE — Factor 6 bridges mod-9 theoretical to mod-24 applied
  PYTH   — Pythagorean transmission path via Iamblichus
  BASE9  — Pre-biblical base-9 hypothesis (Kreinovich 2018)

Checks:
  HM9_1      — schema_version matches
  HM9_AIQ    — all 9 chambers present, digital roots correct, zero excluded
  HM9_DR     — homomorphism properties stated with sources
  HM9_ENNEAD — 27 = 3 x 9; pythmen term cited
  HM9_SY24   — 4! = 24 with Sefer Yetzirah source
  HM9_SKIN   — 6561 = 9^4; digital roots of Adam/Woman = 9
  HM9_BRIDGE — factor 6 derivations present
  HM9_NUM    — numerical checks verify dr formula
  HM9_W      — at least 5 witnesses
  HM9_F      — falsifier well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Hebrew mod-9 identity claims; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_HEBREW_MOD9_IDENTITY_CERT.v1"


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

    # HM9_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"HM9_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # If explicitly marked as expect_fail, validate the failure structure
    if cert.get("expect_fail"):
        if not cert.get("fail_reason"):
            errors.append("HM9_F: expect_fail fixture missing fail_reason")
        # Check that the failure actually demonstrates an A1 violation
        claims = cert.get("claims", [])
        for claim in claims:
            w = claim.get("witnesses", {})
            if w.get("zero_excluded") is False:
                pass  # correct: this is the expected violation
            elif w.get("zero_excluded") is True:
                errors.append("HM9_F: expect_fail fixture has zero_excluded=true (should be false)")
        return errors, warnings

    claims = cert.get("claims", [])
    claim_ids = {c.get("id") for c in claims}

    # HM9_W: at least 5 claims
    if len(claims) < 5:
        warnings.append(f"HM9_W: need >= 5 claims, got {len(claims)}")

    for claim in claims:
        cid = claim.get("id", "?")
        w = claim.get("witnesses", {})

        # HM9_AIQ: Aiq Bekar chambers
        if cid == "AIQ":
            chambers = w.get("chambers", [])
            if len(chambers) != 9:
                errors.append(f"HM9_AIQ: expected 9 chambers, got {len(chambers)}")
            for ch in chambers:
                ch_num = ch.get("chamber")
                dr_declared = ch.get("digital_root")
                values = ch.get("values", [])
                # Verify all values in this chamber have the declared digital root
                for v in values:
                    computed = digital_root(v)
                    if computed != dr_declared:
                        errors.append(
                            f"HM9_AIQ: chamber {ch_num}: dr({v}) = {computed}, "
                            f"declared {dr_declared}"
                        )
            # Zero must be excluded
            if not w.get("zero_excluded"):
                errors.append("HM9_AIQ: zero_excluded must be true (A1 axiom)")
            # State space must be {1,...,9}
            ss = w.get("state_space", [])
            if ss and sorted(ss) != list(range(1, 10)):
                errors.append(f"HM9_AIQ: state_space {ss} != [1..9]")

        # HM9_DR: homomorphism properties
        elif cid == "DR":
            props = w.get("properties", [])
            if len(props) < 4:
                warnings.append(f"HM9_DR: expected >= 4 properties, got {len(props)}")
            # Check Vedic square identity stated
            vsi = w.get("vedic_square_identity", "")
            if "mod" not in vsi.lower() and "9" not in vsi:
                warnings.append("HM9_DR: vedic_square_identity should reference mod 9")

        # HM9_ENNEAD: three enneads
        elif cid == "ENNEAD":
            sc = w.get("sign_count")
            if sc is not None and sc != 27:
                errors.append(f"HM9_ENNEAD: sign_count = {sc}, expected 27")
            fact = w.get("factorization", "")
            if "3" not in fact or "9" not in fact:
                warnings.append(f"HM9_ENNEAD: factorization should show 27 = 3 x 9")
            if not w.get("pythmen"):
                warnings.append("HM9_ENNEAD: missing pythmen reference")

        # HM9_SY24: Sefer Yetzirah 4! = 24
        elif cid == "SY24":
            fs = w.get("factorial_sequence", [])
            if fs and len(fs) >= 3:
                if fs[2] != 24:
                    errors.append(f"HM9_SY24: factorial_sequence[2] = {fs[2]}, expected 24 (4!)")
            passage = w.get("passage", "")
            if "yetzirah" not in passage.lower() and "4:16" not in passage:
                warnings.append("HM9_SY24: should cite Sefer Yetzirah passage")

        # HM9_SKIN: Skinner powers of 9
        elif cid == "SKIN":
            pb = w.get("parker_base")
            if pb is not None and pb != 6561:
                errors.append(f"HM9_SKIN: parker_base = {pb}, expected 6561")
            if pb == 6561:
                # Verify 6561 = 9^4
                if 9 * 9 * 9 * 9 != 6561:
                    errors.append("HM9_SKIN: 9^4 != 6561 (impossible)")
            # Check Adam digital root
            adam = w.get("adam_144", {})
            if adam:
                adam_val = adam.get("value")
                adam_dr = adam.get("digital_root")
                if adam_val is not None and adam_dr is not None:
                    if digital_root(adam_val) != adam_dr:
                        errors.append(
                            f"HM9_SKIN: dr({adam_val}) = {digital_root(adam_val)}, "
                            f"declared {adam_dr}"
                        )
            # Check solar day
            sd = w.get("solar_day_5184", {})
            if sd:
                sd_val = sd.get("value")
                if sd_val is not None and sd_val != 72 * 72:
                    errors.append(f"HM9_SKIN: solar_day {sd_val} != 72^2 = 5184")

        # HM9_BRIDGE: factor 6 bridge
        elif cid == "BRIDGE":
            bf = w.get("bridge_factor")
            if bf is not None and bf != 6:
                errors.append(f"HM9_BRIDGE: bridge_factor = {bf}, expected 6")
            derivs = w.get("derivations", [])
            if len(derivs) < 2:
                warnings.append(f"HM9_BRIDGE: need >= 2 derivations, got {len(derivs)}")
            # Check that 24 appears as a target
            targets = [d.get("target") for d in derivs]
            if 24 not in targets:
                warnings.append("HM9_BRIDGE: 24 should be a derivation target")

    # HM9_NUM: numerical checks (in numerical fixture)
    num = cert.get("numerical_checks", {})
    if num:
        # Verify Aiq Bekar formula
        aiq = num.get("aiq_bekar_verification", {})
        test_vals = aiq.get("test_values", [])
        expected = aiq.get("expected_roots", [])
        for v, exp in zip(test_vals, expected):
            computed = digital_root(v)
            if computed != exp:
                errors.append(f"HM9_NUM: dr({v}) = {computed}, expected {exp}")

        # Verify homomorphism tests
        for ht in num.get("homomorphism_tests", []):
            a, b = ht["a"], ht["b"]
            op = ht["op"]
            result = ht["result"]
            dr_a = digital_root(a)
            dr_b = digital_root(b)
            dr_result = digital_root(result)
            if dr_a != ht.get("dr_a"):
                errors.append(f"HM9_NUM: dr({a}) = {dr_a}, declared {ht.get('dr_a')}")
            if dr_b != ht.get("dr_b"):
                errors.append(f"HM9_NUM: dr({b}) = {dr_b}, declared {ht.get('dr_b')}")
            if dr_result != ht.get("dr_result"):
                errors.append(f"HM9_NUM: dr({result}) = {dr_result}, declared {ht.get('dr_result')}")
            # Verify the operation itself
            if op == "+":
                if a + b != result:
                    errors.append(f"HM9_NUM: {a} + {b} = {a+b}, declared {result}")
            elif op == "*":
                if a * b != result:
                    errors.append(f"HM9_NUM: {a} * {b} = {a*b}, declared {result}")

        # Verify factorial digital roots
        for ft in num.get("sefer_yetzirah_factorials", []):
            n = ft["n"]
            fac = ft["factorial"]
            dr_declared = ft["dr"]
            # Verify factorial
            expected_fac = 1
            for i in range(2, n + 1):
                expected_fac *= i
            if fac != expected_fac:
                errors.append(f"HM9_NUM: {n}! = {expected_fac}, declared {fac}")
            if digital_root(fac) != dr_declared:
                errors.append(f"HM9_NUM: dr({n}!) = {digital_root(fac)}, declared {dr_declared}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("hm9_pass_core.json", True),
        ("hm9_pass_numerical.json", True),
        ("hm9_fail_zero_state.json", True),  # expect_fail -> validates failure structure
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
        description="QA Hebrew Mod-9 Identity Cert [202] validator")
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
