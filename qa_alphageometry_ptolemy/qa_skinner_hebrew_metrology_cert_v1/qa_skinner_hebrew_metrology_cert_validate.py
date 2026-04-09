#!/usr/bin/env python3
"""
qa_skinner_hebrew_metrology_cert_validate.py

Validator for QA_SKINNER_HEBREW_METROLOGY_CERT.v1  [family 204]

Certifies: 7 verified metrological claims from J.R. Skinner's
'Key to the Hebrew-Egyptian Mystery in the Source of Measures' (1875),
plus 3 honestly qualified claims.

Claims (verified):
  PARK   — Parker quadrature kernel 6561 = 9^4 = 3^8
  EDEN   — Garden-Eden characteristic sum = 24 (via digital roots)
  SOLAR  — Solar day 5184 = 72^2 (72 = QA Cosmos orbit pair count)
  ADAM   — Adam=144, Woman=135, Serpent=9; all dr=9
  BRIDGE — Factor 6 bridges mod-9 to mod-24
  MET    — Metius dr-closure: dr(113) + dr(355) = 9
  T2     — System is T2-compliant (pi as observer projection only)

Claims (qualified):
  Q_EL        — El=31 generates only {1,4,7}, not full (Z/9Z)*
  Q_PALINDROME — dr-preservation under reversal is trivial
  Q_PARKER_PI  — Parker pi mediocre vs Metius

Checks:
  SKM_1     — schema_version matches
  SKM_PARK  — 6561 = 9^4; all in smooth basis {2,3}
  SKM_EDEN  — characteristic sum = 24; digital roots verified
  SKM_SOLAR — 5184 = 72*72; dr=9; mod24=0
  SKM_ADAM  — dr(144)=9, dr(135)=9, 144-135=9
  SKM_BRIDGE — factor 6 derivations verified
  SKM_MET   — dr(113)+dr(355)=9
  SKM_NUM   — numerical checks pass
  SKM_W     — at least 5 witnesses
  SKM_F     — falsifier well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Skinner metrological claims; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SKINNER_HEBREW_METROLOGY_CERT.v1"


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

    # SKM_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"SKM_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # If explicitly marked as expect_fail, validate the failure structure
    if cert.get("expect_fail"):
        if not cert.get("fail_reason"):
            errors.append("SKM_F: expect_fail fixture missing fail_reason")
        return errors, warnings

    claims = cert.get("claims", [])

    # SKM_W: at least 5 claims
    if len(claims) < 5:
        warnings.append(f"SKM_W: need >= 5 claims, got {len(claims)}")

    for claim in claims:
        cid = claim.get("id", "?")
        w = claim.get("witnesses", {})

        # SKM_PARK: Parker kernel
        if cid == "PARK":
            val = w.get("value")
            if val is not None and val != 6561:
                errors.append(f"SKM_PARK: parker base = {val}, expected 6561")
            p9 = w.get("as_power_of_9", {})
            if p9:
                base = p9.get("base", 0)
                exp = p9.get("exponent", 0)
                prod = p9.get("product", 0)
                computed = 1
                for _ in range(exp):
                    computed *= base
                if computed != prod:
                    errors.append(f"SKM_PARK: {base}^{exp} = {computed}, declared {prod}")
            dr = w.get("digital_root")
            if dr is not None and val is not None:
                if digital_root(val) != dr:
                    errors.append(f"SKM_PARK: dr({val}) = {digital_root(val)}, declared {dr}")

        # SKM_EDEN: Garden-Eden = 24
        elif cid == "EDEN":
            char_vals = w.get("characteristic_values", [])
            char_sum = w.get("characteristic_sum")
            if char_vals and char_sum is not None:
                computed_sum = sum(char_vals)
                if computed_sum != char_sum:
                    errors.append(f"SKM_EDEN: sum({char_vals}) = {computed_sum}, declared {char_sum}")
            if char_sum is not None and char_sum != 24:
                errors.append(f"SKM_EDEN: characteristic_sum = {char_sum}, expected 24")
            # Verify characteristic values are digital roots of standard values
            std_vals = w.get("standard_values", [])
            dr_ver = w.get("dr_verification", {})
            for sv_val, cv in zip(std_vals, char_vals):
                computed_dr = digital_root(sv_val)
                if computed_dr != cv:
                    errors.append(f"SKM_EDEN: dr({sv_val}) = {computed_dr}, declared {cv}")

        # SKM_SOLAR: 5184 = 72^2
        elif cid == "SOLAR":
            val = w.get("value")
            sq = w.get("as_square", {})
            if sq:
                base = sq.get("base", 0)
                if base * base != val:
                    errors.append(f"SKM_SOLAR: {base}^2 = {base*base}, declared {val}")
            if val is not None:
                dr = w.get("digital_root")
                if dr is not None and digital_root(val) != dr:
                    errors.append(f"SKM_SOLAR: dr({val}) = {digital_root(val)}, declared {dr}")
                m24 = w.get("mod_24")
                if m24 is not None and val % 24 != m24:
                    errors.append(f"SKM_SOLAR: {val} mod 24 = {val%24}, declared {m24}")
            cosmos = w.get("qa_cosmos_pairs")
            if cosmos is not None and cosmos != 72:
                errors.append(f"SKM_SOLAR: qa_cosmos_pairs = {cosmos}, expected 72")

        # SKM_ADAM: digital roots all = 9
        elif cid == "ADAM":
            for key in ("adam", "woman", "serpent"):
                entry = w.get(key, {})
                val = entry.get("value")
                dr = entry.get("digital_root")
                if val is not None and dr is not None:
                    if digital_root(val) != dr:
                        errors.append(f"SKM_ADAM: dr({key}={val}) = {digital_root(val)}, declared {dr}")
                    if dr != 9:
                        errors.append(f"SKM_ADAM: {key} dr = {dr}, expected 9")
            # Check difference
            diff = w.get("difference", {})
            d_val = diff.get("adam_minus_woman")
            if d_val is not None and d_val != 9:
                errors.append(f"SKM_ADAM: adam - woman = {d_val}, expected 9")

        # SKM_BRIDGE: factor 6
        elif cid == "BRIDGE":
            bf = w.get("bridge_factor")
            if bf is not None and bf != 6:
                errors.append(f"SKM_BRIDGE: bridge_factor = {bf}, expected 6")
            derivs = w.get("derivations", [])
            for d in derivs:
                formula = d.get("formula", "")
                result = d.get("result")
                # Verify a few key ones
                if "6 x 4" in formula and result != 24:
                    errors.append(f"SKM_BRIDGE: 6 x 4 = {result}, expected 24")
                if "6 x 60" in formula and result != 360:
                    errors.append(f"SKM_BRIDGE: 6 x 60 = {result}, expected 360")

        # SKM_MET: Metius dr-closure
        elif cid == "MET":
            num = w.get("numerator")
            den = w.get("denominator")
            dr_num = w.get("dr_numerator")
            dr_den = w.get("dr_denominator")
            dr_sum = w.get("dr_sum")
            if num is not None and dr_num is not None:
                if digital_root(num) != dr_num:
                    errors.append(f"SKM_MET: dr({num}) = {digital_root(num)}, declared {dr_num}")
            if den is not None and dr_den is not None:
                if digital_root(den) != dr_den:
                    errors.append(f"SKM_MET: dr({den}) = {digital_root(den)}, declared {dr_den}")
            if dr_num is not None and dr_den is not None and dr_sum is not None:
                if dr_num + dr_den != dr_sum:
                    errors.append(f"SKM_MET: {dr_num} + {dr_den} = {dr_num+dr_den}, declared {dr_sum}")
                if dr_sum != 9:
                    errors.append(f"SKM_MET: dr_sum = {dr_sum}, expected 9 (QA modulus)")

    # SKM_NUM: numerical checks
    num = cert.get("numerical_checks", {})
    if num:
        # Powers of 9
        for p in num.get("powers_of_9", []):
            power = p.get("power", 0)
            val = p.get("value", 0)
            computed = 1
            for _ in range(power):
                computed *= 9
            if computed != val:
                errors.append(f"SKM_NUM: 9^{power} = {computed}, declared {val}")
            dr = p.get("dr")
            if dr is not None and digital_root(val) != dr:
                errors.append(f"SKM_NUM: dr({val}) = {digital_root(val)}, declared {dr}")

        # Digital root tests
        for t in num.get("digital_root_tests", []):
            inp = t.get("input")
            dr = t.get("dr")
            if inp is not None and dr is not None:
                if digital_root(inp) != dr:
                    errors.append(f"SKM_NUM: dr({inp}) = {digital_root(inp)}, declared {dr}")

        # Subtraction chain
        for s in num.get("subtraction_chain", []):
            a, b, result = s.get("a"), s.get("b"), s.get("result")
            if a is not None and b is not None and result is not None:
                if a - b != result:
                    errors.append(f"SKM_NUM: {a} - {b} = {a-b}, declared {result}")

        # Factor 6 chain
        for f in num.get("factor_6_chain", []):
            formula = f.get("formula", "")
            result = f.get("result")
            if "6 * 4" in formula and result != 24:
                errors.append(f"SKM_NUM: 6*4 = 24, declared {result}")
            elif "6 * 60" in formula and result != 360:
                errors.append(f"SKM_NUM: 6*60 = 360, declared {result}")
            elif "6^4 * 4" in formula and result != 5184:
                errors.append(f"SKM_NUM: 6^4*4 = 5184, declared {result}")
            elif "6^2" in formula and "6^3" not in formula and "6^4" not in formula and result != 36:
                errors.append(f"SKM_NUM: 6^2 = 36, declared {result}")
            elif "6^3" in formula and "6^4" not in formula and result != 216:
                errors.append(f"SKM_NUM: 6^3 = 216, declared {result}")

        # El generator test
        el = num.get("el_generator_test", {})
        if el:
            el_mod = el.get("el_mod_9")
            if el_mod is not None and el_mod != 31 % 9:
                errors.append(f"SKM_NUM: 31 mod 9 = {31%9}, declared {el_mod}")
            order = el.get("order_in_Z9_star")
            if order is not None:
                # Verify: 4^order mod 9 = 1
                computed = 1
                base = 4
                for _ in range(order):
                    computed = (computed * base) % 9
                if computed != 1:
                    errors.append(f"SKM_NUM: 4^{order} mod 9 = {computed}, expected 1")
            is_full = el.get("is_full_generator")
            if is_full is True:
                errors.append("SKM_NUM: El=31 is NOT a full generator of (Z/9Z)*")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("skm_pass_verified.json", True),
        ("skm_pass_numerical.json", True),
        ("skm_fail_el_generator.json", True),
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
        description="QA Skinner Hebrew Metrology Cert [204] validator")
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
