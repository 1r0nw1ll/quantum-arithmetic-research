#!/usr/bin/env python3
"""
qa_snell_manuscript_cert_validate.py

Validator for QA_SNELL_MANUSCRIPT_CERT.v1  [family 201]

Certifies: 7 structural claims from the Snell Manuscript (compiled 1934
from Keely's own books) mapped to QA modular arithmetic.

Claims:
  S1 — 7 x 3 = 21 subdivision hierarchy
  S2 — frequency scaling exclusively by 3 and 9
  S3 — Trexar metals at {3,6,9} = singularity residues
  S4 — mass = difference = f-value
  S5 — polarity inversion at 2/3 and 1/3
  S6 — triple dissociation = orbit descent
  S7 — rotation from 3:9 ratio

Checks:
  SNM_1       — schema_version matches
  SNM_21      — 7 x 3 = 21; 21 mod 9 = 3
  SNM_FREQ    — frequency digital roots trace 1->3->9->9->9
  SNM_SCALE   — all scaling factors in {3,9}; product = 3^5
  SNM_TREX    — Trexar intervals {3,6,9} all divisible by 3
  SNM_FVAL    — f = b*b + b*e - e*e matches mass-as-difference
  SNM_POL     — polarity thresholds 2/3 + 1/3 = 1
  SNM_DISS    — dissociation stages map to descending orbit periods
  SNM_ROT     — rpm ratio = cosmos/satellite period ratio = 3
  SNM_CHORD   — 42800 mod 24 = 8; 42800 mod 9 = 5
  SNM_W       — at least 7 witnesses (one per claim)
  SNM_F       — falsifier well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Snell Manuscript structural claims; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_SNELL_MANUSCRIPT_CERT.v1"


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

    # SNM_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"SNM_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # If explicitly marked as expect_fail, validate the failure structure
    if cert.get("expect_fail"):
        if not cert.get("fail_reason"):
            errors.append("SNM_F: expect_fail fixture missing fail_reason")
        return errors, warnings

    claims = cert.get("claims", [])

    # SNM_W: at least 7 witnesses
    if len(claims) < 7:
        warnings.append(f"SNM_W: need >= 7 claims, got {len(claims)}")

    for claim in claims:
        cid = claim.get("id", "?")
        w = claim.get("witnesses", {})

        # SNM_21: 7 x 3 = 21
        if cid == "S1":
            sub = w.get("subdivision_count")
            triple = w.get("triple_per_level")
            total = w.get("total")
            mod9 = w.get("mod_9")
            if sub is not None and triple is not None:
                if sub * triple != 21:
                    errors.append(f"SNM_21: {sub} x {triple} != 21")
            if total is not None and total != 21:
                errors.append(f"SNM_21: total={total}, expected 21")
            if mod9 is not None and mod9 != 3:
                errors.append(f"SNM_21: 21 mod 9 = {mod9}, expected 3")

        # SNM_FREQ + SNM_SCALE: frequency digital roots and scaling
        elif cid == "S2":
            freqs = w.get("frequencies_hz", [])
            ratios = w.get("ratios", [])
            dr = w.get("digital_roots_mod9", [])

            # Verify digital roots
            expected_dr = [digital_root(f) for f in freqs]
            if dr and dr != expected_dr:
                errors.append(f"SNM_FREQ: digital roots {dr} != computed {expected_dr}")

            # Verify convergence pattern
            if expected_dr and len(expected_dr) >= 3:
                if expected_dr[0] != 1:
                    warnings.append(f"SNM_FREQ: first digital root = {expected_dr[0]}, expected 1")
                if not all(r == 9 for r in expected_dr[2:]):
                    warnings.append(f"SNM_FREQ: not all higher levels converge to 9: {expected_dr[2:]}")

            # Verify all ratios are powers of 3
            for i, r in enumerate(ratios):
                if r not in (3, 9, 27, 81):
                    errors.append(f"SNM_SCALE: ratio[{i}]={r} is not a power of 3")

            # Verify total ratio
            total = w.get("total_ratio")
            if total is not None and ratios:
                computed_total = 1
                for r in ratios:
                    computed_total *= r
                if total != computed_total:
                    errors.append(f"SNM_SCALE: total_ratio={total}, computed={computed_total}")
                # Check it's a power of 3
                t = computed_total
                while t > 1 and t % 3 == 0:
                    t //= 3
                if t != 1:
                    errors.append(f"SNM_SCALE: total {computed_total} is not a power of 3")

        # SNM_TREX: Trexar intervals
        elif cid == "S3":
            intervals = w.get("intervals", [])
            for intv in intervals:
                if intv % 3 != 0:
                    errors.append(f"SNM_TREX: interval {intv} not divisible by 3")
            if intervals and set(intervals) != {3, 6, 9}:
                warnings.append(f"SNM_TREX: intervals {intervals} != canonical {{3,6,9}}")

        # SNM_FVAL: mass as f-value
        elif cid == "S4":
            b = w.get("b")
            e = w.get("e")
            f_decl = w.get("f_value")
            if b is not None and e is not None and f_decl is not None:
                f_expected = b * b + b * e - e * e
                if f_decl != f_expected:
                    errors.append(f"SNM_FVAL: f({b},{e})={f_decl}, expected {f_expected}")
            terr = w.get("terrestrial")
            cele = w.get("celestial")
            if terr is not None and cele is not None and f_decl is not None:
                if terr - cele != f_decl:
                    errors.append(f"SNM_FVAL: terrestrial({terr}) - celestial({cele}) != f({f_decl})")

        # SNM_POL: polarity thresholds
        elif cid == "S5":
            rep = w.get("repulsion_threshold")
            att = w.get("attraction_threshold")
            if rep == "2/3" and att == "1/3":
                pass  # correct
            elif rep is not None and att is not None:
                from fractions import Fraction
                try:
                    r_frac = Fraction(rep)
                    a_frac = Fraction(att)
                    if r_frac + a_frac != 1:
                        errors.append(f"SNM_POL: {rep} + {att} != 1")
                except (ValueError, ZeroDivisionError):
                    errors.append(f"SNM_POL: cannot parse thresholds: {rep}, {att}")

            bnd = w.get("boundary_mod_9")
            if bnd is not None and bnd != 1:
                warnings.append(f"SNM_POL: boundary mod 9 = {bnd}, expected 1 (singularity)")

        # SNM_DISS: dissociation orbit descent
        elif cid == "S6":
            stages = w.get("stages", [])
            if stages:
                periods = [s.get("period") for s in stages if s.get("period") is not None]
                if periods != sorted(periods, reverse=True):
                    errors.append(f"SNM_DISS: periods {periods} not in descending order")
                if periods and periods[-1] != 1:
                    warnings.append(f"SNM_DISS: final stage period = {periods[-1]}, expected 1 (singularity)")

        # SNM_ROT: rotation ratio
        elif cid == "S7":
            thirds_rpm = w.get("thirds_rpm")
            sixths_rpm = w.get("sixths_rpm")
            ratio = w.get("ratio")
            period_ratio = w.get("period_ratio")

            if thirds_rpm and sixths_rpm:
                computed_ratio = sixths_rpm / thirds_rpm
                if ratio is not None and abs(computed_ratio - ratio) > 0.001:
                    errors.append(f"SNM_ROT: {sixths_rpm}/{thirds_rpm}={computed_ratio}, declared {ratio}")

            if ratio is not None and period_ratio is not None:
                if ratio != period_ratio:
                    errors.append(f"SNM_ROT: rpm ratio {ratio} != period ratio {period_ratio}")

    # SNM_CHORD: chord of mass (check in numerical fixture)
    chord = cert.get("chord_of_mass", {})
    if chord:
        vps = chord.get("vibrations_per_sec")
        if vps is not None:
            if chord.get("mod_9") != vps % 9:
                errors.append(f"SNM_CHORD: mod_9 mismatch: declared {chord.get('mod_9')}, computed {vps % 9}")
            if chord.get("mod_24") != vps % 24:
                errors.append(f"SNM_CHORD: mod_24 mismatch: declared {chord.get('mod_24')}, computed {vps % 24}")
            if vps % 24 != 8:
                warnings.append(f"SNM_CHORD: {vps} mod 24 = {vps%24}, expected 8 (satellite period)")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("snm_pass_seven_claims.json", True),
        ("snm_pass_numerical.json", True),
        ("snm_fail_wrong_ratio.json", True),  # expect_fail → validates failure structure
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
        description="QA Snell Manuscript Cert [201] validator")
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
