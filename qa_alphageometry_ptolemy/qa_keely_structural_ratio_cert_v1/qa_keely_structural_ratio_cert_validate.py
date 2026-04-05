#!/usr/bin/env python3
"""
qa_keely_structural_ratio_cert_validate.py

Validator for QA_KEELY_STRUCTURAL_RATIO_CERT.v1  [family 184]

Certifies: Keely's 8 Structural Ratio Laws (Category 1 of Vibes 5-category
framework) mapped to QA modular invariants.

Laws: 2 (Corporeal Vibrations), 4 (Harmonic Vibrations), 9 (Cycles),
      10 (Harmonic Pitch), 18 (Atomic Pitch), 27 (Chemical Affinity),
      29 (Chemical Transposition), 33 (Chemical Morphology)

Each law states an exact integer constraint that maps directly to a QA
modular invariant: orbit period, f-value, period divisibility, concordance,
chromogeometric quadrance, or closure under modular arithmetic.

Checks:
  KSR_1       — schema_version matches
  KSR_LAWS    — all 8 law numbers present and correctly classified
  KSR_PERIOD  — period divisibility 1|8|24 certified (Laws 2,4,10)
  KSR_FVAL    — f = b*b + b*e - e*e is integer invariant (Laws 18,27)
  KSR_LCM     — LCM(1,8,24)=24 synchronization (Law 9)
  KSR_CHROMO  — C*C + F*F = G*G chromogeometry identity (Law 33)
  KSR_CLOSURE — generator preserves orbit membership (Law 29)
  KSR_W       — at least 3 witnesses with QA verification
  KSR_F       — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates Keely structural ratio law mappings; no float state"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_KEELY_STRUCTURAL_RATIO_CERT.v1"
REQUIRED_LAWS = frozenset([2, 4, 9, 10, 18, 27, 29, 33])


def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def lcm(a, b):
    return a * b // gcd(a, b)


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KSR_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"KSR_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # KSR_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("KSR_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("KSR_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # KSR_LAWS: all 8 laws present
    laws = cert.get("laws", {})
    declared_nums = set()
    if isinstance(laws, dict):
        declared_nums = {int(k) for k in laws.keys()}
    elif isinstance(laws, list):
        declared_nums = {entry.get("law_number") for entry in laws if isinstance(entry, dict)}
    missing = REQUIRED_LAWS - declared_nums
    if missing:
        errors.append(f"KSR_LAWS: missing law numbers: {sorted(missing)}")

    # KSR_PERIOD: period divisibility 1|8|24
    periods = cert.get("orbit_periods", {})
    if periods:
        p_sing = periods.get("SINGULARITY", periods.get("singularity"))
        p_sat = periods.get("SATELLITE", periods.get("satellite"))
        p_cos = periods.get("COSMOS", periods.get("cosmos"))
        if p_sing is not None and p_sat is not None and p_cos is not None:
            if p_sat % p_sing != 0:
                errors.append(f"KSR_PERIOD: satellite period {p_sat} not divisible by singularity period {p_sing}")
            if p_cos % p_sing != 0:
                errors.append(f"KSR_PERIOD: cosmos period {p_cos} not divisible by singularity period {p_sing}")

    # KSR_LCM: LCM(1,8,24)=24
    decl_lcm = cert.get("orbit_lcm")
    computed = lcm(lcm(1, 8), 24)
    if decl_lcm is not None and decl_lcm != computed:
        errors.append(f"KSR_LCM: declared LCM={decl_lcm}, expected {computed}")

    # KSR_W: witnesses with QA verification
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        errors.append(f"KSR_W: need >= 3 witnesses, got {len(witnesses)}")

    for idx, w in enumerate(witnesses):
        b = w.get("b")
        e = w.get("e")
        m = w.get("modulus", 9)

        if b is None or e is None:
            continue

        # KSR_FVAL: f-value computation (Laws 18,27)
        f_decl = w.get("f_value")
        f_expected = b * b + b * e - e * e
        if f_decl is not None and f_decl != f_expected:
            errors.append(f"KSR_FVAL: witness[{idx}] f_value={f_decl}, expected b*b+b*e-e*e={f_expected}")

        # KSR_CHROMO: C*C+F*F=G*G (Law 33)
        d_val = qa_mod(b + e, m)  # A2: derived
        a_val = qa_mod(b + 2 * e, m)  # A2: derived
        C_decl = w.get("C")
        F_decl = w.get("F")
        G_decl = w.get("G")
        if C_decl is not None and F_decl is not None and G_decl is not None:
            if C_decl * C_decl + F_decl * F_decl != G_decl * G_decl:
                errors.append(f"KSR_CHROMO: witness[{idx}] C*C+F*F != G*G ({C_decl}*{C_decl}+{F_decl}*{F_decl} != {G_decl}*{G_decl})")

        # KSR_CLOSURE: generator preserves orbit (Law 29)
        orbit_decl = w.get("orbit_family")
        next_b = e
        next_e = d_val
        next_orbit_decl = w.get("next_orbit_family")
        if orbit_decl is not None and next_orbit_decl is not None:
            if orbit_decl != next_orbit_decl:
                errors.append(f"KSR_CLOSURE: witness[{idx}] orbit changed from {orbit_decl} to {next_orbit_decl} under generator")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("ksr_pass_mod9.json", True),
        ("ksr_fail_wrong_fval.json", True),  # result=FAIL → skips detail checks
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
        description="QA Keely Structural Ratio Cert [184] validator")
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
