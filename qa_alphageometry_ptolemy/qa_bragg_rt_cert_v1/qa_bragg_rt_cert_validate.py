#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=bragg_rt_fixtures"
"""QA Bragg RT Cert family [157] — certifies Bragg's law as rational trigonometry.

TIER 1 — EXACT ALGEBRAIC REFORMULATION:
  Classical:  nλ = 2d·sin(θ)
  Rational:   n²Q_λ = 4Q_d·s

  where Q_λ = λ²  (wavelength quadrance)
        Q_d = d²  (lattice spacing quadrance)
        s = sin²θ (diffraction spread)

  Derivation: square both sides of nλ = 2d·sinθ
    n²λ² = 4d²·sin²θ
    n²Q_λ = 4Q_d·s

  No transcendental functions. Works over any field. Exact over rationals
  when Q_λ, Q_d, s are rational (or integer-scaled).

MILLER INDEX QUADRANCE (cubic crystals):
  d² = a² / (h² + k² + l²) = a² / Q(h,k,l)
  where Q(h,k,l) = h² + k² + l² is the Miller index quadrance.
  Substituting: n²Q_λ·Q(h,k,l) = 4a²·s

CRYSTAL SYSTEM SPREAD CONDITIONS:
  7 crystal systems classified by lattice angle spreads:
  - Cubic: s_α = s_β = s_γ = 1 (all right angles, spread of 90° = 1)
  - Hexagonal: s_α = s_β = 1, s_γ = 3/4 (γ = 120°, spread = 3/4)
  - Tetragonal: s_α = s_β = s_γ = 1
  - Orthorhombic: s_α = s_β = s_γ = 1
  - (Others have irrational spreads in general)

SOURCE: W.H. Bragg & W.L. Bragg, Proc. R. Soc. A 88, 428-438 (1913).
Rational trig: Wildberger, Divine Proportions (2005).
Crystal universe: Ben Iverson, QA-4.

Checks
------
BRT_1       schema_version == 'QA_BRAGG_RT_CERT.v1'
BRT_BRAGG   n²Q_λ = 4Q_d·s verified for each witness reflection
BRT_MILLER  Q(h,k,l) = h²+k²+l² and d² = a²/Q(h,k,l) for cubic witnesses
BRT_SPREAD  Crystal system spread conditions verified
BRT_PYTH    No transcendental functions in rational formulation
BRT_W       At least 3 witness reflections
BRT_F       Fail detection
"""

import json
import os
import sys
from fractions import Fraction
from math import gcd

SCHEMA = "QA_BRAGG_RT_CERT.v1"


def validate_bragg_reflection(refl):
    """Validate n²Q_λ = 4Q_d·s for a single reflection.

    All values are checked as Fractions for exact arithmetic.
    Returns list of errors.
    """
    errors = []

    n = refl.get("n", 1)

    # Get quadrance values — accept integer or rational
    Q_lambda = Fraction(refl["Q_lambda"]) if "Q_lambda" in refl else None
    Q_d = Fraction(refl["Q_d"]) if "Q_d" in refl else None
    s = Fraction(refl["s"]) if "s" in refl else None

    if Q_lambda is None or Q_d is None or s is None:
        errors.append("BRT_BRAGG: missing Q_lambda, Q_d, or s")
        return errors

    # Core identity: n²Q_λ = 4Q_d·s
    lhs = n * n * Q_lambda
    rhs = 4 * Q_d * s

    # Allow small tolerance for float-derived values
    tol = Fraction(refl.get("tolerance", 0))
    if tol > 0:
        if abs(lhs - rhs) > tol:
            errors.append(f"BRT_BRAGG: n²Q_λ={float(lhs):.6f} ≠ 4Q_d·s={float(rhs):.6f} "
                         f"(diff={float(abs(lhs-rhs)):.2e}, tol={float(tol):.2e})")
    else:
        if lhs != rhs:
            errors.append(f"BRT_BRAGG: n²Q_λ={lhs} ≠ 4Q_d·s={rhs} (exact)")

    return errors


def validate_miller(miller):
    """Validate Miller index quadrance: Q(h,k,l) = h²+k²+l²."""
    errors = []

    h = miller.get("h", 0)
    k = miller.get("k", 0)
    l = miller.get("l", 0)

    Q_miller = h * h + k * k + l * l  # S1: no **2

    declared_Q = miller.get("Q_miller")
    if declared_Q is not None and declared_Q != Q_miller:
        errors.append(f"BRT_MILLER: declared Q({h},{k},{l})={declared_Q} ≠ computed {Q_miller}")

    # For cubic: d² = a²/Q(h,k,l)
    a_lattice = miller.get("a_lattice")
    declared_d_sq = miller.get("d_squared")
    if a_lattice is not None and declared_d_sq is not None and Q_miller > 0:
        computed_d_sq = Fraction(a_lattice * a_lattice, Q_miller)
        declared_d_sq_frac = Fraction(declared_d_sq) if isinstance(declared_d_sq, (int, str)) else Fraction(declared_d_sq).limit_denominator(10**12)
        tol = Fraction(1, 10**6)
        if abs(computed_d_sq - declared_d_sq_frac) > tol:
            errors.append(f"BRT_MILLER: d²={float(declared_d_sq_frac):.6f} ≠ a²/Q={float(computed_d_sq):.6f}")

    return errors


def validate_crystal_spread(crystal):
    """Validate crystal system spread conditions."""
    errors = []

    system = crystal.get("system", "")
    spreads = crystal.get("spreads", {})

    s_alpha = spreads.get("alpha")
    s_beta = spreads.get("beta")
    s_gamma = spreads.get("gamma")

    if system == "cubic":
        for name, val in [("alpha", s_alpha), ("beta", s_beta), ("gamma", s_gamma)]:
            if val is not None:
                if Fraction(val) != Fraction(1):
                    errors.append(f"BRT_SPREAD: cubic {name} spread={val} ≠ 1 (90° has spread 1)")

    elif system == "hexagonal":
        for name, val in [("alpha", s_alpha), ("beta", s_beta)]:
            if val is not None:
                if Fraction(val) != Fraction(1):
                    errors.append(f"BRT_SPREAD: hexagonal {name} spread={val} ≠ 1")
        if s_gamma is not None:
            if Fraction(s_gamma) != Fraction(3, 4):
                errors.append(f"BRT_SPREAD: hexagonal gamma spread={s_gamma} ≠ 3/4 (120° has spread 3/4)")

    return errors


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # BRT_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("BRT_1", f"schema_version must be {SCHEMA}")

    # BRT_BRAGG — Bragg identity for each witness
    witnesses = cert.get("witnesses", [])
    for i, w in enumerate(witnesses):
        werrs = validate_bragg_reflection(w)
        for we in werrs:
            err("BRT_BRAGG", f"witness {i}: {we}")

    # BRT_MILLER — Miller index quadrance
    miller_witnesses = cert.get("miller_witnesses", [])
    for i, m in enumerate(miller_witnesses):
        merrs = validate_miller(m)
        for me in merrs:
            err("BRT_MILLER", f"miller {i}: {me}")

    # BRT_SPREAD — crystal system spreads
    crystal_witnesses = cert.get("crystal_witnesses", [])
    for i, c in enumerate(crystal_witnesses):
        cerrs = validate_crystal_spread(c)
        for ce in cerrs:
            err("BRT_SPREAD", f"crystal {i}: {ce}")

    # BRT_PYTH — formulation is transcendental-free (declared)
    if cert.get("transcendental_free") is not True:
        warnings.append("BRT_PYTH: transcendental_free not declared true")

    # BRT_W — at least 3 witnesses
    total_witnesses = len(witnesses) + len(miller_witnesses) + len(crystal_witnesses)
    if total_witnesses < 3:
        err("BRT_W", f"need ≥3 total witnesses, got {total_witnesses}")

    # BRT_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("BRT_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("BRT_F: declared FAIL but no fail_ledger entries and all checks pass")

    return {
        "ok": not has_errors,
        "errors": errors,
        "warnings": warnings,
        "schema": SCHEMA,
    }


def self_test():
    """Run validator against bundled fixtures."""
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    results = {"pass_count": 0, "fail_count": 0, "errors": []}

    for fname in sorted(os.listdir(fixture_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(fixture_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            cert = json.load(f)

        out = validate(cert)
        declared = cert.get("result", "UNKNOWN")

        if declared == "PASS" and out["ok"]:
            results["pass_count"] += 1
        elif declared == "FAIL" and not out["ok"]:
            results["fail_count"] += 1
        else:
            results["errors"].append({
                "fixture": fname,
                "declared": declared,
                "validator_ok": out["ok"],
                "issues": out["errors"],
            })

    results["ok"] = len(results["errors"]) == 0
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    parser.add_argument("--self-test", action="store_true", help="Run self-test against fixtures")
    parser.add_argument("cert_file", nargs="?", help="Path to certificate JSON")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.cert_file:
        with open(args.cert_file, "r", encoding="utf-8") as f:
            cert = json.load(f)
        result = validate(cert)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
