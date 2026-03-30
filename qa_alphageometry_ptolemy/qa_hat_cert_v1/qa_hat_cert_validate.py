#!/usr/bin/env python3
"""
qa_hat_cert_validate.py

Validator for QA_HAT_CERT.v1  [family 131]

Certifies the bridge between H. Lee Price's half-angle tangents (HATs)
and QA direction structure:

  HAT₁ = e/d = C/(G+F)          [primary; Price 2008 §2]
  HAT₂ = (d-e)/(d+e) = F/(G+C)  [secondary]
  spread s = E/G = HAT₁²/(1+HAT₁²)  [Wildberger rational trig]

Checks
------
HAT_1  schema_version == 'QA_HAT_CERT.v1'
HAT_2  HAT1 numerator/denominator == e/d (reduced)
HAT_3  HAT1 == C/(G+F) (Price formula — must equal e/d)
HAT_4  HAT2 == (d-e)/(d+e)
HAT_5  HAT2 == F/(G+C) (Price formula)
HAT_6  HAT1² == E/D
HAT_7  spread s == E/G
HAT_8  spread s == HAT1²/(1+HAT1²)
HAT_W  ≥3 witnesses (witness fixture)
HAT_F  fundamental witness (d=2,e=1) present with HAT1=1/2
"""

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from fractions import Fraction
from pathlib import Path

SCHEMA_VERSION = "QA_HAT_CERT.v1"


def check_direction(d, e, cert):
    """Verify all HAT identities for a single direction fixture."""
    errors = []
    C = 2 * d * e
    F = d * d - e * e
    G = d * d + e * e
    E = e * e
    D = d * d

    # HAT1 from cert
    h1 = cert.get("hat_primary", {})
    h1_num = h1.get("hat1", {}).get("numerator")
    h1_den = h1.get("hat1", {}).get("denominator")

    if h1_num is not None and h1_den is not None:
        # HAT_2: hat1 == e/d (reduced)
        g = gcd(e, d)
        exp_num, exp_den = e // g, d // g
        if h1_num != exp_num or h1_den != exp_den:
            errors.append(f"HAT_2: hat1={h1_num}/{h1_den}, expected e/d={exp_num}/{exp_den}")

        # HAT_3: C/(G+F) == e/d
        gpf = G + F  # = 2d²
        if gpf == 0 or C * exp_den != exp_num * gpf:
            errors.append(f"HAT_3: C/(G+F)={C}/{gpf} ≠ e/d={exp_num}/{exp_den}")

        # HAT_6: hat1² == E/D
        # hat1² = h1_num²/h1_den²; E/D = e²/d² = (e/d)² — same after reduction
        if h1_num * h1_num * D != E * h1_den * h1_den:
            errors.append(f"HAT_6: hat1²={h1_num*h1_num}/{h1_den*h1_den} ≠ E/D={E}/{D}")

    # HAT2
    h2 = cert.get("hat_secondary", {})
    h2_num = h2.get("hat2", {}).get("numerator")
    h2_den = h2.get("hat2", {}).get("denominator")

    if h2_num is not None and h2_den is not None:
        dm = d - e
        dp = d + e
        g2 = gcd(dm, dp)
        exp2_num, exp2_den = dm // g2, dp // g2
        # HAT_4
        if h2_num != exp2_num or h2_den != exp2_den:
            errors.append(f"HAT_4: hat2={h2_num}/{h2_den}, expected (d-e)/(d+e)={exp2_num}/{exp2_den}")
        # HAT_5: F/(G+C) == (d-e)/(d+e)
        gpc = G + C  # = (d+e)²
        if F * exp2_den != exp2_num * gpc:
            errors.append(f"HAT_5: F/(G+C)={F}/{gpc} ≠ (d-e)/(d+e)={exp2_num}/{exp2_den}")

    # Spread
    sp = cert.get("spread", {})
    s_num = sp.get("s", {}).get("numerator")
    s_den = sp.get("s", {}).get("denominator")

    if s_num is not None and s_den is not None:
        gs = gcd(E, G)
        exp_s_num, exp_s_den = E // gs, G // gs
        # HAT_7
        if s_num != exp_s_num or s_den != exp_s_den:
            errors.append(f"HAT_7: spread={s_num}/{s_den}, expected E/G={exp_s_num}/{exp_s_den}")
        # HAT_8: s == hat1²/(1+hat1²) = E/D / (1 + E/D) = E/(D+E) = E/G ✓ (since D+E=G)
        if D + E != G:
            errors.append(f"HAT_8: D+E={D+E} ≠ G={G} — identity requires d²+e²=G")

    return errors


def check_witness(w):
    """Verify all HAT identities for a witness entry."""
    errors = []
    d, e = w["d"], w["e"]
    C = 2 * d * e
    F = d * d - e * e
    G = d * d + e * e
    E = e * e
    D = d * d

    # HAT1
    h1n = w.get("HAT1_num")
    h1d = w.get("HAT1_den")
    if h1n is not None and h1d is not None:
        g = gcd(e, d)
        if h1n != e // g or h1d != d // g:
            errors.append(f"({d},{e}) HAT1={h1n}/{h1d} ≠ e/d={e//g}/{d//g}")
        if C * (d // g) != (e // g) * (G + F):
            errors.append(f"({d},{e}) C/(G+F)≠e/d")

    # HAT2
    h2n = w.get("HAT2_num")
    h2d = w.get("HAT2_den")
    if h2n is not None and h2d is not None:
        dm, dp = d - e, d + e
        g2 = gcd(dm, dp)
        if h2n != dm // g2 or h2d != dp // g2:
            errors.append(f"({d},{e}) HAT2={h2n}/{h2d} ≠ (d-e)/(d+e)={dm//g2}/{dp//g2}")

    # Spread
    sn = w.get("spread_num")
    sd = w.get("spread_den")
    if sn is not None and sd is not None:
        gs = gcd(E, G)
        if sn != E // gs or sd != G // gs:
            errors.append(f"({d},{e}) spread={sn}/{sd} ≠ E/G={E//gs}/{G//gs}")
        if D + E != G:
            errors.append(f"({d},{e}) D+E≠G — HAT_8 identity fails")

    # Fibonacci box det (informational — just verify declared det matches)
    box = w.get("fibonacci_box")
    det_declared = w.get("det")
    if box is not None and det_declared is not None:
        det_computed = box[0][0] * box[1][1] - box[0][1] * box[1][0]
        if det_computed != det_declared:
            errors.append(f"({d},{e}) Fibonacci box det declared={det_declared}, computed={det_computed}")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # HAT_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"HAT_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}")

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print(f"  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # Direction fixture checks
    direction = cert.get("direction")
    if direction:
        d, e = direction.get("d", 0), direction.get("e", 0)
        dir_errors = check_direction(d, e, cert)
        errors.extend(dir_errors)

    # Witness fixture checks
    general = cert.get("general_theorem")
    if general:
        witnesses = cert.get("witnesses", [])
        if len(witnesses) < 3:
            errors.append(f"HAT_W FAIL: need ≥3 witnesses, got {len(witnesses)}")
        else:
            has_fundamental = False
            for w in witnesses:
                werr = check_witness(w)
                errors.extend([f"HAT_W/F: {e}" for e in werr])
                if w.get("d") == 2 and w.get("e") == 1:
                    has_fundamental = True
                    h1n = w.get("HAT1_num")
                    h1d = w.get("HAT1_den")
                    if h1n != 1 or h1d != 2:
                        errors.append(f"HAT_F FAIL: fundamental (2,1) HAT1={h1n}/{h1d}, expected 1/2")
            if not has_fundamental:
                errors.append("HAT_F FAIL: no fundamental witness (d=2,e=1) found")

    # Internal validation_checks
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend([f"internal check {c['check_id']} not passed" for c in failed_internal])

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "hat_pass_fundamental.json",
        "hat_pass_witnesses.json",
    ]
    results = []
    all_ok = True
    for fname in expected_pass:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errors, warnings = validate(fpath)
            passed = len(errors) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue
        if not passed:
            all_ok = False
        results.append({"fixture": fname, "ok": passed, "errors": errors})
    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA HAT Cert [131] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths
    if not paths:
        here = Path(__file__).parent / "fixtures"
        paths = list(here.glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errors, warnings = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warnings:
            print(f"  WARN: {w}")
        for e in errors:
            print(f"  FAIL: {e}")
        if not errors:
            print(f"  PASS")
        else:
            total_errors += len(errors)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print(f"\nAll fixtures PASS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
