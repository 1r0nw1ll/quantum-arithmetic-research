#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=historical_nav_fixtures"
"""QA Historical Nav Cert family [167] — certifies structural correspondence
between historical navigation systems and QA integer arithmetic.

TIER 2 — STRUCTURAL CORRESPONDENCE:

  Five historical navigation systems, each using integer-ratio methods:

  1. BABYLON (~1800 BCE): Plimpton 322
     - Table of Pythagorean triples (C, F, G) = Berggren tree nodes
     - Each row = direction (d, e) + quadrance G = d² + e²
     - Cert [138] QA_PLIMPTON322_CERT.v1 already proven
     - Navigation interpretation: gnomonic waypoint table

  2. EGYPT (~1600 BCE): Seked system
     - Seked = horizontal run per unit vertical rise = integer ratio
     - Seked of pyramid face = 5.5 palms per cubit (= 5½ : 7)
     - In QA: seked² / (seked² + 1) = spread of slope angle
     - Integer arithmetic → rational spread

  3. POLYNESIA (~1000 CE): Star compass
     - 32 star houses around horizon (named directions)
     - Discrete bearing system ≈ mod-32
     - Stars rise/set at fixed spread from north
     - Dead reckoning by island-to-island "etak" reference

  4. NORSE (~800 CE): Sun stones
     - Calcite crystal (Iceland spar) polarizes sunlight
     - Measures sun position through clouds as spread of polarization
     - Integer bearing from sun position
     - Combined with latitude sailing (constant-latitude = loxodrome)

  5. ARAB (~900 CE): Kamal
     - Wooden card held at arm's length
     - Measures star altitude in finger-widths = INTEGER increments
     - Each finger-width ≈ 1.5° altitude ≈ fixed spread increment
     - Latitude determination by integer measurement

  COMMON STRUCTURE: All systems operate on:
    (a) discrete direction states (integer ratios, named houses, finger-widths)
    (b) integer arithmetic for computation
    (c) observer projection only at measurement and landfall
    This IS QA navigation — Theorem NT before Theorem NT was named.

SOURCE: Plimpton 322 (Mansfield & Wildberger 2017/2021); Polynesian
        navigation (Lewis 1972); Norse sun stones (Ropars et al. 2012);
        Arab kamal (Tibbetts 1971); Egyptian seked (Gillings 1972).

Checks
------
HN_1         schema_version == 'QA_HISTORICAL_NAV_CERT.v1'
HN_SYSTEM    each system has name, era, method, qa_equivalent
HN_SEKED     seked → spread conversion verified (seked²/(seked²+1) = spread)
HN_KAMAL     finger-width → spread conversion verified
HN_TRIPLE    Plimpton 322 triples satisfy C² + F² = G²
HN_W         at least 3 historical system witnesses
HN_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_HISTORICAL_NAV_CERT.v1"


def seked_to_spread(seked_num, seked_den):
    """Convert Egyptian seked (ratio) to spread of slope angle.

    Seked = horizontal_run / vertical_rise
    tan(slope) = rise/run = den/num (inverted)
    spread = sin²(slope) = tan²(slope) / (1 + tan²(slope))
           = (den/num)² / (1 + (den/num)²)
           = den² / (den² + num²)
    """
    return (seked_den * seked_den) / (seked_den * seked_den + seked_num * seked_num)


def finger_to_spread(fingers, finger_deg=1.5):
    """Convert kamal finger-widths to spread.

    Each finger ≈ 1.5° of altitude.
    spread = sin²(fingers * finger_deg)
    """
    angle = fingers * finger_deg
    return math.sin(math.radians(angle)) * math.sin(math.radians(angle))


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # HN_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("HN_1", f"schema_version must be {SCHEMA}")

    systems = cert.get("systems", [])

    for i, sys_data in enumerate(systems):
        # HN_SYSTEM — structural completeness
        name = sys_data.get("name")
        era = sys_data.get("era")
        method = sys_data.get("method")
        qa_equiv = sys_data.get("qa_equivalent")

        if not all([name, era, method, qa_equiv]):
            err("HN_SYSTEM", f"system {i}: missing name, era, method, or qa_equivalent")

        # HN_SEKED — Egyptian seked witnesses
        seked_data = sys_data.get("seked_witnesses")
        if seked_data is not None:
            for j, sw in enumerate(seked_data):
                num = sw.get("seked_num")
                den = sw.get("seked_den", 1)
                declared_spread = sw.get("spread")
                tol = sw.get("tolerance", 1e-6)

                if num is not None and declared_spread is not None:
                    computed = seked_to_spread(num, den)
                    if abs(declared_spread - computed) > tol:
                        err("HN_SEKED", f"system {i} seked {j}: "
                            f"spread={declared_spread} != {den}²/({den}²+{num}²)={computed:.8f}")

        # HN_KAMAL — Arab kamal witnesses
        kamal_data = sys_data.get("kamal_witnesses")
        if kamal_data is not None:
            for j, kw in enumerate(kamal_data):
                fingers = kw.get("fingers")
                declared_spread = kw.get("spread")
                finger_deg = kw.get("finger_deg", 1.5)
                tol = kw.get("tolerance", 1e-4)

                if fingers is not None and declared_spread is not None:
                    computed = finger_to_spread(fingers, finger_deg)
                    if abs(declared_spread - computed) > tol:
                        err("HN_KAMAL", f"system {i} kamal {j}: "
                            f"spread={declared_spread} != sin²({fingers}×{finger_deg}°)={computed:.6f}")

        # HN_TRIPLE — Plimpton 322 triples
        triples = sys_data.get("pythagorean_triples")
        if triples is not None:
            for j, triple in enumerate(triples):
                C = triple.get("C")
                F = triple.get("F")
                G = triple.get("G")
                if C is not None and F is not None and G is not None:
                    if C * C + F * F != G * G:
                        err("HN_TRIPLE", f"system {i} triple {j}: "
                            f"C²+F²={C*C+F*F} != G²={G*G}")

    # HN_W — at least 3 systems
    if len(systems) < 3:
        err("HN_W", f"need >= 3 systems, got {len(systems)}")

    # HN_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("HN_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("HN_F: declared FAIL but no fail_ledger and all checks pass")

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
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test against fixtures")
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
