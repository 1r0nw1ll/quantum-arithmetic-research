#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=wgs84_ellipse_fixtures"
"""QA WGS84 Ellipse Cert family [156] — certifies that the WGS84 reference
ellipsoid (Earth's oblate spheroid) IS a QA quantum ellipse.

TIER 1 — EXACT REFORMULATION:
  Earth shape QN = (101, 9, 110, 119)
    eccentricity = e/d = 9/110 = 0.081818...
    WGS84 first eccentricity = 0.081819...  (0.001% error)
    axis ratio = sqrt(ab)/d = sqrt(12019)/110 = 0.996647...
    WGS84 b/a = 0.996647...  (7 significant figures)

  Earth orbit QN = (59, 1, 60, 61)
    eccentricity = e/d = 1/60 = 0.016667...
    orbital eccentricity = 0.016709...  (0.25% error)

  Triple (C,F,G) = (1980, 12019, 12181)
  C^2 + F^2 = G^2 (Pythagorean identity)

SOURCE: WGS84 parameters (NIMA Technical Report 8350.2, 2000).
Quantum ellipse: Ben Iverson (Pyth-1). QN fitting: Will Dale 2026-04-01.

Checks
------
WGS_1       schema_version == 'QA_WGS84_ELLIPSE_CERT.v1'
WGS_QN      QN (b,e,d,a) satisfies b+e=d, b+2e=a, gcd(b,e)=1
WGS_TRIPLE  F=ab, C=2de, G=d*d+e*e, C*C+F*F=G*G (S1 compliant: d*d not d-squared)
WGS_ECC     |e/d - wgs84_ecc| / wgs84_ecc < declared tolerance
WGS_AXIS    |sqrt(ab)/d - wgs84_ratio| / wgs84_ratio < declared tolerance
WGS_ORBIT   orbit QN declared with eccentricity match
WGS_W       at least 1 witness (shape or orbit)
WGS_F       fail detection
"""

import json
import math
import os
import sys
from math import gcd

SCHEMA = "QA_WGS84_ELLIPSE_CERT.v1"

# WGS84 reference values (NIMA TR 8350.2)
WGS84_A = 6378137.0          # equatorial semi-major axis (m)
WGS84_B = 6356752.314245     # polar semi-minor axis (m)
WGS84_F = 1 / 298.257223563  # flattening
WGS84_E_SQ = 2 * WGS84_F - WGS84_F * WGS84_F
WGS84_ECC = math.sqrt(WGS84_E_SQ)       # 0.0818191908...
WGS84_RATIO = WGS84_B / WGS84_A          # 0.9966471893...

# Earth orbital eccentricity (J2000.0)
ORBIT_ECC = 0.0167086


def validate_qn(b, e, d, a):
    """Validate QN tuple structure. Returns list of errors."""
    errors = []
    if b + e != d:
        errors.append(f"b+e={b+e} != d={d}")
    if b + 2 * e != a:
        errors.append(f"b+2e={b+2*e} != a={a}")
    if gcd(abs(b), abs(e)) != 1:
        errors.append(f"gcd(b,e)=gcd({b},{e})={gcd(abs(b),abs(e))} != 1")
    if b <= 0 or e <= 0:
        errors.append(f"b={b}, e={e} must both be positive")
    return errors


def validate_triple(b, e, d, a):
    """Validate (C,F,G) triple. Returns (C,F,G, errors)."""
    errors = []
    F = a * b
    C = 2 * d * e
    G = d * d + e * e  # S1: d*d not d**2
    if C * C + F * F != G * G:
        errors.append(f"C*C+F*F={C*C+F*F} != G*G={G*G}")
    return C, F, G, errors


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # WGS_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("WGS_1", f"schema_version must be {SCHEMA}")

    # WGS_QN — shape QN validation
    shape = cert.get("shape_qn", {})
    if not shape:
        err("WGS_QN", "shape_qn section missing")
    else:
        b = shape.get("b", 0)
        e = shape.get("e", 0)
        d = shape.get("d", 0)
        a = shape.get("a", 0)
        qn_errors = validate_qn(b, e, d, a)
        for qe in qn_errors:
            err("WGS_QN", f"shape QN ({b},{e},{d},{a}): {qe}")

        # WGS_TRIPLE — (C,F,G) triple
        triple = cert.get("shape_triple", {})
        C, F, G, triple_errors = validate_triple(b, e, d, a)
        for te in triple_errors:
            err("WGS_TRIPLE", te)

        if triple:
            if triple.get("C") is not None and triple["C"] != C:
                err("WGS_TRIPLE", f"declared C={triple['C']} != computed C={C}")
            if triple.get("F") is not None and triple["F"] != F:
                err("WGS_TRIPLE", f"declared F={triple['F']} != computed F={F}")
            if triple.get("G") is not None and triple["G"] != G:
                err("WGS_TRIPLE", f"declared G={triple['G']} != computed G={G}")

        # WGS_ECC — eccentricity match
        ecc_qa = e / d if d > 0 else 0.0
        tol_ecc = cert.get("tolerance_ecc", 0.001)
        rel_err_ecc = abs(ecc_qa - WGS84_ECC) / WGS84_ECC
        if rel_err_ecc > tol_ecc:
            err("WGS_ECC", f"e/d={ecc_qa:.10f} vs WGS84 {WGS84_ECC:.10f}: "
                f"relative error {rel_err_ecc:.6f} > tolerance {tol_ecc}")

        # WGS_AXIS — axis ratio match
        axis_qa = math.sqrt(a * b) / d if d > 0 else 0.0
        tol_axis = cert.get("tolerance_axis", 1e-6)
        rel_err_axis = abs(axis_qa - WGS84_RATIO) / WGS84_RATIO
        if rel_err_axis > tol_axis:
            err("WGS_AXIS", f"sqrt(ab)/d={axis_qa:.10f} vs WGS84 b/a={WGS84_RATIO:.10f}: "
                f"relative error {rel_err_axis:.2e} > tolerance {tol_axis}")

    # WGS_ORBIT — orbit QN validation
    orbit = cert.get("orbit_qn", {})
    if orbit:
        ob = orbit.get("b", 0)
        oe = orbit.get("e", 0)
        od = orbit.get("d", 0)
        oa = orbit.get("a", 0)
        oqn_errors = validate_qn(ob, oe, od, oa)
        for oqe in oqn_errors:
            err("WGS_ORBIT", f"orbit QN ({ob},{oe},{od},{oa}): {oqe}")

        ecc_orbit_qa = oe / od if od > 0 else 0.0
        tol_orbit = cert.get("tolerance_orbit_ecc", 0.01)
        rel_err_orbit = abs(ecc_orbit_qa - ORBIT_ECC) / ORBIT_ECC
        if rel_err_orbit > tol_orbit:
            err("WGS_ORBIT", f"e/d={ecc_orbit_qa:.10f} vs orbital {ORBIT_ECC:.10f}: "
                f"relative error {rel_err_orbit:.6f} > tolerance {tol_orbit}")
    else:
        warnings.append("WGS_ORBIT: no orbit_qn section")

    # WGS_W — at least one witness
    has_shape = bool(cert.get("shape_qn"))
    has_orbit = bool(cert.get("orbit_qn"))
    if not has_shape and not has_orbit:
        err("WGS_W", "need at least one QN witness (shape or orbit)")

    # WGS_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("WGS_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("WGS_F: declared FAIL but no fail_ledger entries and all checks pass")

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
