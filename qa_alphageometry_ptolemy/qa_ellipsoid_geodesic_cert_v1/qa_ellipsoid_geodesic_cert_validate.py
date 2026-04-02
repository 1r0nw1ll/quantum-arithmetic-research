#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=ellipsoid_geodesic_fixtures"
"""QA Ellipsoid Geodesic Cert family [168] — certifies geodesic properties of
the WGS84 quantum ellipse in QN component arithmetic.

TIER 1 — EXACT REFORMULATION:

  Earth shape QN: (b, e, d, a) = (101, 9, 110, 119)
  Triple: (C, F, G) = (1980, 12019, 12181), C² + F² = G²

  CURVATURE RADII in QN components:
    N (prime vertical) = a_earth · d / √(d² - e²·s_φ)
    N² = a_earth² · d² / (d² - e²·s_φ)

    M (meridional) = a_earth · F / (d² - e²·s_φ)^(3/2) · d
    M/N = F / (d² - e²·s_φ)    (EXACT in QN components)

    At pole (s_φ=1): M/N = F/(d²-e²) = F/F = 1 (sphere-like)
    At equator (s_φ=0): M/N = F/d² = 12019/12100 = 0.9933...

  AXIS RATIO:
    b/a = √(1-e²_ell) = √(F/d²) = √F / d = √12019 / 110

  DISCRIMINANT (cert [140]):
    I = C - F = 1980 - 12019 = -10039 < 0  → ELLIPSE

  QUANTUM LATTICE POINTS — spreads where s_φ = p/q with q|QN:
    s_φ = 0 → equator
    s_φ = 1 → pole
    s_φ = e²/G = 81/12181 → eccentricity resonance (lat ≈ 4.68°)
    s_φ = C/G = 1980/12181 → green/blue ratio (lat ≈ 23.78° ≈ Tropic!)
    s_φ = F/G = 12019/12181 → red/blue ratio (lat ≈ 83.38°)

SOURCE: WGS84 (NIMA TR 8350.2); cert [156] QA WGS84 Ellipse;
        cert [140] QA Conic Discriminant; Wildberger (2005).

Checks
------
EG_1         schema_version == 'QA_ELLIPSOID_GEODESIC_CERT.v1'
EG_QN        QN tuple satisfies d=b+e, a=b+2e, C²+F²=G²
EG_CURV      M/N = F/(d²-e²·s_φ) verified at witness latitudes
EG_AXIS      b/a = √F/d verified against WGS84
EG_DISC      I = C-F < 0 (ellipse classification)
EG_LATTICE   quantum lattice points verified
EG_W         at least 3 latitude witnesses
EG_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_ELLIPSOID_GEODESIC_CERT.v1"

# WGS84
WGS84_A = 6378137.0
WGS84_E_SQ = 0.00669437999014


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # EG_1
    if cert.get("schema_version") != SCHEMA:
        err("EG_1", f"schema_version must be {SCHEMA}")

    qn = cert.get("shape_qn", {})
    b = qn.get("b", 0)
    e = qn.get("e", 0)
    d = qn.get("d", 0)
    a = qn.get("a", 0)

    # EG_QN — tuple validity
    if d != b + e:
        err("EG_QN", f"d={d} != b+e={b+e}")
    if a != b + 2 * e:
        err("EG_QN", f"a={a} != b+2e={b+2*e}")

    C = 2 * d * e
    F = d * d - e * e
    G = d * d + e * e

    declared_C = qn.get("C")
    declared_F = qn.get("F")
    declared_G = qn.get("G")
    if declared_C is not None and declared_C != C:
        err("EG_QN", f"C={declared_C} != 2de={C}")
    if declared_F is not None and declared_F != F:
        err("EG_QN", f"F={declared_F} != d²-e²={F}")
    if declared_G is not None and declared_G != G:
        err("EG_QN", f"G={declared_G} != d²+e²={G}")
    if C * C + F * F != G * G:
        err("EG_QN", f"C²+F²={C*C+F*F} != G²={G*G}")

    # EG_DISC — discriminant
    I_val = C - F
    declared_I = cert.get("discriminant_I")
    if declared_I is not None and declared_I != I_val:
        err("EG_DISC", f"I={declared_I} != C-F={I_val}")
    if I_val >= 0:
        err("EG_DISC", f"I={I_val} >= 0, not an ellipse")

    # EG_AXIS — axis ratio
    declared_axis = cert.get("axis_ratio_ba")
    if declared_axis is not None:
        computed = math.sqrt(F) / d
        tol = cert.get("axis_tolerance", 1e-6)
        if abs(declared_axis - computed) > tol:
            err("EG_AXIS", f"b/a={declared_axis} != √F/d={computed:.8f}")

    # EG_CURV — curvature witnesses
    witnesses = cert.get("witnesses", [])
    for i, w in enumerate(witnesses):
        lat = w.get("lat_deg")
        s_phi = w.get("s_phi")
        tol = w.get("tolerance", 1e-6)

        if s_phi is None and lat is not None:
            s_phi = math.sin(math.radians(lat)) * math.sin(math.radians(lat))

        if s_phi is None:
            err("EG_CURV", f"witness {i}: no s_phi or lat_deg")
            continue

        # M/N = F / (d² - e²·s_φ)
        denom = d * d - e * e * s_phi
        computed_ratio = F / denom

        declared_ratio = w.get("curvature_ratio_MN")
        if declared_ratio is not None:
            if abs(declared_ratio - computed_ratio) > tol:
                err("EG_CURV", f"witness {i}: M/N={declared_ratio} "
                    f"!= F/(d²-e²s_φ)={computed_ratio:.8f}")

        # N value
        declared_N = w.get("N_meters")
        if declared_N is not None:
            e_sq_ell = (e * e) / (d * d)
            computed_N = WGS84_A / math.sqrt(1 - e_sq_ell * s_phi)
            if abs(declared_N - computed_N) > 1.0:  # within 1m
                err("EG_CURV", f"witness {i}: N={declared_N} != {computed_N:.1f}")

    # EG_LATTICE — quantum lattice points
    lattice = cert.get("quantum_lattice_points", [])
    for j, lp in enumerate(lattice):
        num = lp.get("s_num")
        den = lp.get("s_den")
        declared_lat = lp.get("lat_deg")
        tol_lat = lp.get("lat_tolerance", 0.01)

        if num is not None and den is not None and den != 0:
            s = num / den
            if 0 <= s <= 1:
                computed_lat = math.degrees(math.asin(math.sqrt(s)))
                if declared_lat is not None:
                    if abs(declared_lat - computed_lat) > tol_lat:
                        err("EG_LATTICE", f"lattice {j}: lat={declared_lat} "
                            f"!= arcsin(√({num}/{den}))={computed_lat:.4f}")

    # EG_W
    if len(witnesses) < 3:
        err("EG_W", f"need >= 3 witnesses, got {len(witnesses)}")

    # EG_F
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("EG_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("EG_F: declared FAIL but no fail_ledger and all checks pass")

    return {"ok": not has_errors, "errors": errors, "warnings": warnings, "schema": SCHEMA}


def self_test():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    results = {"pass_count": 0, "fail_count": 0, "errors": []}
    for fname in sorted(os.listdir(fixture_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(fixture_dir, fname), "r", encoding="utf-8") as f:
            cert = json.load(f)
        out = validate(cert)
        declared = cert.get("result", "UNKNOWN")
        if declared == "PASS" and out["ok"]:
            results["pass_count"] += 1
        elif declared == "FAIL" and not out["ok"]:
            results["fail_count"] += 1
        else:
            results["errors"].append({"fixture": fname, "declared": declared,
                                       "validator_ok": out["ok"], "issues": out["errors"]})
    results["ok"] = len(results["errors"]) == 0
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("cert_file", nargs="?")
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
