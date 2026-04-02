#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=ellipsoid_slice_fixtures"
"""QA Ellipsoid Slice Cert family [169] — certifies QA-style slicing of the
WGS84 quantum ellipse into latitude circles, meridian ellipses, and
chromogeometric curve families.

TIER 1+2 — EXACT REFORMULATION + STRUCTURAL:

  LATITUDE SLICES (circles):
    At spread s_φ, the latitude circle has:
      R² = N² · c_φ = a² · d² · c_φ / (d² - e²·s_φ)
    where c_φ = 1 - s_φ. All rational in spreads + QN components.
    Every latitude slice is a CIRCLE (I = 0 in 2D = degenerate conic).

  MERIDIAN SLICES (ellipses):
    Every meridian is an ellipse with:
      semi-major = a_earth, semi-minor = a_earth · √F / d
      axis ratio = √F/d = √12019/110
      eccentricity² = 1 - F/d² = e²/d² = 81/12100
    Same QN as the shape itself — SELF-SIMILAR.

  CHROMOGEOMETRIC SLICES — three families of curves:
    Given a point (s_φ, s_λ) on the ellipsoid, its QA direction has:
      C = 2de = constant green quadrance curves (area contours)
      F = d²-e² = constant red quadrance curves (Minkowski contours)
      G = d²+e² = constant blue quadrance curves (Euclidean contours)
    C-slices, F-slices, G-slices form three orthogonal families
    satisfying C² + F² = G² at every point.

  PISANO LATITUDE BANDS:
    π(24) = 24 → the mod-24 T-operator divides the longitude into 24 bands.
    Each band spans 360°/24 = 15° of longitude.
    The 24-fold partition = the 24 hours of the day = the mod-24 compass.

SOURCE: WGS84; cert [156]; cert [140] conic discriminant;
        cert [125] chromogeometry.

Checks
------
SL_1         schema_version == 'QA_ELLIPSOID_SLICE_CERT.v1'
SL_LAT       R² = a²·d²·c_φ/(d²-e²·s_φ) verified at witness latitudes
SL_MER       meridian axis ratio = √F/d verified
SL_CHROMO    C²+F²=G² for witness chromogeometric slices
SL_BAND      24-fold Pisano partition verified (360/24 = 15°)
SL_W         at least 3 slice witnesses
SL_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_ELLIPSOID_SLICE_CERT.v1"

WGS84_A = 6378137.0


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("SL_1", f"schema_version must be {SCHEMA}")

    qn = cert.get("shape_qn", {})
    b = qn.get("b", 0)
    e = qn.get("e", 0)
    d = qn.get("d", 0)
    a_qn = qn.get("a", 0)
    F = d * d - e * e
    G = d * d + e * e
    C = 2 * d * e

    # SL_LAT — latitude circle quadrances
    lat_slices = cert.get("latitude_slices", [])
    for i, sl in enumerate(lat_slices):
        s_phi = sl.get("s_phi")
        if s_phi is None:
            lat = sl.get("lat_deg")
            if lat is not None:
                s_phi = math.sin(math.radians(lat)) * math.sin(math.radians(lat))
        if s_phi is None:
            continue

        c_phi = 1.0 - s_phi
        # R² = a² · d² · c_φ / (d² - e²·s_φ)
        denom = d * d - e * e * s_phi
        computed_R_sq = WGS84_A * WGS84_A * d * d * c_phi / denom

        declared_R_sq = sl.get("R_squared")
        tol = sl.get("tolerance", 1e6)  # in m²
        if declared_R_sq is not None:
            if abs(declared_R_sq - computed_R_sq) > tol:
                err("SL_LAT", f"slice {i}: R²={declared_R_sq:.0f} "
                    f"!= computed {computed_R_sq:.0f}")

        declared_R = sl.get("R_meters")
        if declared_R is not None:
            computed_R = math.sqrt(computed_R_sq)
            if abs(declared_R - computed_R) > 1.0:
                err("SL_LAT", f"slice {i}: R={declared_R:.1f} "
                    f"!= computed {computed_R:.1f}")

    # SL_MER — meridian ellipse
    meridian = cert.get("meridian_ellipse")
    if meridian is not None:
        declared_ratio = meridian.get("axis_ratio_ba")
        if declared_ratio is not None:
            computed = math.sqrt(F) / d
            tol = meridian.get("tolerance", 1e-6)
            if abs(declared_ratio - computed) > tol:
                err("SL_MER", f"axis ratio={declared_ratio} != √F/d={computed:.8f}")

        declared_ecc_sq = meridian.get("eccentricity_sq")
        if declared_ecc_sq is not None:
            computed_ecc = (e * e) / (d * d)
            if abs(declared_ecc_sq - computed_ecc) > 1e-8:
                err("SL_MER", f"e²={declared_ecc_sq} != e_QN²/d_QN²={computed_ecc:.8f}")

    # SL_CHROMO — chromogeometric slices
    chromo_slices = cert.get("chromo_slices", [])
    for j, cs in enumerate(chromo_slices):
        d_dir = cs.get("d_dir")
        e_dir = cs.get("e_dir")
        if d_dir is not None and e_dir is not None:
            C_val = 2 * d_dir * e_dir
            F_val = d_dir * d_dir - e_dir * e_dir
            G_val = d_dir * d_dir + e_dir * e_dir

            declared = cs.get("C"), cs.get("F"), cs.get("G")
            computed = C_val, F_val, G_val
            for name, dec, comp in zip(["C", "F", "G"], declared, computed):
                if dec is not None and dec != comp:
                    err("SL_CHROMO", f"chromo {j}: {name}={dec} != {comp}")

            if C_val * C_val + F_val * F_val != G_val * G_val:
                err("SL_CHROMO", f"chromo {j}: C²+F²={C_val*C_val+F_val*F_val} != G²={G_val*G_val}")

    # SL_BAND — Pisano bands
    band_data = cert.get("pisano_bands")
    if band_data is not None:
        n_bands = band_data.get("n_bands")
        band_width = band_data.get("band_width_deg")
        if n_bands is not None and band_width is not None:
            if abs(n_bands * band_width - 360.0) > 0.01:
                err("SL_BAND", f"{n_bands} × {band_width}° = {n_bands*band_width}° != 360°")

    # SL_W
    total = len(lat_slices) + len(chromo_slices)
    if total < 3:
        err("SL_W", f"need >= 3 total slice witnesses, got {total}")

    # SL_F
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("SL_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("SL_F: declared FAIL but no fail_ledger and all checks pass")

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
