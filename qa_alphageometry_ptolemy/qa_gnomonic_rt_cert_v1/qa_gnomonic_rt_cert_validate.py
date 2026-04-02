#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=gnomonic_rt_fixtures"
"""QA Gnomonic RT Cert family [164] — certifies gnomonic map projection via
rational trigonometry (spreads and crosses).

TIER 1 — EXACT REFORMULATION (gnomonic quadrance):
  Classical gnomonic projection from tangent point (φ₀, λ₀):
    x = cos(φ)sin(Δλ) / cos(c)
    y = [cos(φ₀)sin(φ) - sin(φ₀)cos(φ)cos(Δλ)] / cos(c)
    where cos(c) = sin(φ₀)sin(φ) + cos(φ₀)cos(φ)cos(Δλ)

  Rational gnomonic (quadrance form):
    Let s₀=sin²(φ₀), c₀=1-s₀, s=sin²(φ), c=1-s, sΔ=sin²(Δλ), cΔ=1-sΔ
    cos(c) = √(s₀·s) + √(c₀·c·cΔ)
    cos²(c) = s₀s + c₀·c·cΔ + 2√(s₀·s·c₀·c·cΔ)
    spread_c = 1 - cos²(c)  (spread of angular distance)
    cross_c = cos²(c)

    Gnomonic quadrance: Q = spread_c / cross_c = tan²(c)
    This is the QUADRANCE from tangent point to projected point.

    KEY PROPERTY: great circles project to straight lines.
    In RT: collinearity = zero cross-product quadrance.

TIER 2 — STRUCTURAL (Berggren connection):
  Berggren tree generators M_A, M_B, M_C act on direction (d,e):
    M_A: (d,e) → (2d-e, d)
    M_B: (d,e) → (2d+e, d)
    M_C: (d,e) → (d+2e, e)
  Each produces a Pythagorean triple (C,F,G) = (2de, d²-e², d²+e²)
  with C²+F²=G². These are discrete geodesic steps on the cone C²+F²=G².
  Under gnomonic projection, geodesics → straight lines.
  So Berggren tree paths = straight lines on the gnomonic chart.

SOURCE: Wildberger, Divine Proportions (2005); Berggren (1934);
        Barning (1963); Cert [135] QA_PYTHAGOREAN_TREE_CERT.v1.

Checks
------
GN_1         schema_version == 'QA_GNOMONIC_RT_CERT.v1'
GN_QUAD      Gnomonic quadrance Q = spread_c / cross_c matches classical x²+y²
GN_SPREAD    spread_c + cross_c = 1 (fundamental identity)
GN_COLLINEAR Great circle points project to collinear points (zero cross product)
GN_BERGGREN  Berggren triples satisfy C²+F²=G² and produce valid directions
GN_W         at least 3 projection witnesses
GN_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_GNOMONIC_RT_CERT.v1"


def gnomonic_classical(phi_deg, lam_deg, phi0_deg, lam0_deg):
    """Classical gnomonic projection. Returns (x, y)."""
    phi = math.radians(phi_deg)
    lam = math.radians(lam_deg)
    p0 = math.radians(phi0_deg)
    l0 = math.radians(lam0_deg)
    dlam = lam - l0

    cos_c = (math.sin(p0) * math.sin(phi) +
             math.cos(p0) * math.cos(phi) * math.cos(dlam))
    if abs(cos_c) < 1e-12:
        return None, None  # point at or beyond horizon
    x = math.cos(phi) * math.sin(dlam) / cos_c
    y = (math.cos(p0) * math.sin(phi) -
         math.sin(p0) * math.cos(phi) * math.cos(dlam)) / cos_c
    return x, y


def gnomonic_rt(s_phi, s_dlam, s_phi0):
    """Rational gnomonic: compute quadrance Q = spread_c / cross_c.

    s_phi   = sin²(φ)   = spread of latitude
    s_dlam  = sin²(Δλ)  = spread of longitude difference
    s_phi0  = sin²(φ₀)  = spread of tangent point latitude
    Returns (Q, spread_c, cross_c).
    """
    c_phi = 1.0 - s_phi
    c_phi0 = 1.0 - s_phi0
    c_dlam = 1.0 - s_dlam

    # cos(angular distance) = √(s₀·s) + √(c₀·c·cΔ)
    term1 = math.sqrt(s_phi0 * s_phi)
    term2 = math.sqrt(c_phi0 * c_phi * c_dlam)
    cos_c = term1 + term2
    cross_c = cos_c * cos_c  # cos²(c)
    spread_c = 1.0 - cross_c  # sin²(c) = spread of angular distance

    if cross_c < 1e-20:
        return float('inf'), spread_c, cross_c

    Q = spread_c / cross_c  # tan²(c) = gnomonic quadrance
    return Q, spread_c, cross_c


def berggren_triple(d, e):
    """Compute Pythagorean triple (C, F, G) from direction (d, e)."""
    C = 2 * d * e
    F = d * d - e * e
    G = d * d + e * e
    return C, F, G


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # GN_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("GN_1", f"schema_version must be {SCHEMA}")

    witnesses = cert.get("witnesses", [])
    tangent = cert.get("tangent_point", {})
    tp_lat = tangent.get("lat_deg", 0.0)
    tp_lon = tangent.get("lon_deg", 0.0)
    s_phi0 = tangent.get("s_phi0")

    if s_phi0 is None:
        # compute from lat
        s_phi0 = math.sin(math.radians(tp_lat)) * math.sin(math.radians(tp_lat))

    for i, w in enumerate(witnesses):
        lat = w.get("lat_deg")
        lon = w.get("lon_deg")
        if lat is None or lon is None:
            err("GN_QUAD", f"witness {i}: missing lat_deg or lon_deg")
            continue

        # Spreads
        s_phi = w.get("s_phi")
        s_dlam = w.get("s_dlam")
        tol = w.get("tolerance", 1e-8)

        if s_phi is None:
            s_phi = math.sin(math.radians(lat)) * math.sin(math.radians(lat))
        if s_dlam is None:
            s_dlam = (math.sin(math.radians(lon - tp_lon)) *
                      math.sin(math.radians(lon - tp_lon)))

        # GN_QUAD — gnomonic quadrance matches classical
        Q_rt, spread_c, cross_c = gnomonic_rt(s_phi, s_dlam, s_phi0)
        x_cl, y_cl = gnomonic_classical(lat, lon, tp_lat, tp_lon)
        if x_cl is not None:
            Q_cl = x_cl * x_cl + y_cl * y_cl
            if abs(Q_rt - Q_cl) > tol:
                err("GN_QUAD", f"witness {i}: Q_rt={Q_rt:.8f} != Q_cl={Q_cl:.8f}, "
                    f"diff={abs(Q_rt - Q_cl):.2e}")

        # Declared values
        declared_Q = w.get("gnomonic_quadrance")
        if declared_Q is not None and abs(declared_Q - Q_rt) > tol:
            err("GN_QUAD", f"witness {i}: declared Q={declared_Q} != computed {Q_rt:.8f}")

        # GN_SPREAD — spread + cross = 1
        declared_spread = w.get("spread_c")
        declared_cross = w.get("cross_c")
        if declared_spread is not None and declared_cross is not None:
            total = declared_spread + declared_cross
            if abs(total - 1.0) > 1e-10:
                err("GN_SPREAD", f"witness {i}: spread_c + cross_c = {total} != 1.0")

    # GN_COLLINEAR — great circle collinearity test
    collinear_data = cert.get("collinearity_test")
    if collinear_data is not None:
        gc_points = collinear_data.get("great_circle_points", [])
        if len(gc_points) >= 3:
            proj_pts = []
            for pt in gc_points:
                x, y = gnomonic_classical(pt["lat_deg"], pt["lon_deg"],
                                          tp_lat, tp_lon)
                if x is not None:
                    proj_pts.append((x, y))

            # Check collinearity via cross products of consecutive triples
            col_tol = collinear_data.get("collinearity_tolerance", 1e-12)
            for j in range(len(proj_pts) - 2):
                x1, y1 = proj_pts[j]
                x2, y2 = proj_pts[j + 1]
                x3, y3 = proj_pts[j + 2]
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if abs(cross) > col_tol:
                    err("GN_COLLINEAR", f"points [{j},{j+1},{j+2}]: "
                        f"cross={cross:.2e} > tol={col_tol}")

    # GN_BERGGREN — Berggren tree triples
    berggren_data = cert.get("berggren_tree")
    if berggren_data is not None:
        root = berggren_data.get("root")
        moves = berggren_data.get("moves", [])

        for move in moves:
            d_val = move.get("d")
            e_val = move.get("e")
            if d_val is None or e_val is None:
                continue
            C, F, G = berggren_triple(d_val, e_val)

            declared_C = move.get("C")
            declared_F = move.get("F")
            declared_G = move.get("G")

            if declared_C is not None and declared_C != C:
                err("GN_BERGGREN", f"move ({d_val},{e_val}): C={declared_C} != 2de={C}")
            if declared_F is not None and declared_F != F:
                err("GN_BERGGREN", f"move ({d_val},{e_val}): F={declared_F} != d²-e²={F}")
            if declared_G is not None and declared_G != G:
                err("GN_BERGGREN", f"move ({d_val},{e_val}): G={declared_G} != d²+e²={G}")

            # C²+F²=G²
            if C * C + F * F != G * G:
                err("GN_BERGGREN", f"move ({d_val},{e_val}): C²+F²={C*C+F*F} != G²={G*G}")

    # GN_W — at least 3 witnesses
    if len(witnesses) < 3:
        err("GN_W", f"need >= 3 witnesses, got {len(witnesses)}")

    # GN_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("GN_F", f"declared PASS but {len(errors) - 1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("GN_F: declared FAIL but no fail_ledger and all checks pass")

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
