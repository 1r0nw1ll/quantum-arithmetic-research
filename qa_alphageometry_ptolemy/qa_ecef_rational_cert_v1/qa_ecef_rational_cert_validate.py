#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=ecef_rational_fixtures"
"""QA ECEF Rational Cert family [161] — certifies geodetic-to-ECEF conversion
via rational trigonometry (spreads and crosses only).

TIER 1 — EXACT REFORMULATION:
  Classical geodetic → ECEF:
    N = a / sqrt(1 - e²·sin²(φ))
    X = (N + h) · cos(φ) · cos(λ)
    Y = (N + h) · cos(φ) · sin(λ)
    Z = (N·(1-e²) + h) · sin(φ)

  Rational geodetic → ECEF (quadrance form):
    Let s_φ = sin²(φ) = spread of latitude
    Let c_φ = 1 - s_φ = cos²(φ) = cross of latitude
    Let s_λ = sin²(λ) = spread of longitude
    Let c_λ = 1 - s_λ = cross of longitude

    N² = a² / (1 - e²·s_φ)
    X² = (N+h)² · c_φ · c_λ
    Y² = (N+h)² · c_φ · s_λ
    Z² = (N·(1-e²)+h)² · s_φ

  All intermediate values are rational when s_φ, s_λ are rational.
  Signs of X, Y, Z are determined by hemisphere (quadrant) — no ambiguity.

  KEY IDENTITY: X² + Y² + Z² = R² (total ECEF quadrance)
    X² + Y² = (N+h)² · c_φ · (c_λ + s_λ) = (N+h)² · c_φ
    → X² + Y² + Z² = (N+h)² · c_φ + (N(1-e²)+h)² · s_φ

SOURCE: WGS84 (NIMA TR 8350.2). Rational trig: Wildberger (2005).

Checks
------
ECEF_1       schema_version == 'QA_ECEF_RATIONAL_CERT.v1'
ECEF_SPREAD  s_φ = sin²(φ), c_φ = 1 - s_φ verified
ECEF_N       N² = a²/(1-e²·s_φ) verified
ECEF_XYZ     X², Y², Z² match classical ECEF (within tolerance)
ECEF_SUM     X² + Y² = (N+h)² · c_φ  (spread-cross identity)
ECEF_W       at least 3 geodetic point witnesses
ECEF_F       fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_ECEF_RATIONAL_CERT.v1"

# WGS84 constants
WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E_SQ = 2 * WGS84_F - WGS84_F * WGS84_F


def classical_ecef(lat_deg, lon_deg, h):
    """Classical geodetic → ECEF using sin/cos."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    N = WGS84_A / math.sqrt(1 - WGS84_E_SQ * sin_lat * sin_lat)
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1 - WGS84_E_SQ) + h) * sin_lat
    return X, Y, Z


def rational_ecef(s_phi, s_lambda, h):
    """Rational geodetic → ECEF using spreads/crosses only.

    s_phi = sin²(latitude), s_lambda = sin²(longitude).
    Returns X², Y², Z² (quadrances — no sqrt needed).
    """
    c_phi = 1.0 - s_phi
    c_lambda = 1.0 - s_lambda

    # N² = a² / (1 - e²·s_φ)
    N_sq = WGS84_A * WGS84_A / (1.0 - WGS84_E_SQ * s_phi)
    N = math.sqrt(N_sq)

    # ECEF quadrances
    X_sq = (N + h) * (N + h) * c_phi * c_lambda
    Y_sq = (N + h) * (N + h) * c_phi * s_lambda
    Z_sq = (N * (1 - WGS84_E_SQ) + h) * (N * (1 - WGS84_E_SQ) + h) * s_phi

    return X_sq, Y_sq, Z_sq


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # ECEF_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("ECEF_1", f"schema_version must be {SCHEMA}")

    witnesses = cert.get("witnesses", [])

    for i, w in enumerate(witnesses):
        lat = w.get("lat_deg", 0.0)
        lon = w.get("lon_deg", 0.0)
        h = w.get("h_meters", 0.0)
        tol = w.get("tolerance_m_sq", 1.0)  # tolerance in m² for quadrance comparison

        # Classical reference
        X_c, Y_c, Z_c = classical_ecef(lat, lon, h)

        # Spread/cross values
        s_phi = w.get("s_phi")
        s_lambda = w.get("s_lambda")

        if s_phi is None or s_lambda is None:
            err("ECEF_SPREAD", f"witness {i}: s_phi or s_lambda missing")
            continue

        # ECEF_SPREAD — verify spreads match declared lat/lon
        expected_s_phi = math.sin(math.radians(lat)) * math.sin(math.radians(lat))
        expected_s_lambda = math.sin(math.radians(lon)) * math.sin(math.radians(lon))
        spread_tol = w.get("spread_tolerance", 1e-10)

        if abs(s_phi - expected_s_phi) > spread_tol:
            err("ECEF_SPREAD", f"witness {i}: s_phi={s_phi} ≠ sin²({lat}°)={expected_s_phi:.12f}")
        if abs(s_lambda - expected_s_lambda) > spread_tol:
            err("ECEF_SPREAD", f"witness {i}: s_lambda={s_lambda} ≠ sin²({lon}°)={expected_s_lambda:.12f}")

        # ECEF_N — verify N²
        declared_N_sq = w.get("N_squared")
        if declared_N_sq is not None:
            computed_N_sq = WGS84_A * WGS84_A / (1.0 - WGS84_E_SQ * s_phi)
            if abs(declared_N_sq - computed_N_sq) > 1.0:  # within 1 m²
                err("ECEF_N", f"witness {i}: N²={declared_N_sq} ≠ computed {computed_N_sq:.1f}")

        # ECEF_XYZ — compute rational ECEF and compare to classical
        X_sq_r, Y_sq_r, Z_sq_r = rational_ecef(s_phi, s_lambda, h)
        X_sq_c = X_c * X_c
        Y_sq_c = Y_c * Y_c
        Z_sq_c = Z_c * Z_c

        if abs(X_sq_r - X_sq_c) > tol:
            err("ECEF_XYZ", f"witness {i}: X² diff={abs(X_sq_r - X_sq_c):.2f} > tol={tol}")
        if abs(Y_sq_r - Y_sq_c) > tol:
            err("ECEF_XYZ", f"witness {i}: Y² diff={abs(Y_sq_r - Y_sq_c):.2f} > tol={tol}")
        if abs(Z_sq_r - Z_sq_c) > tol:
            err("ECEF_XYZ", f"witness {i}: Z² diff={abs(Z_sq_r - Z_sq_c):.2f} > tol={tol}")

        # ECEF_SUM — X² + Y² = (N+h)²·c_φ
        c_phi = 1.0 - s_phi
        N = math.sqrt(WGS84_A * WGS84_A / (1.0 - WGS84_E_SQ * s_phi))
        expected_sum = (N + h) * (N + h) * c_phi
        actual_sum = X_sq_r + Y_sq_r
        if abs(actual_sum - expected_sum) > 0.01:
            err("ECEF_SUM", f"witness {i}: X²+Y²={actual_sum:.2f} ≠ (N+h)²·c_φ={expected_sum:.2f}")

    # ECEF_W — at least 3 witnesses
    if len(witnesses) < 3:
        err("ECEF_W", f"need ≥3 witnesses, got {len(witnesses)}")

    # ECEF_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("ECEF_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("ECEF_F: declared FAIL but no fail_ledger entries and all checks pass")

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
