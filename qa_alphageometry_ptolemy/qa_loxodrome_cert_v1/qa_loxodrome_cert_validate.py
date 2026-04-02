#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=loxodrome_fixtures"
"""QA Loxodrome Cert family [166] — certifies that loxodromes (rhumb lines)
arise from QA T-operator iteration on a mod-m lattice.

TIER 2 — STRUCTURAL:

  A loxodrome (rhumb line) is a path of constant bearing on a sphere.
  On a Mercator chart, loxodromes are straight lines.

  QA discrete loxodrome:
    Fix initial state (b₀, e₀) on mod-m lattice.
    Iterate T-operator: (b,e) → (e, b+e) mod m
    Each step maintains the SAME generator (T) = constant "bearing"
    Period = Pisano period π(m) (the path cycles after π(m) steps)

  Bearing spread:
    For direction (d, e), the spread with the vertical (north):
    s_bearing = e² / (d² + e²) = e² / G
    This is a pure rational quantity.

  Mercator connection:
    The Mercator y-coordinate (isometric latitude):
      ψ = arctanh(sin(φ))
    Identity: s_φ = sin²(φ) = tanh²(ψ)
    So the spread of latitude IS the squared hyperbolic tangent
    of the Mercator coordinate. Spreads are Mercator-native.

  Orbit partition of loxodromes:
    Cosmos (period 24): full circumnavigation loxodromes
    Satellite (period 8): reduced loxodromes (8 principal winds)
    Singularity (period 1): degenerate (no motion)

SOURCE: Mercator (1569); Wildberger (2005); Wall/Pisano period;
        cert [163] QA Dead Reckoning.

Checks
------
LX_1         schema_version == 'QA_LOXODROME_CERT.v1'
LX_PATH      T-operator path has correct period (Pisano)
LX_BEARING   Bearing spread = e²/G for witness directions
LX_MERCATOR  s_φ = tanh²(ψ) identity verified
LX_ORBIT     Orbit classification matches period
LX_W         at least 3 loxodrome witnesses
LX_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_LOXODROME_CERT.v1"


def t_operator(b, e, m):
    """Single QA T-operator step, A1 compliant."""
    return ((e - 1) % m) + 1, (((b + e) - 1) % m) + 1


def loxodrome_period(b0, e0, m, max_steps=500):
    """Compute period of T-operator path from (b0,e0) mod m."""
    b, e = b0, e0
    for step in range(1, max_steps + 1):
        b, e = t_operator(b, e, m)
        if (b, e) == (b0, e0):
            return step
    return -1  # did not cycle within max_steps


def classify_orbit(period):
    """Classify orbit from period."""
    if period == 1:
        return "singularity"
    elif period <= 8:
        return "satellite"
    else:
        return "cosmos"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # LX_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("LX_1", f"schema_version must be {SCHEMA}")

    m = cert.get("modulus", 24)
    witnesses = cert.get("witnesses", [])

    for i, w in enumerate(witnesses):
        b0 = w.get("b0")
        e0 = w.get("e0")
        if b0 is None or e0 is None:
            err("LX_PATH", f"witness {i}: missing b0 or e0")
            continue

        # LX_PATH — verify period
        declared_period = w.get("period")
        computed_period = loxodrome_period(b0, e0, m)
        if declared_period is not None and declared_period != computed_period:
            err("LX_PATH", f"witness {i}: period={computed_period}, "
                f"declared {declared_period}")

        # LX_BEARING — bearing spread = e²/G
        bearing = w.get("bearing_spread")
        d_val = w.get("d")
        e_val = w.get("e_dir")  # direction e, not state e
        if bearing is not None and d_val is not None and e_val is not None:
            G = d_val * d_val + e_val * e_val
            if G > 0:
                expected = (e_val * e_val) / G
                tol = w.get("tolerance", 1e-8)
                if abs(bearing - expected) > tol:
                    err("LX_BEARING", f"witness {i}: bearing_spread={bearing} "
                        f"!= e²/G={expected}")

        # LX_ORBIT — orbit classification
        declared_orbit = w.get("orbit")
        if declared_orbit is not None:
            computed_orbit = classify_orbit(computed_period)
            if computed_orbit != declared_orbit:
                err("LX_ORBIT", f"witness {i}: orbit={computed_orbit}, "
                    f"declared {declared_orbit}")

    # LX_MERCATOR — s_φ = tanh²(ψ)
    mercator_data = cert.get("mercator_identity")
    if mercator_data is not None:
        for j, entry in enumerate(mercator_data):
            lat = entry.get("lat_deg")
            s_phi = entry.get("s_phi")
            psi = entry.get("psi_isometric")
            tol = entry.get("tolerance", 1e-8)

            if lat is not None and s_phi is not None:
                expected_s = math.sin(math.radians(lat)) * math.sin(math.radians(lat))
                if abs(s_phi - expected_s) > tol:
                    err("LX_MERCATOR", f"mercator {j}: s_phi={s_phi} "
                        f"!= sin²({lat}°)={expected_s:.8f}")

            if psi is not None and s_phi is not None:
                tanh_sq = math.tanh(psi) * math.tanh(psi)
                if abs(s_phi - tanh_sq) > tol:
                    err("LX_MERCATOR", f"mercator {j}: s_phi={s_phi} "
                        f"!= tanh²({psi})={tanh_sq:.8f}")

    # LX_W — at least 3 witnesses
    if len(witnesses) < 3:
        err("LX_W", f"need >= 3 witnesses, got {len(witnesses)}")

    # LX_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("LX_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("LX_F: declared FAIL but no fail_ledger and all checks pass")

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
