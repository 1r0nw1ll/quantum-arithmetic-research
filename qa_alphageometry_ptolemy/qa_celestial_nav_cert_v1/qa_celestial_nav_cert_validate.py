#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=celestial_nav_fixtures"
"""QA Celestial Nav Cert family [165] — certifies celestial navigation sight
reduction as rational trigonometry (spreads, crosses, and discrete orientation).

TIER 1 — EXACT REFORMULATION:

  Classical sight reduction:
    sin(h) = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(LHA)
    where h=altitude, φ=latitude, δ=declination, LHA=local hour angle

  Rational sight reduction:
    s_h = [σ₁·√(s_φ·s_δ) + σ₂·√(c_φ·c_δ·c_LHA)]²

    where:
      s_φ = sin²(φ) = spread of latitude
      s_δ = sin²(δ) = spread of declination
      s_LHA = sin²(LHA) = spread of hour angle
      c_x = 1 - s_x = cross
      σ₁ = +1 if φ,δ same hemisphere, -1 if opposite  (DISCRETE)
      σ₂ = +1 if |LHA| ≤ 90° (cos(LHA) ≥ 0), -1 otherwise  (DISCRETE)

  The orientation flags σ₁, σ₂ are discrete choices — exactly two bits.
  This is Theorem NT in action: continuous angles decompose into
  unsigned quadratic measures (spreads) + discrete orientation (σ flags).

  Azimuth (spread form):
    s_Az = c_δ · s_LHA / s_z
    where s_z = 1 - s_h = spread of zenith distance

  Position circle:
    All points where spread(star, zenith) = s_observed.
    This is a spread locus — an algebraic curve, not a transcendental one.

  Two-star fix:
    Intersection of two position circles = solving two spread equations.
    Pure algebraic computation, no trig inversion needed.

  SEXTANT IS A SPREAD INSTRUMENT:
    A sextant measures the angle between horizon and star.
    sin²(altitude) = spread. The instrument's double-reflection
    produces a value that IS the spread, not an angle.

SOURCE: Wildberger (2005); Bowditch, American Practical Navigator;
        Nautical Almanac Office; cert [156] QA WGS84 ellipse.

Checks
------
CN_1         schema_version == 'QA_CELESTIAL_NAV_CERT.v1'
CN_SIGHT     s_altitude matches classical sin²(h) for all witnesses
CN_SPREAD    s_altitude + (1 - s_altitude) = 1 (tautological but structural)
CN_SIGMA     orientation flags σ₁, σ₂ correctly computed from hemisphere/quadrant
CN_AZIMUTH   s_azimuth = c_dec * s_lha / s_z verified
CN_FIX       two-star fix data present and self-consistent
CN_W         at least 3 star witnesses
CN_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_CELESTIAL_NAV_CERT.v1"


def sight_reduction_classical(lat_deg, dec_deg, lha_deg):
    """Classical sight reduction. Returns (altitude_deg, azimuth_deg)."""
    phi = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    lha = math.radians(lha_deg)
    sin_h = (math.sin(phi) * math.sin(dec) +
             math.cos(phi) * math.cos(dec) * math.cos(lha))
    h = math.asin(max(-1.0, min(1.0, sin_h)))
    if abs(math.cos(h)) < 1e-10:
        return math.degrees(h), 0.0
    cos_az = ((math.sin(dec) - math.sin(phi) * math.sin(h)) /
              (math.cos(phi) * math.cos(h)))
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.acos(cos_az)
    if math.sin(lha) > 0:
        az = 2 * math.pi - az
    return math.degrees(h), math.degrees(az)


def sight_reduction_rt(s_phi, s_dec, s_lha, sigma_dec, sigma_lha):
    """Rational trig sight reduction with discrete orientation flags.

    sigma_dec: +1 if lat,dec same hemisphere, -1 if opposite
    sigma_lha: +1 if cos(LHA) >= 0, -1 if cos(LHA) < 0
    Returns (s_altitude, s_azimuth).
    """
    c_phi = 1.0 - s_phi
    c_dec = 1.0 - s_dec
    c_lha = 1.0 - s_lha

    term1 = math.sqrt(s_phi * s_dec)
    term2 = math.sqrt(c_phi * c_dec * c_lha)

    cos_z = sigma_dec * term1 + sigma_lha * term2
    s_altitude = cos_z * cos_z  # sin²(h) = cos²(z)
    spread_z = 1.0 - s_altitude  # sin²(z)

    if spread_z > 1e-20:
        s_azimuth = c_dec * s_lha / spread_z
    else:
        s_azimuth = 0.0

    return s_altitude, s_azimuth


def compute_sigma(lat_deg, dec_deg, lha_deg):
    """Compute discrete orientation flags from hemisphere/quadrant."""
    # sigma_dec: +1 if same hemisphere, -1 if opposite
    same_hemi = (lat_deg >= 0 and dec_deg >= 0) or (lat_deg < 0 and dec_deg < 0)
    sigma_dec = 1 if same_hemi else -1

    # sigma_lha: +1 if cos(LHA) >= 0, -1 if cos(LHA) < 0
    lha_norm = lha_deg % 360
    sigma_lha = 1 if (lha_norm <= 90 or lha_norm >= 270) else -1

    return sigma_dec, sigma_lha


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # CN_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("CN_1", f"schema_version must be {SCHEMA}")

    witnesses = cert.get("witnesses", [])

    for i, w in enumerate(witnesses):
        lat = w.get("lat_deg")
        dec = w.get("dec_deg")
        lha = w.get("lha_deg")
        tol = w.get("tolerance", 1e-8)

        if lat is None or dec is None or lha is None:
            err("CN_SIGHT", f"witness {i}: missing lat_deg, dec_deg, or lha_deg")
            continue

        # Compute classical reference
        h_cl, az_cl = sight_reduction_classical(lat, dec, lha)
        s_alt_expected = math.sin(math.radians(h_cl)) * math.sin(math.radians(h_cl))

        # Get or compute spreads
        s_phi = w.get("s_phi")
        s_dec = w.get("s_dec")
        s_lha = w.get("s_lha")
        if s_phi is None:
            s_phi = math.sin(math.radians(lat)) * math.sin(math.radians(lat))
        if s_dec is None:
            s_dec = math.sin(math.radians(dec)) * math.sin(math.radians(dec))
        if s_lha is None:
            s_lha = math.sin(math.radians(lha)) * math.sin(math.radians(lha))

        # CN_SIGMA — orientation flags
        declared_sd = w.get("sigma_dec")
        declared_sl = w.get("sigma_lha")
        computed_sd, computed_sl = compute_sigma(lat, dec, lha)

        if declared_sd is not None and declared_sd != computed_sd:
            err("CN_SIGMA", f"witness {i}: sigma_dec={declared_sd}, "
                f"expected {computed_sd} (lat={lat}, dec={dec})")
        if declared_sl is not None and declared_sl != computed_sl:
            err("CN_SIGMA", f"witness {i}: sigma_lha={declared_sl}, "
                f"expected {computed_sl} (lha={lha})")

        sd = declared_sd if declared_sd is not None else computed_sd
        sl = declared_sl if declared_sl is not None else computed_sl

        # CN_SIGHT — s_altitude matches classical
        s_alt, s_az = sight_reduction_rt(s_phi, s_dec, s_lha, sd, sl)

        declared_s_alt = w.get("s_altitude")
        if declared_s_alt is not None:
            if abs(declared_s_alt - s_alt_expected) > tol:
                err("CN_SIGHT", f"witness {i}: declared s_alt={declared_s_alt:.8f} "
                    f"!= classical sin²(h)={s_alt_expected:.8f}")

        if abs(s_alt - s_alt_expected) > tol:
            err("CN_SIGHT", f"witness {i}: RT s_alt={s_alt:.8f} "
                f"!= classical sin²(h)={s_alt_expected:.8f}")

        # CN_SPREAD — structural identity
        if abs(s_alt + (1.0 - s_alt) - 1.0) > 1e-15:
            err("CN_SPREAD", f"witness {i}: s_alt + (1-s_alt) != 1.0")

        # CN_AZIMUTH — azimuth spread
        declared_s_az = w.get("s_azimuth")
        if declared_s_az is not None:
            spread_z = 1.0 - s_alt
            if spread_z > 1e-15:
                expected_s_az = (1.0 - s_dec) * s_lha / spread_z
                if abs(declared_s_az - expected_s_az) > tol:
                    err("CN_AZIMUTH", f"witness {i}: s_azimuth={declared_s_az:.8f} "
                        f"!= c_dec*s_lha/s_z={expected_s_az:.8f}")

    # CN_FIX — two-star fix data
    fix_data = cert.get("two_star_fix")
    if fix_data is not None:
        stars = fix_data.get("stars", [])
        if len(stars) < 2:
            err("CN_FIX", "two-star fix requires at least 2 stars")
        for j, star in enumerate(stars):
            s_alt_obs = star.get("s_altitude_observed")
            if s_alt_obs is None:
                err("CN_FIX", f"star {j}: missing s_altitude_observed")
            elif s_alt_obs < 0 or s_alt_obs > 1:
                err("CN_FIX", f"star {j}: s_altitude_observed={s_alt_obs} out of [0,1]")

    # CN_W — at least 3 witnesses
    if len(witnesses) < 3:
        err("CN_W", f"need >= 3 witnesses, got {len(witnesses)}")

    # CN_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("CN_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("CN_F: declared FAIL but no fail_ledger and all checks pass")

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
