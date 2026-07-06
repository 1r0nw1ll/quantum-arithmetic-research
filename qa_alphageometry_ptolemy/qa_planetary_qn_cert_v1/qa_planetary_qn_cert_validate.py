#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=planetary_qn_fixtures"
"""QA Planetary QN Cert family [177] — catalogs quantum numbers for solar
system bodies and verifies harmonic connections.

TIER 2 — STRUCTURAL CATALOG:

  For each body with eccentricity ε:
    Best-fit QN: (b, e, d, a) where e/d ≈ ε, d=b+e, a=b+2e
    Triple: C=2de, F=d²-e², G=d²+e², C²+F²=G²
    Characteristic latitude: arcsin(√(2ε/(1+ε²)))

  HARMONIC CONNECTIONS (Law of Harmonics [149]):
    Bodies sharing prime factors in their QN products are harmonically linked.

SOURCE: NASA Planetary Fact Sheets (nssdc.gsfc.nasa.gov); Archinal et al.
        (2018), Celestial Mechanics and Dynamical Astronomy 130:22, DOI:
        10.1007/s10569-017-9805-5. Iverson QA; cert [149] Law of
        Harmonics; cert [156] WGS84 (Earth shape QN).

Checks
------
PQ_1         schema_version == 'QA_PLANETARY_QN_CERT.v1'
PQ_TUPLE     d=b+e, a=b+2e for each body QN
PQ_TRIPLE    C²+F²=G² for each body
PQ_ECC       e/d approximates declared eccentricity within tolerance
PQ_REAL      declared eccentricity matches independently hardcoded real
             astronomical reference values (catches a declared value
             that is self-consistent with its own QN but doesn't match
             reality -- found 2026-07-06: Mars_shape's declared 0.06489
             was just its own QN's e/d, not real Mars data, off by ~40x
             its own tolerance from the true 0.10837)
PQ_HARMONIC  declared harmonic connections verified (shared prime factors)
PQ_W         at least 5 body witnesses
PQ_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_PLANETARY_QN_CERT.v1"

# Independently-sourced real astronomical eccentricities, keyed by the
# fixture's "name" field. Shape eccentricities computed from equatorial/
# polar radii (NASA Planetary Fact Sheets); orbit eccentricities are the
# standard mean values (NASA Planetary Fact Sheets / JPL). This exists so
# PQ_REAL can catch a declared "eccentricity" that is merely self-
# consistent with its own QN (e/d) rather than actually matching reality.
REAL_ECCENTRICITY = {
    "Earth_shape": 0.08181919084262149,
    "Earth_orbit": 0.0167086,
    "Mars_shape": 0.10836599942613576,
    "Mars_orbit": 0.0934,
    "Jupiter_shape": 0.35430478842375246,
    "Jupiter_orbit": 0.04839266,
    "Saturn_shape": 0.43165753827961145,
    "Saturn_orbit": 0.0541506,
    "Uranus_orbit": 0.04716771,
    "Moon_orbit": 0.0549,
}


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("PQ_1", f"schema_version must be {SCHEMA}")

    bodies = cert.get("bodies", [])

    for i, body in enumerate(bodies):
        name = body.get("name", f"body_{i}")
        qn = body.get("qn")
        if qn is None:
            continue

        b = qn.get("b", 0)
        e = qn.get("e", 0)
        d = qn.get("d", 0)
        a = qn.get("a", 0)

        # PQ_TUPLE
        if d != b + e:
            err("PQ_TUPLE", f"{name}: d={d} != b+e={b+e}")
        if a != b + 2 * e:
            err("PQ_TUPLE", f"{name}: a={a} != b+2e={b+2*e}")

        # PQ_TRIPLE
        C = 2 * d * e
        F = d * d - e * e
        G = d * d + e * e
        if C * C + F * F != G * G:
            err("PQ_TRIPLE", f"{name}: C²+F²={C*C+F*F} != G²={G*G}")

        # PQ_ECC
        declared_ecc = body.get("eccentricity")
        ecc_tol = body.get("ecc_tolerance", 0.001)
        if declared_ecc is not None and d > 0:
            computed_ecc = e / d
            if abs(computed_ecc - declared_ecc) > ecc_tol:
                err("PQ_ECC", f"{name}: e/d={computed_ecc:.6f} != declared ε={declared_ecc:.6f}")

        # PQ_REAL — declared eccentricity must match real astronomical data,
        # not just be self-consistent with its own QN's e/d (see module
        # docstring: this is what let the Mars_shape bug through before).
        real_ecc = REAL_ECCENTRICITY.get(name)
        if real_ecc is not None and declared_ecc is not None:
            if abs(declared_ecc - real_ecc) > ecc_tol:
                err("PQ_REAL", f"{name}: declared ε={declared_ecc:.6f} != real ε={real_ecc:.6f} "
                    f"(tolerance {ecc_tol})")

    # PQ_HARMONIC — check declared harmonic connections
    harmonics = cert.get("harmonic_connections", [])
    body_map = {}
    for body in bodies:
        name = body.get("name")
        qn = body.get("qn")
        if name and qn:
            b, e, d, a = qn["b"], qn["e"], qn["d"], qn["a"]
            body_map[name] = b * e * d * a

    for h in harmonics:
        body1 = h.get("body1")
        body2 = h.get("body2")
        shared = h.get("shared_factor")
        if body1 in body_map and body2 in body_map:
            g = gcd(body_map[body1], body_map[body2])
            if shared is not None and g % shared != 0:
                err("PQ_HARMONIC", f"{body1}↔{body2}: declared shared factor {shared} "
                    f"does not divide GCD={g}")

    # PQ_W
    if len(bodies) < 5:
        err("PQ_W", f"need >= 5 bodies, got {len(bodies)}")

    # PQ_F
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("PQ_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("PQ_F: declared FAIL but no fail_ledger and all checks pass")

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
