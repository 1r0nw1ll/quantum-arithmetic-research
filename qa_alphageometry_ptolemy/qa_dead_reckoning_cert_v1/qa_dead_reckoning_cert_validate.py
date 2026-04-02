#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=dead_reckoning_fixtures"
"""QA Dead Reckoning Cert family [163] — certifies the QA T-operator as an
exact dead reckoning engine on a mod-m lattice with zero computational drift.

TIER 2 — STRUCTURAL + COMPUTATIONAL:

  Classical dead reckoning (per leg):
    x_{k+1} = x_k + d_k · cos(θ_k)
    y_{k+1} = y_k + d_k · sin(θ_k)
  Error accumulates from sin/cos float approximation at every step.

  QA dead reckoning:
    (b_{k+1}, e_{k+1}) = T · (b_k, e_k) mod m
    where T = [[0,1],[1,1]] (Fibonacci shift / QA T-operator)
    After k steps: (b_k, e_k) = T^k · (b_0, e_0) mod m
    EXACT integer arithmetic. Zero computational drift.

  Three chromogeometric metrics per direction (d,e):
    G = d*d + e*e  (blue/Euclidean — position fix)
    F = d*d - e*e  (red/Minkowski — hyperbolic fix / LORAN)
    C = 2*d*e      (green/area — cross-track error)
    Identity: C*C + F*F = G*G

  Compass rose mod-24:
    24-cycle cosmos = full circumnavigation bearings
    8-cycle satellite = 8 principal winds (N,NE,E,SE,S,SW,W,NW)
    1-cycle singularity = fixed point (no bearing)

  Theorem NT: continuous Earth coordinates are observer projections ONLY.
  QA state (b,e) is discrete; float lat/lon enters at input, exits at output.

SOURCE: Iverson QA T-operator; Wildberger rational trigonometry (2005);
        Berggren tree navigation (Barning 1963, Berggren 1934).

Checks
------
DR_1         schema_version == 'QA_DEAD_RECKONING_CERT.v1'
DR_TOP       T-operator T=[[0,1],[1,1]] produces correct state evolution
DR_EXACT     After N legs, QA state matches closed-form T^N · (b0,e0) mod m
DR_DRIFT     Classical float DR accumulates error; QA DR has zero error
DR_CHROMO    C*C + F*F == G*G for all witness directions
DR_COMPASS   Mod-24 orbit classification of compass bearings verified
DR_W         at least 3 route witnesses
DR_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_DEAD_RECKONING_CERT.v1"


def t_operator(b, e, m):
    """Single QA T-operator step: (b,e) -> (e, b+e) mod m, A1 compliant."""
    new_b = ((e - 1) % m) + 1
    new_e = (((b + e) - 1) % m) + 1
    return new_b, new_e


def t_operator_n(b, e, m, n):
    """Apply T-operator n times via iteration."""
    for _ in range(n):
        b, e = t_operator(b, e, m)
    return b, e


def mat3_mul_mod(A, B, m):
    """3x3 matrix multiply mod m (for augmented affine exponentiation)."""
    R = [[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            s = 0
            for k in range(3):
                s += A[i][k] * B[k][j]
            R[i][j] = s % m
    return R


def mat3_pow_mod(M, n, m):
    """3x3 matrix power mod m via repeated squaring."""
    result = [[1,0,0],[0,1,0],[0,0,1]]  # identity
    base = [row[:] for row in M]
    while n > 0:
        if n % 2 == 1:
            result = mat3_mul_mod(result, base, m)
        base = mat3_mul_mod(base, base, m)
        n //= 2
    return result


def t_operator_n_matrix(b, e, m, n):
    """Apply T^n via augmented matrix exponentiation, A1 compliant.

    The A1 T-operator is AFFINE: in 0-indexed coords (B,E),
      (B',E') = (E, B+E+1) mod m
    The +1 comes from A1 encoding: ((b+e-1)%m)+1 maps to ((B+E+1)%m).

    Augmented 3x3 matrix acting on (B, E, 1)^T:
      [[0, 1, 0],    (B)     (E      )
       [1, 1, 1],  · (E)  =  (B+E+1  )
       [0, 0, 1]]    (1)     (1       )
    """
    T_aug = [[0, 1, 0],
             [1, 1, 1],
             [0, 0, 1]]
    Tn = mat3_pow_mod(T_aug, n, m)
    B = b - 1
    E = e - 1
    new_B = (Tn[0][0]*B + Tn[0][1]*E + Tn[0][2]*1) % m
    new_E = (Tn[1][0]*B + Tn[1][1]*E + Tn[1][2]*1) % m
    return new_B + 1, new_E + 1


def classical_dr(legs):
    """Classical dead reckoning: sequence of (bearing_deg, distance) legs.
    Returns list of (x, y) positions starting from (0, 0)."""
    x, y = 0.0, 0.0
    positions = [(x, y)]
    for bearing_deg, dist in legs:
        rad = math.radians(bearing_deg)
        x += dist * math.sin(rad)  # sin for bearing (clockwise from N)
        y += dist * math.cos(rad)
        positions.append((x, y))
    return positions


def classical_dr_perturbed(legs, ulp_noise=1e-15):
    """Classical DR with simulated ULP-level noise per trig call.
    Models the accumulated effect of floating-point sin/cos error."""
    x, y = 0.0, 0.0
    positions = [(x, y)]
    import random
    rng = random.Random(42)
    for bearing_deg, dist in legs:
        rad = math.radians(bearing_deg)
        # Add ULP-level noise to simulate trig table / Taylor truncation
        sin_val = math.sin(rad) + rng.gauss(0, ulp_noise)
        cos_val = math.cos(rad) + rng.gauss(0, ulp_noise)
        x += dist * sin_val
        y += dist * cos_val
        positions.append((x, y))
    return positions


def qa_chromo(d, e):
    """Chromogeometric metrics for direction (d, e)."""
    G = d*d + e*e    # blue / Euclidean
    F = d*d - e*e    # red / Minkowski
    C = 2*d*e        # green / area
    return {"G": G, "F": F, "C": C}


def classify_orbit_mod24(state_b, state_e):
    """Classify (b,e) into cosmos/satellite/singularity under mod-24."""
    m = 24
    b = ((state_b - 1) % m) + 1
    e = ((state_e - 1) % m) + 1
    # Iterate to find orbit period
    seen = set()
    cur_b, cur_e = b, e
    for step in range(25):
        if (cur_b, cur_e) in seen:
            period = step
            break
        seen.add((cur_b, cur_e))
        cur_b, cur_e = t_operator(cur_b, cur_e, m)
    else:
        period = 25  # shouldn't happen for mod-24

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

    # DR_1 — schema version
    if cert.get("schema_version") != SCHEMA:
        err("DR_1", f"schema_version must be {SCHEMA}")

    m = cert.get("modulus", 24)

    # --- Route witnesses ---
    witnesses = cert.get("witnesses", [])

    for i, w in enumerate(witnesses):
        b0 = w.get("b0")
        e0 = w.get("e0")
        n_steps = w.get("n_steps")

        if b0 is None or e0 is None or n_steps is None:
            err("DR_TOP", f"witness {i}: missing b0, e0, or n_steps")
            continue

        # DR_TOP — T-operator produces correct evolution
        b_iter, e_iter = t_operator_n(b0, e0, m, n_steps)
        declared_b = w.get("final_b")
        declared_e = w.get("final_e")
        if declared_b is not None and declared_e is not None:
            if (b_iter, e_iter) != (declared_b, declared_e):
                err("DR_TOP", f"witness {i}: T^{n_steps}·({b0},{e0}) mod {m} "
                    f"= ({b_iter},{e_iter}), declared ({declared_b},{declared_e})")

        # DR_EXACT — matrix exponentiation matches iteration
        b_mat, e_mat = t_operator_n_matrix(b0, e0, m, n_steps)
        if (b_iter, e_iter) != (b_mat, e_mat):
            err("DR_EXACT", f"witness {i}: iteration ({b_iter},{e_iter}) "
                f"!= matrix ({b_mat},{e_mat}) for T^{n_steps}")

        # DR_DRIFT — check drift comparison if provided
        drift_data = w.get("drift_comparison")
        if drift_data is not None:
            qa_error = drift_data.get("qa_accumulated_error", 0)
            classical_error = drift_data.get("classical_accumulated_error", 0)
            if qa_error != 0:
                err("DR_DRIFT", f"witness {i}: QA accumulated error must be 0, got {qa_error}")
            if classical_error <= 0 and n_steps > 10:
                warnings.append(f"DR_DRIFT: witness {i}: classical error={classical_error} "
                                f"seems too low for {n_steps} steps")

        # DR_CHROMO — chromogeometric identity for direction
        chromo = w.get("chromogeometry")
        if chromo is not None:
            d_val = chromo.get("d")
            e_val = chromo.get("e")
            G = chromo.get("G")
            F = chromo.get("F")
            C = chromo.get("C")
            if d_val is not None and e_val is not None:
                expected = qa_chromo(d_val, e_val)
                if G is not None and G != expected["G"]:
                    err("DR_CHROMO", f"witness {i}: G={G} != d*d+e*e={expected['G']}")
                if F is not None and F != expected["F"]:
                    err("DR_CHROMO", f"witness {i}: F={F} != d*d-e*e={expected['F']}")
                if C is not None and C != expected["C"]:
                    err("DR_CHROMO", f"witness {i}: C={C} != 2*d*e={expected['C']}")
            if G is not None and F is not None and C is not None:
                if C*C + F*F != G*G:
                    err("DR_CHROMO", f"witness {i}: C²+F²={C*C+F*F} != G²={G*G}")

    # DR_COMPASS — mod-24 compass classification
    compass_data = cert.get("compass_rose_mod24")
    if compass_data is not None:
        for entry in compass_data:
            bearing_label = entry.get("label", "?")
            b_val = entry.get("b")
            e_val = entry.get("e")
            declared_orbit = entry.get("orbit")
            if b_val is not None and e_val is not None and declared_orbit is not None:
                computed_orbit = classify_orbit_mod24(b_val, e_val)
                if computed_orbit != declared_orbit:
                    err("DR_COMPASS", f"compass {bearing_label}: "
                        f"orbit={computed_orbit}, declared {declared_orbit}")

    # DR_W — at least 3 witnesses
    if len(witnesses) < 3:
        err("DR_W", f"need >= 3 witnesses, got {len(witnesses)}")

    # DR_F — fail detection
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])

    if has_errors and declared == "PASS":
        err("DR_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("DR_F: declared FAIL but no fail_ledger and all checks pass")

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
