#!/usr/bin/env python3
"""QA Sixteen Identities Cert family [148] — certifies the 16 named quantities
(A through L, plus X,W,Y,Z) of a prime Pythagorean direction and their
algebraic relationships.

Given a primitive direction (d,e) with d>e>0, gcd(d,e)=1, opposite parity:

  b = d-e,  a = d+e  (so b+e=d, b+2e=a=d+e — standard QN)
  F = d*d - e*e = b*a,  C = 2*d*e,  G = d*d + e*e

The 16 identities (Iverson, Pyth-1):
  A = a*a              B = b*b
  C = 2*d*e            D = d*d
  E = e*e              F = b*a = d*d - e*e
  G = d*d + e*e        H = C + F
  I = C - F            J = b*d
  K = a*d              L = b*a*d*e / 6 = F*C / 12
  X = d*e = C/2        W = d*(e+a) = X+K
  Y = a*a - d*d        Z = E + K

Key algebraic relations certified:
  ID_GCA: G + C = A (= a^2)
  ID_GCB: G - C = B (= b^2)
  ID_GAB: G = (A+B)/2
  ID_FCG: F^2 + C^2 = G^2
  ID_HI:  H^2 + I^2 = 2*G^2
  ID_L12: L = C*F/12 is a positive integer
  ID_PAR: C is 4-par (C%4==0); G is 5-par (G%4==1)

Checks: SI_1 (schema), SI_2 (direction valid), SI_IDEN (all 16 quantities
recomputed), SI_REL (algebraic relations verified), SI_PAR (C=4-par, G=5-par),
SI_L (L integer), SI_W (>=3 direction witnesses), SI_F (fundamental (2,1)).
"""

import json
import os
import sys
from math import gcd


SCHEMA = "QA_SIXTEEN_IDENTITIES_CERT.v1"


def compute_identities(d, e):
    """Compute all 16 named quantities for direction (d,e)."""
    b = d - e
    a = d + e
    A = a * a
    B = b * b
    C = 2 * d * e
    D = d * d
    E = e * e
    F = b * a          # = d*d - e*e
    G = d * d + e * e
    H = C + F
    I = C - F
    J = b * d
    K = a * d
    L_num = b * a * d * e
    L = L_num // 6     # must be exact
    X = d * e          # = C/2
    W = d * (e + a)    # = X + K
    Y = a * a - d * d  # = A - D
    Z = E + K

    return {
        "b": b, "a": a,
        "A": A, "B": B, "C": C, "D": D, "E": E, "F": F,
        "G": G, "H": H, "I": I, "J": J, "K": K, "L": L,
        "X": X, "W": W, "Y": Y, "Z": Z,
        "_L_exact": (L_num % 6 == 0),
    }


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("SI_1", f"schema_version must be {SCHEMA}")

    directions = cert.get("directions", [])
    if not isinstance(directions, list) or len(directions) == 0:
        err("SI_W", "directions array is empty")
        return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

    has_fundamental = False

    for i, dr in enumerate(directions):
        d_val = dr.get("d", 0)
        e_val = dr.get("e", 0)

        if d_val == 2 and e_val == 1:
            has_fundamental = True

        # SI_2 — direction valid
        if d_val <= e_val or e_val <= 0:
            err("SI_2", f"direction[{i}] ({d_val},{e_val}): need d>e>0")
            continue
        if gcd(d_val, e_val) != 1:
            err("SI_2", f"direction[{i}] ({d_val},{e_val}): gcd != 1")
        if (d_val + e_val) % 2 == 0:
            err("SI_2", f"direction[{i}] ({d_val},{e_val}): same parity (need opposite)")

        computed = compute_identities(d_val, e_val)
        declared = dr.get("identities", {})

        # SI_IDEN — all 16 quantities match
        for key in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "X", "W", "Y", "Z"]:
            if key in declared and declared[key] != computed[key]:
                err("SI_IDEN", f"direction[{i}] ({d_val},{e_val}): {key} declared={declared[key]} computed={computed[key]}")

        # SI_REL — algebraic relations
        A, B, C, D, E, F = computed["A"], computed["B"], computed["C"], computed["D"], computed["E"], computed["F"]
        G, H, I, J, K, L = computed["G"], computed["H"], computed["I"], computed["J"], computed["K"], computed["L"]
        X, W, Y, Z = computed["X"], computed["W"], computed["Y"], computed["Z"]

        # G + C = A
        if G + C != A:
            err("SI_REL", f"direction[{i}]: G+C={G+C} != A={A}")
        # G - C = B
        if G - C != B:
            err("SI_REL", f"direction[{i}]: G-C={G-C} != B={B}")
        # G = (A+B)/2
        if 2 * G != A + B:
            err("SI_REL", f"direction[{i}]: 2G={2*G} != A+B={A+B}")
        # F^2 + C^2 = G^2
        if F * F + C * C != G * G:
            err("SI_REL", f"direction[{i}]: F²+C²={F*F+C*C} != G²={G*G}")
        # H^2 + I^2 = 2*G^2
        if H * H + I * I != 2 * G * G:
            err("SI_REL", f"direction[{i}]: H²+I²={H*H+I*I} != 2G²={2*G*G}")
        # F = b*a
        b_val, a_val = computed["b"], computed["a"]
        if F != b_val * a_val:
            err("SI_REL", f"direction[{i}]: F={F} != b*a={b_val*a_val}")
        # W = X + K
        if W != X + K:
            err("SI_REL", f"direction[{i}]: W={W} != X+K={X+K}")
        # Z = E + K
        if Z != E + K:
            err("SI_REL", f"direction[{i}]: Z={Z} != E+K={E+K}")
        # Y = A - D
        if Y != A - D:
            err("SI_REL", f"direction[{i}]: Y={Y} != A-D={A-D}")

        # SI_PAR — C is 4-par (C%4==0), G is 5-par (G%4==1)
        if C % 4 != 0:
            err("SI_PAR", f"direction[{i}]: C={C} is not 4-par (C%4={C%4})")
        if G % 4 != 1:
            err("SI_PAR", f"direction[{i}]: G={G} is not 5-par (G%4={G%4})")

        # SI_L — L is a positive integer
        if not computed["_L_exact"]:
            err("SI_L", f"direction[{i}]: L=bade/6 is not exact integer")
        if L <= 0:
            err("SI_L", f"direction[{i}]: L={L} not positive")

    # SI_W — at least 3 direction witnesses
    if len(directions) < 3:
        err("SI_W", f"need >=3 directions, got {len(directions)}")

    # SI_F — fundamental
    if not has_fundamental:
        err("SI_F", "no direction (2,1)")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "si_pass_fundamental.json": True,
        "si_pass_witnesses.json": True,
    }
    results = []
    for fname, should_pass in expected.items():
        path = os.path.join(fix_dir, fname)
        with open(path) as f:
            cert = json.load(f)
        res = validate(cert)
        ok = res["ok"] == should_pass
        results.append({
            "fixture": fname,
            "expected_pass": should_pass,
            "actual_pass": res["ok"],
            "ok": ok,
            "errors": res["errors"],
            "warnings": res["warnings"],
        })
    return results


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        results = self_test()
        all_ok = all(r["ok"] for r in results)
        print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cert = json.load(f)
        print(json.dumps(validate(cert), indent=2))
    else:
        print("Usage: python qa_sixteen_identities_cert_validate.py [--self-test | <fixture.json>]")
