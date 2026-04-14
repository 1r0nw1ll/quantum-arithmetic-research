#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=integer_directions"
"""QA Equilateral Triangle Cert family [152] — certifies the equilateral
triangle identities W, Y, Z and their Eisenstein norm property.

DEFINITIONS (Iverson, QA-2 Ch 7, elements.txt):
  W = d(e+a) = X+K = de + da
  Y = A-D = a²-d² = C+E = 2de+e² (dual definition)
  Z = E+K = e²+ad

EISENSTEIN NORM (certified also in [133]):
  F²-FW+W² = Z²   (Eisenstein triple (F,W,Z))
  Y²-YW+W² = Z²   (Eisenstein triple (Y,W,Z))

KEY IDENTITY: Y = C+E = A-D (the equilateral link — connects the
Pythagorean elements C,E to the square elements A,D via equilateral
triangle geometry).

Additional: F+Y = W (the three equilateral measurements sum correctly).

Checks: ET_1 (schema), ET_DEF (W,Y,Z computed correctly), ET_DUAL
(Y=A-D=C+E both hold), ET_EIS (both Eisenstein norms = Z²),
ET_SUM (F+Y=W), ET_W (>=3 directions), ET_F (fundamental (2,1)).
"""

import json
import os
import sys
from math import gcd
from pathlib import Path

# Canonical element computation lives in qa_alphageometry_ptolemy/qa_elements.py.
# Cert-local reimplementation is forbidden by ELEM-2; use qa_elements.qa_elements()
# exclusively. This cert addresses equilateral-triangle impossibility in QA;
# the underlying element field set is canonical.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qa_elements import qa_elements


SCHEMA = "QA_EQUILATERAL_TRIANGLE_CERT.v1"


def compute_equilateral_elements(d, e):
    """Cert wrapper: parameterized by (d, e) per cert convention.

    Recovers b = d - e (forced by A2: d = b+e), then delegates all
    element computation to canonical qa_elements(). The Y_CE alias
    (C + E) and the dict shape are kept because the cert's identity
    checks (ET_DUAL: Y = A-D = C+E) reference both forms.
    """
    b = d - e
    elem = qa_elements(b, e)
    return {
        "b": elem.b, "a": elem.a,
        "A": elem.A, "B": elem.B, "C": elem.C, "D": elem.D, "E": elem.E,
        "F": elem.F, "G": elem.G, "X": elem.X, "K": elem.K, "W": elem.W,
        "Y_AD": elem.Y,            # A - D
        "Y_CE": elem.C + elem.E,   # 2de + e² — same identity, distinct algebraic form
        "Z": elem.Z,
    }


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("ET_1", f"schema_version must be {SCHEMA}")

    directions = cert.get("directions", [])
    has_fundamental = False

    for i, dr in enumerate(directions):
        d_val = dr.get("d", 0)
        e_val = dr.get("e", 0)
        if d_val == 2 and e_val == 1:
            has_fundamental = True
        if d_val <= e_val or e_val <= 0:
            err("ET_DEF", f"direction[{i}] ({d_val},{e_val}): need d>e>0")
            continue

        comp = compute_equilateral_elements(d_val, e_val)
        decl = dr.get("identities", {})

        # ET_DEF — W, Y, Z computed correctly
        for key in ["W", "Z"]:
            if key in decl and decl[key] != comp[key]:
                err("ET_DEF", f"direction[{i}] ({d_val},{e_val}): {key} declared={decl[key]} computed={comp[key]}")
        if "Y" in decl:
            if decl["Y"] != comp["Y_AD"]:
                err("ET_DEF", f"direction[{i}] ({d_val},{e_val}): Y declared={decl['Y']} computed={comp['Y_AD']}")

        # ET_DUAL — Y = A-D = C+E
        if comp["Y_AD"] != comp["Y_CE"]:
            err("ET_DUAL", f"direction[{i}] ({d_val},{e_val}): A-D={comp['Y_AD']} != C+E={comp['Y_CE']}")

        # ET_EIS — Eisenstein norms
        F, W, Z = comp["F"], comp["W"], comp["Z"]
        Y = comp["Y_AD"]
        eis1 = F * F - F * W + W * W
        eis2 = Y * Y - Y * W + W * W
        z_sq = Z * Z
        if eis1 != z_sq:
            err("ET_EIS", f"direction[{i}] ({d_val},{e_val}): F²-FW+W²={eis1} != Z²={z_sq}")
        if eis2 != z_sq:
            err("ET_EIS", f"direction[{i}] ({d_val},{e_val}): Y²-YW+W²={eis2} != Z²={z_sq}")

        # ET_SUM — F+Y=W
        if F + Y != W:
            err("ET_SUM", f"direction[{i}] ({d_val},{e_val}): F+Y={F+Y} != W={W}")

    if len(directions) < 3:
        err("ET_W", f"need >=3 directions, got {len(directions)}")
    if not has_fundamental:
        err("ET_F", "no direction (2,1)")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "et_pass_eisenstein.json": True,
        "et_pass_dual_identity.json": True,
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
        print("Usage: python qa_equilateral_triangle_cert_validate.py [--self-test | <fixture.json>]")
