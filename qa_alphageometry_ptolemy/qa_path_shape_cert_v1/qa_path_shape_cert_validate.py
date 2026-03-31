#!/usr/bin/env python3
"""QA Path Shape Cert family [145] — classifies generator-sequence shapes
in the Pythagorean tree.

A **path shape** is the generator sequence (g_1, ..., g_n) where each g_i
is one of M_A, M_B, M_C (the three Pythagorean-tree generators).

Four shape classes:
  UNIFORM_A  — only M_A moves; (d,e)->(d+1,d) each step
  UNIFORM_B  — only M_B moves; Pell chain, P(x,y) alternates sign
  UNIFORM_C  — only M_C moves; e constant, d grows by +2e
  MIXED      — two or more distinct generators used

Invariants certified:
  - Primitivity: gcd(d,e)=1 preserved at every step
  - F^2+C^2=G^2 at every step (identity, not checked, structural)
  - UNIFORM_B: Pell norm P(d-e, e) = (d-e)^2 - 2*e^2 alternates sign
  - UNIFORM_C: e is constant along the path

Checks: PS_1 (schema), PS_2 (direction validity), PS_CLASS (shape class
matches moves), PS_INV_B (Pell alternation for UNIFORM_B), PS_INV_C
(e constant for UNIFORM_C), PS_W (>=4 paths, one per class), PS_F
(fundamental root (2,1)).

Source: Ben Iverson Koenig tree, Barning 1963/Hall 1970/Price 2008,
certs [134] Egyptian Fraction, [135] Pythagorean Tree, [141] Pell Norm.
"""

import json
import os
import sys
from math import gcd

SCHEMA = "QA_PATH_SHAPE_CERT.v1"
VALID_MOVES = frozenset(["M_A", "M_B", "M_C"])
VALID_CLASSES = frozenset(["UNIFORM_A", "UNIFORM_B", "UNIFORM_C", "MIXED"])


# --- QA primitives (S1-safe: no **2) ---

def qa_triple(d, e):
    return d*d - e*e, 2*d*e, d*d + e*e   # F, C, G


def apply_move(d, e, move):
    if move == "M_A":
        return 2*d - e, d
    elif move == "M_B":
        return 2*d + e, d
    elif move == "M_C":
        return d + 2*e, e
    else:
        raise ValueError(f"unknown move: {move}")


def pell_norm(d, e):
    """P(x,y) = x^2 - 2*y^2 where x=d-e, y=e."""
    x = d - e
    return x*x - 2*e*e


def classify_moves(moves):
    """Determine shape class from the generator sequence."""
    s = set(moves)
    if len(s) == 0:
        return None
    if len(s) > 1:
        return "MIXED"
    only = s.pop()
    return {"M_A": "UNIFORM_A", "M_B": "UNIFORM_B", "M_C": "UNIFORM_C"}[only]


# --- Validator ---

def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # PS_1 — schema
    if cert.get("schema_version") != SCHEMA:
        err("PS_1", f"schema_version must be {SCHEMA}")

    paths = cert.get("paths", [])
    if not isinstance(paths, list) or len(paths) == 0:
        err("PS_W", "paths array is empty")
        return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

    classes_seen = set()
    has_fundamental = False

    for i, p in enumerate(paths):
        root = p.get("root", {})
        d0, e0 = root.get("d", 0), root.get("e", 0)
        moves = p.get("moves", [])
        shape_class = p.get("shape_class", "")
        steps = p.get("steps", [])

        if d0 == 2 and e0 == 1:
            has_fundamental = True

        # PS_2 — root direction valid
        if d0 <= e0 or e0 <= 0:
            err("PS_2", f"path[{i}] root ({d0},{e0}): need d>e>0")
            continue
        if gcd(d0, e0) != 1:
            err("PS_2", f"path[{i}] root ({d0},{e0}): gcd != 1")

        # PS_CLASS — declared shape matches actual moves
        expected_class = classify_moves(moves)
        if shape_class != expected_class:
            err("PS_CLASS", f"path[{i}] declared {shape_class} but moves give {expected_class}")
        classes_seen.add(shape_class)

        # Verify every move is valid
        for j, m in enumerate(moves):
            if m not in VALID_MOVES:
                err("PS_CLASS", f"path[{i}] move[{j}] unknown: {m}")

        # Replay the path and verify steps
        d, e = d0, e0
        pell_signs = [1 if pell_norm(d, e) > 0 else -1 if pell_norm(d, e) < 0 else 0]

        for j, m in enumerate(moves):
            d_new, e_new = apply_move(d, e, m)
            # PS_2 — direction valid at every step
            if d_new <= e_new or e_new <= 0:
                err("PS_2", f"path[{i}] step {j}: ({d_new},{e_new}) invalid direction")
            if gcd(d_new, e_new) != 1:
                err("PS_2", f"path[{i}] step {j}: gcd({d_new},{e_new}) != 1")

            # Check declared step matches computed
            if j < len(steps):
                s = steps[j]
                sd, se = s.get("d", -1), s.get("e", -1)
                if (sd, se) != (d_new, e_new):
                    err("PS_2", f"path[{i}] step[{j}] declared ({sd},{se}) != computed ({d_new},{e_new})")
                # Check F, C, G
                F_exp, C_exp, G_exp = qa_triple(d_new, e_new)
                if s.get("F") != F_exp or s.get("C") != C_exp or s.get("G") != G_exp:
                    err("PS_2", f"path[{i}] step[{j}] triple mismatch: expected F={F_exp} C={C_exp} G={G_exp}")

            pn = pell_norm(d_new, e_new)
            pell_signs.append(1 if pn > 0 else -1 if pn < 0 else 0)

            d, e = d_new, e_new

        # PS_INV_B — Pell alternation for UNIFORM_B
        if shape_class == "UNIFORM_B":
            for j in range(1, len(pell_signs)):
                if pell_signs[j] == pell_signs[j-1]:
                    err("PS_INV_B", f"path[{i}] Pell norm does not alternate at step {j}")
                    break

        # PS_INV_C — e constant for UNIFORM_C
        if shape_class == "UNIFORM_C":
            e_val = e0
            d_cur, e_cur = d0, e0
            for j, m in enumerate(moves):
                d_cur, e_cur = apply_move(d_cur, e_cur, m)
                if e_cur != e_val:
                    err("PS_INV_C", f"path[{i}] step {j}: e changed from {e_val} to {e_cur}")
                    break

    # PS_W — at least 4 paths, one per shape class
    if len(paths) < 4:
        err("PS_W", f"need >=4 paths, got {len(paths)}")
    for c in VALID_CLASSES:
        if c not in classes_seen:
            warnings.append(f"PS_W: shape class {c} not witnessed")

    # PS_F — fundamental root present
    if not has_fundamental:
        err("PS_F", "no path with root (2,1)")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


# --- Self-test ---

def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "ps_pass_four_shapes.json": True,
        "ps_pass_invariants.json": True,
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
        print("Usage: python qa_path_shape_cert_validate.py [--self-test | <fixture.json>]")
