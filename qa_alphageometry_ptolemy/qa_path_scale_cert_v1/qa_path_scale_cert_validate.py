#!/usr/bin/env python3
"""QA Path Scale Cert family [146] — certifies G=d^2+e^2 growth profiles
along Pythagorean-tree generator paths.

Scale classes:
  EXPONENTIAL  — G ratio converges to a constant > 1 (UNIFORM_B: 3+2sqrt2)
  POLYNOMIAL   — G ratio -> 1 (polynomial growth: UNIFORM_A, UNIFORM_C)

Key result: UNIFORM_B (M_B-only) paths have G_{n+1}/G_n -> 3+2*sqrt(2)
= 5.82842712... This is because M_B's matrix [[2,1],[1,0]] has dominant
eigenvalue 1+sqrt(2), so d grows by factor (1+sqrt(2)) per step, and
G ~ d^2 * const, giving G ratio -> (1+sqrt(2))^2 = 3+2*sqrt(2).

All forward paths from any primitive direction have G strictly increasing.

Checks: SC_1 (schema), SC_2 (G values correct), SC_GROWTH (G monotone
increasing), SC_RATIO (G ratios match declared values within tolerance),
SC_CONV_B (for EXPONENTIAL paths, final ratio within 0.01 of 3+2sqrt2),
SC_W (>=3 paths), SC_F (root (2,1) present).

Source: Pell equation theory (silver ratio), certs [135] Pythagorean Tree,
[141] Pell Norm, [145] Path Shape.
"""

import json
import math
import os
import sys

SCHEMA = "QA_PATH_SCALE_CERT.v1"
PELL_LIMIT = 3.0 + 2.0 * math.sqrt(2.0)   # 5.82842712474619...
VALID_MOVES = frozenset(["M_A", "M_B", "M_C"])
VALID_SCALE_CLASSES = frozenset(["EXPONENTIAL", "POLYNOMIAL"])


def apply_move(d, e, move):
    if move == "M_A":
        return 2*d - e, d
    elif move == "M_B":
        return 2*d + e, d
    elif move == "M_C":
        return d + 2*e, e
    else:
        raise ValueError(f"unknown move: {move}")


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # SC_1 — schema
    if cert.get("schema_version") != SCHEMA:
        err("SC_1", f"schema_version must be {SCHEMA}")

    paths = cert.get("paths", [])
    if not isinstance(paths, list) or len(paths) == 0:
        err("SC_W", "paths array is empty")
        return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}

    has_fundamental = False

    for i, p in enumerate(paths):
        root = p.get("root", {})
        d0, e0 = root.get("d", 0), root.get("e", 0)
        moves = p.get("moves", [])
        scale_class = p.get("scale_class", "")
        G_seq_decl = p.get("G_sequence", [])
        ratios_decl = p.get("G_ratios", [])

        if d0 == 2 and e0 == 1:
            has_fundamental = True

        if scale_class not in VALID_SCALE_CLASSES:
            err("SC_1", f"path[{i}] unknown scale_class: {scale_class}")

        # Replay path, compute G sequence
        d, e = d0, e0
        G_computed = [d*d + e*e]

        for j, m in enumerate(moves):
            if m not in VALID_MOVES:
                err("SC_2", f"path[{i}] move[{j}] unknown: {m}")
                continue
            d, e = apply_move(d, e, m)
            G_computed.append(d*d + e*e)

        # SC_2 — G values correct
        if len(G_seq_decl) > 0:
            if G_computed != G_seq_decl:
                err("SC_2", f"path[{i}] G_sequence mismatch: computed {G_computed} != declared {G_seq_decl}")

        # SC_GROWTH — G monotone increasing
        for j in range(1, len(G_computed)):
            if G_computed[j] <= G_computed[j-1]:
                err("SC_GROWTH", f"path[{i}] G not increasing at step {j}: {G_computed[j-1]} -> {G_computed[j]}")
                break

        # Compute ratios
        ratios_computed = []
        for j in range(1, len(G_computed)):
            ratios_computed.append(G_computed[j] / G_computed[j-1])

        # SC_RATIO — declared ratios within tolerance
        if len(ratios_decl) > 0:
            if len(ratios_decl) != len(ratios_computed):
                err("SC_RATIO", f"path[{i}] G_ratios length mismatch: {len(ratios_decl)} != {len(ratios_computed)}")
            else:
                for j in range(len(ratios_decl)):
                    if abs(ratios_decl[j] - ratios_computed[j]) > 0.001:
                        err("SC_RATIO", f"path[{i}] ratio[{j}] mismatch: declared {ratios_decl[j]} vs computed {ratios_computed[j]:.6f}")

        # SC_CONV_B — exponential convergence to 3+2sqrt2
        if scale_class == "EXPONENTIAL":
            if len(ratios_computed) >= 3:
                final_ratio = ratios_computed[-1]
                if abs(final_ratio - PELL_LIMIT) > 0.01:
                    err("SC_CONV_B", f"path[{i}] final ratio {final_ratio:.6f} not within 0.01 of {PELL_LIMIT:.6f}")
            else:
                warnings.append(f"SC_CONV_B: path[{i}] too short to test convergence (need >=4 steps)")

    # SC_W — at least 3 paths
    if len(paths) < 3:
        err("SC_W", f"need >=3 paths, got {len(paths)}")

    # SC_F — fundamental
    if not has_fundamental:
        err("SC_F", "no path with root (2,1)")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


# --- Self-test ---

def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "sc_pass_scale_classes.json": True,
        "sc_pass_pell_convergence.json": True,
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
        print("Usage: python qa_path_scale_cert_validate.py [--self-test | <fixture.json>]")
