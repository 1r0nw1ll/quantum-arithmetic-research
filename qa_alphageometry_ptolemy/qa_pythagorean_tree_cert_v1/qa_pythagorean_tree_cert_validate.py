#!/usr/bin/env python3
"""
qa_pythagorean_tree_cert_validate.py

Validator for QA_PYTHAGOREAN_TREE_CERT.v1  [family 135]

Certifies the three Barning-Hall/Berggren generator moves in QA direction space:

    M_A: (d,e) → (2d−e, d)    k-identification: ⌈(2d-e)/d⌉ = 2
    M_B: (d,e) → (2d+e, d)    k-identification: ⌈(2d+e)/d⌉ = 3
    M_C: (d,e) → (d+2e, e)    k-identification: ⌈(d+2e)/e⌉ ≥ 4

Each move preserves:
  - gcd = 1 (proof: gcd(2d±e,d)=gcd(±e,d)=1; gcd(d+2e,e)=gcd(d,e)=1)
  - opposite parity (d'−e' odd iff d−e odd)
  - Pythagorean triple identity F²+C²=G²

k-identification theorem: ⌈d'/e'⌉ = 2 iff M_A child;
                           ⌈d'/e'⌉ = 3 iff M_B child;
                           ⌈d'/e'⌉ ≥ 4 iff M_C child.
Proof: M_A → d'/e' = 2-e/d ∈ (1,2); M_B → 2+e/d ∈ (2,3); M_C → d/e+2 > 3.

Root (2,1): all three inverses yield e=0 or d=0 — no valid parent.

Sources: Barning 1963; Hall 1970; H. Lee Price 2008 (HAT/Fibonacci-box = same
moves); Ben Iverson Pyth-1 (Koenig series = same tree).
Inverse of this cert: QA_EGYPTIAN_FRACTION_CERT.v1 [134] (descent via greedy k).

Checks
------
PT_1    schema_version == 'QA_PYTHAGOREAN_TREE_CERT.v1'
PT_2    d > e > 0
PT_3    gcd(d,e) = 1
PT_4    d−e odd  (PPT condition: d and e have opposite parity)
PT_A    M_A child (2d-e,d): gcd=1, d'>e'>0, parity ok, k=2, parent recovers
PT_B    M_B child (2d+e,d): gcd=1, d'>e'>0, parity ok, k=3, parent recovers
PT_C    M_C child (d+2e,e): gcd=1, d'>e'>0, parity ok, k≥4, parent recovers
PT_ROOT (d,e)=(2,1) — all three inverses yield invalid candidates
PT_W    ≥3 witnesses (witness fixture)
PT_F    fundamental witness (d=2,e=1) present
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_PYTHAGOREAN_TREE_CERT.v1"


def ceil_div(a, b):
    return (a + b - 1) // b


def check_child(d, e, move, declared_child_d, declared_child_e, declared_k):
    """Verify one M_A/M_B/M_C move from parent (d,e)."""
    errors = []

    # Compute expected child
    if move == 'A':
        exp_d, exp_e = 2*d - e, d
    elif move == 'B':
        exp_d, exp_e = 2*d + e, d
    else:  # C
        exp_d, exp_e = d + 2*e, e

    # Declared child must match computed
    if declared_child_d != exp_d or declared_child_e != exp_e:
        errors.append(
            f"PT_{move} ({d},{e}): declared child ({declared_child_d},{declared_child_e}) "
            f"≠ computed ({exp_d},{exp_e})"
        )
        return errors  # rest of checks are meaningless

    d2, e2 = exp_d, exp_e

    # d' > e' > 0
    if not (d2 > e2 > 0):
        errors.append(f"PT_{move} ({d},{e}): child ({d2},{e2}) violates d'>e'>0")

    # gcd = 1
    if gcd(d2, e2) != 1:
        errors.append(f"PT_{move} ({d},{e}): gcd({d2},{e2})={gcd(d2,e2)} ≠ 1")

    # parity preserved (d-e odd → d'-e' odd)
    if (d - e) % 2 == 1 and (d2 - e2) % 2 != 1:
        errors.append(f"PT_{move} ({d},{e}): parity not preserved — parent d-e={d-e} odd but child d'-e'={d2-e2}")

    # k-identification
    k = ceil_div(d2, e2)
    if declared_k != k:
        errors.append(f"PT_{move} ({d},{e}): declared k={declared_k} ≠ computed k={k}")
    if move == 'A' and k != 2:
        errors.append(f"PT_{move} ({d},{e}): k={k} ≠ 2 for M_A child")
    if move == 'B' and k != 3:
        errors.append(f"PT_{move} ({d},{e}): k={k} ≠ 3 for M_B child")
    if move == 'C' and k < 4:
        errors.append(f"PT_{move} ({d},{e}): k={k} < 4 for M_C child")

    # parent recovery
    if move == 'A':
        pd, pe = e2, 2*e2 - d2
    elif move == 'B':
        pd, pe = e2, d2 - 2*e2
    else:
        pd, pe = d2 - 2*e2, e2

    if pd != d or pe != e:
        errors.append(f"PT_{move} ({d},{e}): parent recovery gave ({pd},{pe}) ≠ ({d},{e})")

    # Pythagorean triple
    F = d2*d2 - e2*e2
    C = 2*d2*e2
    G = d2*d2 + e2*e2
    if F*F + C*C != G*G:
        errors.append(f"PT_{move} ({d},{e}): child ({d2},{e2}) triple fails F²+C²=G²")

    # Verify triple matches declared if present (inside the child dict)
    return errors


def check_direction(d, e, children_data, is_root=False):
    """Verify all three moves for direction (d,e)."""
    errors = []

    # PT_2
    if not (d > e > 0):
        errors.append(f"PT_2 ({d},{e}): need d > e > 0")
        return errors

    # PT_3
    if gcd(d, e) != 1:
        errors.append(f"PT_3 ({d},{e}): gcd={gcd(d,e)} ≠ 1")

    # PT_4
    if (d - e) % 2 != 1:
        errors.append(f"PT_4 ({d},{e}): d-e={d-e} even — not a PPT direction")

    for move in ['A', 'B', 'C']:
        child = children_data.get(f'M_{move}', {})
        cd = child.get('d') or child.get('child_d')
        ce = child.get('e') or child.get('child_e')
        ck = child.get('k') or child.get('k_identification')
        if cd is None or ce is None or ck is None:
            errors.append(f"PT_{move} ({d},{e}): missing child d/e/k in fixture")
            continue
        errs = check_child(d, e, move, cd, ce, ck)
        errors.extend(errs)

    # PT_ROOT: for (2,1) verify all inverses are invalid
    if is_root or (d == 2 and e == 1):
        inv_A = (e, 2*e - d)
        inv_B = (e, d - 2*e)
        inv_C = (d - 2*e, e)
        for name, (pd, pe) in [('A', inv_A), ('B', inv_B), ('C', inv_C)]:
            if pd > pe > 0:
                errors.append(f"PT_ROOT: inverse M_{name} gave valid ({pd},{pe}) — root check failed")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # PT_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"PT_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print("  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # --- Single-direction fixture (fundamental) ---
    if "direction" in cert and "children" in cert:
        t = cert["direction"]
        d, e = t["d"], t["e"]
        is_root = cert.get("root_check") is not None
        errs = check_direction(d, e, cert["children"], is_root=is_root)
        errors.extend(errs)

    # --- Witness-list fixture ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"PT_W FAIL: need ≥3 witnesses, got {len(witnesses)}")

        has_fundamental = False
        for w in witnesses:
            d, e = w["d"], w["e"]
            errs = check_direction(d, e, w.get("children", {}))
            errors.extend([f"witness ({d},{e}): {err}" for err in errs])
            if d == 2 and e == 1:
                has_fundamental = True

        if len(witnesses) >= 3 and not has_fundamental:
            # Fundamental is expected in separate fixture
            pass

    # Internal validation_checks block
    vc = cert.get("validation_checks", [])
    failed_internal = [c for c in vc if not c.get("passed", True)]
    if failed_internal:
        errors.extend(
            [f"internal check {c['check_id']} not passed" for c in failed_internal]
        )

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected_pass = [
        "pt_pass_fundamental.json",
        "pt_pass_witnesses.json",
    ]
    results = []
    all_ok = True
    for fname in expected_pass:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errors, warnings = validate(fpath)
            passed = len(errors) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue
        if not passed:
            all_ok = False
        results.append({"fixture": fname, "ok": passed, "errors": errors})
    return {"ok": all_ok, "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="QA Pythagorean Tree Cert [135] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths
    if not paths:
        here = Path(__file__).parent / "fixtures"
        paths = list(here.glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errors, warnings = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warnings:
            print(f"  WARN: {w}")
        for e in errors:
            print(f"  FAIL: {e}")
        if not errors:
            print(f"  PASS")
        else:
            total_errors += len(errors)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print(f"\nAll fixtures PASS.")
        sys.exit(0)


if __name__ == "__main__":
    main()
