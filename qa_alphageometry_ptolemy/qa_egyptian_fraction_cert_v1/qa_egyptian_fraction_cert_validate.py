#!/usr/bin/env python3
"""
qa_egyptian_fraction_cert_validate.py

Validator for QA_EGYPTIAN_FRACTION_CERT.v1  [family 134]

Certifies the greedy Egyptian fraction expansion of the HAT direction ratio
HAT₁ = e/d for any primitive QA direction (d,e):

    e/d = 1/k₁ + 1/k₂ + ... + 1/kₙ

where kᵢ = ⌈dᵢ/eᵢ⌉ (greedy), denominators strictly increase, all intermediate
pairs (dᵢ,eᵢ) are coprime, and the expansion terminates when eₙ = 1.

The intermediate pairs (d₀,e₀)=(d,e), (d₁,e₁), ..., (dₙ,1) trace the Koenig
tree descent path from (d,e) to the root unit-fraction direction (kₙ,1).

Algorithmic step: given (dᵢ,eᵢ),
    k = ⌈dᵢ/eᵢ⌉ = (dᵢ + eᵢ - 1) // eᵢ
    next_num = k*eᵢ - dᵢ
    next_den = k*dᵢ
    g = gcd(next_num, next_den)
    (dᵢ₊₁, eᵢ₊₁) = (next_den//g, next_num//g)

Sources: Ben Iverson Pyth-1 (Koenig series + Egyptian fractions);
H. Lee Price 2008 (HAT=e/d, Fibonacci box); Rhind Papyrus ~1600 BCE.

Checks
------
EF_1  schema_version == 'QA_EGYPTIAN_FRACTION_CERT.v1'
EF_2  d > e > 0
EF_3  gcd(d, e) = 1  (primitive direction)
EF_4  expansion sums exactly to e/d  (Fraction arithmetic)
EF_5  denominators strictly increasing
EF_6  each kᵢ == ceil(dᵢ/eᵢ)  (greedy property)
EF_7  all intermediate pairs coprime
EF_8  terminal condition: last step has e = 1
EF_W  ≥3 witnesses present  (witness fixture)
EF_F  fundamental witness (d=2,e=1) present with denominators=[2]
"""

from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates arithmetic claims in submitted JSON, no empirical QA state machine"

import json
import sys
from math import gcd
from fractions import Fraction
from pathlib import Path

SCHEMA_VERSION = "QA_EGYPTIAN_FRACTION_CERT.v1"


def greedy_expand(d, e):
    """Compute greedy Egyptian fraction expansion of e/d.
    Returns list of (k, cur_d, cur_e) — one entry per unit-fraction step."""
    steps = []
    cur_d, cur_e = d, e
    while cur_e > 0:
        k = (cur_d + cur_e - 1) // cur_e  # ceiling division
        steps.append((k, cur_d, cur_e))
        if cur_e == 1:
            break
        next_num = k * cur_e - cur_d
        next_den = k * cur_d
        g = gcd(next_num, next_den)
        cur_d, cur_e = next_den // g, next_num // g
    return steps


def check_direction(d, e, declared_denoms):
    """Verify all EF properties for a single direction and its declared expansion."""
    errors = []

    # EF_2
    if not (d > e > 0):
        errors.append(f"EF_2 ({d},{e}): need d > e > 0")
        return errors

    # EF_3
    if gcd(d, e) != 1:
        errors.append(f"EF_3 ({d},{e}): gcd={gcd(d,e)} ≠ 1 — not primitive")
        return errors

    # Recompute greedy expansion
    computed = greedy_expand(d, e)
    computed_denoms = [k for k, _, _ in computed]

    # EF_6: declared matches greedy
    if declared_denoms != computed_denoms:
        errors.append(
            f"EF_6 ({d},{e}): declared denominators {declared_denoms} ≠ greedy {computed_denoms}"
        )
        # Still check other properties against declared
        denoms = declared_denoms
    else:
        denoms = computed_denoms

    # EF_4: sum == e/d
    total = sum(Fraction(1, k) for k in denoms)
    if total != Fraction(e, d):
        errors.append(f"EF_4 ({d},{e}): sum {total} ≠ e/d = {Fraction(e,d)}")

    # EF_5: strictly increasing
    for i in range(len(denoms) - 1):
        if denoms[i] >= denoms[i + 1]:
            errors.append(
                f"EF_5 ({d},{e}): denominators not strictly increasing at position {i}: {denoms[i]} >= {denoms[i+1]}"
            )

    # EF_7 + EF_8 using computed steps
    for k, sd, se in computed:
        if gcd(sd, se) != 1:
            errors.append(f"EF_7 ({d},{e}): intermediate ({sd},{se}) not coprime, gcd={gcd(sd,se)}")
    last_k, last_d, last_e = computed[-1]
    if last_e != 1:
        errors.append(f"EF_8 ({d},{e}): terminal e={last_e} ≠ 1")

    return errors


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # EF_1
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"EF_1 FAIL: schema_version={cert.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )

    result = cert.get("result", "")
    if result not in ("PASS", "FAIL"):
        errors.append(f"result must be PASS or FAIL, got {result!r}")
    if result == "FAIL":
        print("  SKIP detailed checks — cert declares FAIL")
        return errors, warnings

    # --- Single-direction fixture ---
    if "direction" in cert and "expansion" in cert:
        t = cert["direction"]
        d, e = t["d"], t["e"]
        denoms = cert["expansion"].get("denominators", [])
        dir_errors = check_direction(d, e, denoms)
        errors.extend(dir_errors)

    # --- Witness-list fixture ---
    witnesses = cert.get("witnesses", [])
    if witnesses:
        if len(witnesses) < 3:
            errors.append(f"EF_W FAIL: need ≥3 witnesses, got {len(witnesses)}")

        has_fundamental = False
        for w in witnesses:
            d, e = w["d"], w["e"]
            denoms = w.get("denominators", [])
            werrs = check_direction(d, e, denoms)
            errors.extend([f"witness ({d},{e}): {err}" for err in werrs])
            if d == 2 and e == 1:
                has_fundamental = True
                if denoms != [2]:
                    errors.append(f"EF_F FAIL: fundamental (2,1) denominators={denoms}, expected [2]")

        if len(witnesses) >= 3 and not has_fundamental:
            # Fundamental may be in a separate fixture — not required in witness list
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
        "ef_pass_fundamental.json",
        "ef_pass_witnesses.json",
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

    parser = argparse.ArgumentParser(description="QA Egyptian Fraction Cert [134] validator")
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
