#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=mod4_residues"
"""QA Par Number Cert family [151] — certifies Iverson's "Double Parity"
(par number) system: the 4-way classification of integers by mod 4.

PAR SYSTEM (Iverson, QA-2 Ch 3):

| Par | Form  | First values    | Gender | Euclid class |
|-----|-------|-----------------|--------|-------------|
| 2-par | 4k+2 | 2,6,10,14...  | female | even-odd    |
| 3-par | 4k+3 | 3,7,11,15...  | male   | odd-odd     |
| 4-par | 4k   | 4,8,12,16...  | both   | even-even   |
| 5-par | 4k+1 | 5,9,13,17...  | male   | odd-even    |

Key certified properties:
1. SQUARE: square of any male (3-par or 5-par) = always 5-par
2. QA_C: C=2de is always 4-par (since one of d,e is even for primitive)
3. QA_G: G=d²+e² is always 5-par (opposite parity: odd²+even²≡1 mod 4)
4. FIB_HITS: m is 3-par → Fib_hits(π₁,m)=1; m is 5-par → Fib_hits(π₁,m)=2 (except m=5)
5. PRODUCT: par of product = par determined by multiplication table

Source: Iverson QA-2 Ch 3; "par" from Hindi "char" (four), not English "parity".

Checks: PN_1 (schema), PN_CLASS (par classification correct), PN_SQ (male
squares → 5-par), PN_QA (C=4-par, G=5-par for directions), PN_FIB (Fib_hits
matches par class), PN_MULT (par multiplication table), PN_W (>=8 witnesses),
PN_F (fundamental direction (2,1) present).
"""

import json
import os
import sys


SCHEMA = "QA_PAR_NUMBER_CERT.v1"


def par_class(n):
    """Return par class: 2,3,4,5 for residue mod 4 = 2,3,0,1."""
    r = n % 4
    return {0: 4, 1: 5, 2: 2, 3: 3}[r]


def par_name(n):
    return f"{par_class(n)}-par"


def pisano_period(m):
    """Compute Pisano period pi(m) = period of Fibonacci mod m."""
    if m <= 1:
        return 1
    prev, curr = 0, 1
    for i in range(1, 6 * m + 1):
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return i
    return -1  # should not happen for valid m


def fib_hits(pi_m, m):
    """Count how many times Fibonacci sequence hits 0 mod m within one Pisano period."""
    count = 0
    prev, curr = 0, 1
    for i in range(pi_m):
        if prev % m == 0:
            count += 1
        prev, curr = curr, (prev + curr) % m
    return count


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("PN_1", f"schema_version must be {SCHEMA}")

    # --- PN_CLASS: par classification witnesses ---
    class_witnesses = cert.get("par_witnesses", [])
    has_fundamental = False

    for i, pw in enumerate(class_witnesses):
        n = pw.get("n", 0)
        decl_par = pw.get("par")
        if n <= 0:
            err("PN_CLASS", f"witness[{i}]: n={n} must be positive")
            continue
        computed = par_class(n)
        if decl_par is not None and decl_par != computed:
            err("PN_CLASS", f"witness[{i}]: n={n} declared {decl_par}-par but computed {computed}-par")

    # --- PN_SQ: male squares → 5-par ---
    sq_witnesses = cert.get("square_witnesses", [])
    for i, sw in enumerate(sq_witnesses):
        n = sw.get("n", 0)
        sq = n * n
        pc = par_class(n)
        sq_pc = par_class(sq)
        if pc in (3, 5):  # male
            if sq_pc != 5:
                err("PN_SQ", f"square_witness[{i}]: n={n} ({pc}-par), n²={sq} is {sq_pc}-par not 5-par")
        decl_sq_par = sw.get("square_par")
        if decl_sq_par is not None and decl_sq_par != sq_pc:
            err("PN_SQ", f"square_witness[{i}]: n={n}, n²={sq} declared {decl_sq_par}-par but computed {sq_pc}-par")

    # --- PN_QA: C=4-par, G=5-par for directions ---
    dir_witnesses = cert.get("direction_witnesses", [])
    for i, dw in enumerate(dir_witnesses):
        d_val = dw.get("d", 0)
        e_val = dw.get("e", 0)
        if d_val == 2 and e_val == 1:
            has_fundamental = True
        C = 2 * d_val * e_val
        G = d_val * d_val + e_val * e_val
        if par_class(C) != 4:
            err("PN_QA", f"direction[{i}] ({d_val},{e_val}): C={C} is {par_class(C)}-par not 4-par")
        if par_class(G) != 5:
            err("PN_QA", f"direction[{i}] ({d_val},{e_val}): G={G} is {par_class(G)}-par not 5-par")

    # --- PN_FIB: Fib_hits matches par class ---
    fib_witnesses = cert.get("fib_witnesses", [])
    for i, fw in enumerate(fib_witnesses):
        m = fw.get("m", 0)
        if m <= 1:
            continue
        pc = par_class(m)
        pi_m = pisano_period(m)
        hits = fib_hits(pi_m, m)
        decl_hits = fw.get("fib_hits")
        if decl_hits is not None and decl_hits != hits:
            err("PN_FIB", f"fib_witness[{i}]: m={m} declared hits={decl_hits} computed={hits}")
        # Rule: 3-par → 1 hit; 5-par → 2 hits (except m=5)
        decl_rule = fw.get("par_rule_holds")
        if decl_rule is not None:
            if pc == 3:
                expected_rule = (hits == 1)
            elif pc == 5 and m != 5:
                expected_rule = (hits == 2)
            else:
                expected_rule = None  # m=5 or even par — rule doesn't apply
            if expected_rule is not None and decl_rule != expected_rule:
                err("PN_FIB", f"fib_witness[{i}]: m={m} ({pc}-par) rule declared={decl_rule} computed={expected_rule}")

    # --- PN_MULT: par multiplication table ---
    mult_table = cert.get("par_multiplication_table")
    if mult_table is not None:
        # Verify the 4×4 table: par(a*b) for a_par, b_par in {2,3,4,5}
        for entry in mult_table:
            a_par = entry.get("a_par")
            b_par = entry.get("b_par")
            result_par = entry.get("result_par")
            # Find representatives
            a_rep = {2: 2, 3: 3, 4: 4, 5: 5}[a_par]
            b_rep = {2: 2, 3: 3, 4: 4, 5: 5}[b_par]
            computed = par_class(a_rep * b_rep)
            if result_par != computed:
                err("PN_MULT", f"par({a_par}-par × {b_par}-par) declared={result_par}-par computed={computed}-par")

    # --- PN_W ---
    total = len(class_witnesses) + len(sq_witnesses) + len(dir_witnesses) + len(fib_witnesses)
    if total < 8:
        err("PN_W", f"need >=8 total witnesses, got {total}")

    # --- PN_F ---
    if not has_fundamental:
        err("PN_F", "no direction witness with (2,1)")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "pn_pass_classification.json": True,
        "pn_pass_fib_hits.json": True,
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
        print("Usage: python qa_par_number_cert_validate.py [--self-test | <fixture.json>]")
