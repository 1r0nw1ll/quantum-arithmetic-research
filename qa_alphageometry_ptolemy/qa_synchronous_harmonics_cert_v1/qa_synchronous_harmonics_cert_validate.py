#!/usr/bin/env python3
"""QA Synchronous Harmonics Cert family [147] — certifies the synchronization
and interference rules for coprime and par-classified wavelets.

Core theorems (Iverson, Pyth-2 Ch XIII, QA-2 Ch 6):

1. SYNC: Two coprime periods m,n synchronize at time = m*n (minimum).
   Non-coprime periods synchronize at LCM(m,n) < m*n.

2. PAR INTERFERENCE: For odd wavelets classified by par = n mod 4:
   - 3-par (4k+3): HIGH at 3/4 mark, LOW at 1/4 mark
   - 5-par (4k+1): HIGH at 1/4 mark, LOW at 3/4 mark
   - Same-par pairs SUPPORT (constructive at quarter-points)
   - Cross-par pairs OPPOSE (destructive at quarter-points)

3. PRODUCT-OF-6: All QA Quantum Number products are multiples of 6
   (because 2 and 3 are always factors in the bead numbers).

Checks: SH_1 (schema), SH_SYNC (coprime→product; non-coprime→LCM<product),
SH_PAR (par classification correct; same-par support, cross-par oppose),
SH_PROD6 (QN products divisible by 6), SH_W (>=5 pair witnesses),
SH_F (fundamental: periods 3,5 synchronize at 15).
"""

import json
import os
import sys
from math import gcd


SCHEMA = "QA_SYNCHRONOUS_HARMONICS_CERT.v1"
VALID_PAR = frozenset([1, 2, 3, 0])  # mod 4 residues: 5-par=1, 2-par=2, 3-par=3, 4-par=0


def par_class(n):
    """Par classification: 2-par=4k+2, 3-par=4k+3, 4-par=4k, 5-par=4k+1."""
    return n % 4


def par_name(n):
    r = n % 4
    return {0: "4-par", 1: "5-par", 2: "2-par", 3: "3-par"}[r]


def lcm(a, b):
    return a * b // gcd(a, b)


def par_sign(n):
    """Quarter-point phase sign for odd wavelet of period n.
    5-par (n%4==1): +1 at 1/4 mark (HIGH).
    3-par (n%4==3): -1 at 1/4 mark (LOW).
    Returns +1 or -1 for odd wavelets, 0 for even."""
    r = n % 4
    if r == 1:
        return 1    # 5-par: HIGH at 1/4
    elif r == 3:
        return -1   # 3-par: LOW at 1/4
    else:
        return 0    # even wavelets (2-par, 4-par) — not in this rule


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    # SH_1 — schema
    if cert.get("schema_version") != SCHEMA:
        err("SH_1", f"schema_version must be {SCHEMA}")

    # --- SH_SYNC: synchronization pairs ---
    sync_pairs = cert.get("sync_pairs", [])
    has_fundamental = False

    for i, sp in enumerate(sync_pairs):
        m, n = sp.get("m", 0), sp.get("n", 0)
        if m <= 0 or n <= 0:
            err("SH_SYNC", f"sync_pair[{i}] invalid: m={m}, n={n}")
            continue

        g = gcd(m, n)
        is_coprime = (g == 1)
        sync_time = lcm(m, n)

        if m == 3 and n == 5:
            has_fundamental = True

        # Check declared coprimality
        decl_coprime = sp.get("coprime")
        if decl_coprime is not None and decl_coprime != is_coprime:
            err("SH_SYNC", f"sync_pair[{i}] ({m},{n}): declared coprime={decl_coprime} but gcd={g}")

        # Check declared sync_time
        decl_sync = sp.get("sync_time")
        if decl_sync is not None and decl_sync != sync_time:
            err("SH_SYNC", f"sync_pair[{i}] ({m},{n}): declared sync_time={decl_sync} but LCM={sync_time}")

        # Coprime → sync at product; non-coprime → LCM < product
        if is_coprime:
            if sync_time != m * n:
                err("SH_SYNC", f"sync_pair[{i}] ({m},{n}): coprime but LCM={sync_time} != product={m*n}")
        else:
            if sync_time >= m * n:
                err("SH_SYNC", f"sync_pair[{i}] ({m},{n}): non-coprime but LCM={sync_time} >= product={m*n}")

    # --- SH_PAR: par interference pairs ---
    par_pairs = cert.get("par_pairs", [])

    for i, pp in enumerate(par_pairs):
        p1, p2 = pp.get("p1", 0), pp.get("p2", 0)
        if p1 <= 0 or p2 <= 0:
            err("SH_PAR", f"par_pair[{i}] invalid: p1={p1}, p2={p2}")
            continue

        # Check par classification
        decl_par1 = pp.get("par1")
        decl_par2 = pp.get("par2")
        if decl_par1 is not None and decl_par1 != par_name(p1):
            err("SH_PAR", f"par_pair[{i}] p1={p1}: declared {decl_par1} but actual {par_name(p1)}")
        if decl_par2 is not None and decl_par2 != par_name(p2):
            err("SH_PAR", f"par_pair[{i}] p2={p2}: declared {decl_par2} but actual {par_name(p2)}")

        # Check interference type
        s1, s2 = par_sign(p1), par_sign(p2)
        if s1 == 0 or s2 == 0:
            # Even wavelets: par interference rule only applies to odd wavelets
            continue

        computed_interference = "SUPPORT" if s1 == s2 else "OPPOSE"
        decl_interference = pp.get("interference")
        if decl_interference is not None and decl_interference != computed_interference:
            err("SH_PAR", f"par_pair[{i}] ({p1},{p2}): declared {decl_interference} but {par_name(p1)}×{par_name(p2)}={computed_interference}")

    # --- SH_PROD6: QN products divisible by 6 ---
    qn_witnesses = cert.get("qn_witnesses", [])
    for i, qn in enumerate(qn_witnesses):
        b, e = qn.get("b", 0), qn.get("e", 0)
        d_val = b + e
        a_val = b + 2 * e
        product = b * e * d_val * a_val
        decl_product = qn.get("product")
        if decl_product is not None and decl_product != product:
            err("SH_PROD6", f"qn[{i}] ({b},{e}): declared product={decl_product} but computed={product}")
        if product % 6 != 0:
            err("SH_PROD6", f"qn[{i}] ({b},{e}): product={product} not divisible by 6")

    # --- SH_W: witness counts ---
    total = len(sync_pairs) + len(par_pairs)
    if total < 5:
        err("SH_W", f"need >=5 total witnesses, got {total}")

    # --- SH_F: fundamental ---
    if not has_fundamental:
        err("SH_F", "no sync_pair with (3,5) — the fundamental coprime odd pair")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


# --- Self-test ---

def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "sh_pass_sync_and_par.json": True,
        "sh_pass_qn_products.json": True,
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
        print("Usage: python qa_synchronous_harmonics_cert_validate.py [--self-test | <fixture.json>]")
