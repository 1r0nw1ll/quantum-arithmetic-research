#!/usr/bin/env python3
"""
qa_madelung_d_ordering_cert_validate.py  [family 220]

QA_MADELUNG_D_ORDERING_CERT.v1 validator.

Claim: atomic (n, l) subshells with n>=1, 0<=l<=n-1, identified as QA (b, e) = (n, l),
satisfy d = b + e = n + l = Madelung quantum. The aufbau filling order = QA (d, -e)
ascending sort. Selection rule: within-d antidiagonal (b,e)->(b+1,e-1); between-d jump
(b,0)->(ceil((b+2)/2), floor(b/2)). Derived: pop=4e+2; period-k pop = 2*ceil((k+1)/2)^2;
shell-n total = 2n^2.
"""

QA_COMPLIANCE = "cert_validator - Madelung aufbau = QA (d,-e) sort on (n,l) lattice; integer state space; A1/A2 compliant (n>=1 so b>=1 always; l>=0 so e>=0; constraint l<=n-1 enforces b>=e+1>=1); raw d=b+e, a=b+2e; no ** operator; no float state"

import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_MADELUNG_D_ORDERING_CERT.v1"


def qa_madelung_sort(n_subshells):
    """Generate first n_subshells Madelung (n, l) pairs by QA (d, -e) ascending sort."""
    cands = []
    n_max = n_subshells + 5
    for n in range(1, n_max):
        for l in range(n):
            cands.append((n + l, -l, n, l))
    cands.sort()
    return [(n, l) for _, _, n, l in cands[:n_subshells]]


def is_within_d_step(a, b):
    (b1, e1) = a
    (b2, e2) = b
    return (b1 + e1) == (b2 + e2) and b2 == b1 + 1 and e2 == e1 - 1


def is_between_d_jump(a, b):
    (b1, e1) = a
    (b2, e2) = b
    if e1 != 0:
        return False
    d1 = b1
    expected_b = (d1 + 2 + 1) // 2
    expected_e = d1 // 2
    return (b2, e2) == (expected_b, expected_e)


def _run_checks(fixture):
    results = {}

    results["MAD_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    seq = fixture.get("madelung_sequence", [])
    seq_ok = isinstance(seq, list) and len(seq) >= 1
    results["MAD_MAPPING"] = seq_ok
    if seq_ok:
        for row in seq:
            needed = ("k", "n", "l", "b", "e", "d", "a", "population")
            if not all(isinstance(row.get(f), int) for f in needed):
                results["MAD_MAPPING"] = False
                break
            n, l, b, e, d, a, pop = (row[f] for f in ("n", "l", "b", "e", "d", "a", "population"))
            if not (n >= 1 and 0 <= l <= n - 1):
                results["MAD_MAPPING"] = False
                break
            if b != n or e != l or d != n + l or a != n + 2 * l:
                results["MAD_MAPPING"] = False
                break

    # MAD_ORDER: independently regenerate QA sort and compare
    if seq_ok:
        regen = qa_madelung_sort(len(seq))
        tabled = [(row["n"], row["l"]) for row in seq]
        results["MAD_ORDER"] = regen == tabled
    else:
        results["MAD_ORDER"] = False

    # MAD_RULE: every transition is within-d step or between-d jump
    if seq_ok and len(seq) >= 2:
        all_ok = True
        for i in range(len(seq) - 1):
            a_pt = (seq[i]["b"], seq[i]["e"])
            b_pt = (seq[i + 1]["b"], seq[i + 1]["e"])
            if not (is_within_d_step(a_pt, b_pt) or is_between_d_jump(a_pt, b_pt)):
                all_ok = False
                break
        results["MAD_RULE"] = all_ok
    else:
        results["MAD_RULE"] = len(seq) >= 1

    # MAD_POP: every row's population = 4e+2
    if seq_ok:
        results["MAD_POP"] = all(row["population"] == 4 * row["e"] + 2 for row in seq)
    else:
        results["MAD_POP"] = False

    # MAD_PERIODS
    periods = fixture.get("period_structure", [])
    p_ok = isinstance(periods, list) and len(periods) >= 1
    if p_ok:
        for p in periods:
            k = p.get("period")
            pop = p.get("population")
            if not (isinstance(k, int) and isinstance(pop, int)):
                p_ok = False
                break
            m = (k + 1 + 1) // 2  # ceil((k+1)/2)
            if pop != 2 * m * m:
                p_ok = False
                break
    results["MAD_PERIODS"] = p_ok

    # MAD_SHELL: cumulative total through shell n = 2 * (1^2 + 2^2 + ... + n^2)
    shells = fixture.get("shell_totals", [])
    s_ok = isinstance(shells, list) and len(shells) >= 1
    if s_ok:
        for s in shells:
            n = s.get("n")
            tot = s.get("total")
            if not (isinstance(n, int) and isinstance(tot, int)):
                s_ok = False
                break
            expected = 2 * sum(k * k for k in range(1, n + 1))
            if tot != expected:
                s_ok = False
                break
    results["MAD_SHELL"] = s_ok

    # MAD_CLASS: structural_claim declares d-ordering (by explicit mapping / filling_order keys)
    sc = fixture.get("structural_claim", {})
    cls_ok = isinstance(sc, dict) and "(d, -e)" in sc.get("filling_order", "") and "(b, e) = (n, l)" in sc.get("mapping", "")
    results["MAD_CLASS"] = cls_ok

    # MAD_SRC
    src = fixture.get("source_attribution", "")
    results["MAD_SRC"] = (
        ("Madelung" in src or "Klechkowski" in src)
        and ("Will Dale" in src or "Dale" in src)
    )

    # MAD_WITNESS
    ws = fixture.get("witnesses", [])
    w_ok = isinstance(ws, list) and len(ws) >= 3
    if w_ok:
        kinds = {w.get("kind") for w in ws}
        required = {"exact_sort", "selection_rule", "derived_structure"}
        if not required.issubset(kinds):
            w_ok = False
    results["MAD_WITNESS"] = w_ok

    # MAD_F
    fl = fixture.get("fail_ledger")
    results["MAD_F"] = isinstance(fl, list)

    return results


def validate_fixture(path):
    with open(path) as f:
        fixture = json.load(f)
    checks = _run_checks(fixture)
    expected = fixture.get("result", "PASS")
    all_pass = all(checks.values())
    actual = "PASS" if all_pass else "FAIL"
    ok = actual == expected
    return {"ok": ok, "expected": expected, "actual": actual, "checks": checks}


def self_test():
    fdir = Path(__file__).parent / "fixtures"
    results = {}
    for fp in sorted(fdir.glob("*.json")):
        results[fp.name] = validate_fixture(fp)
    all_ok = all(r["ok"] for r in results.values())
    print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    elif len(sys.argv) > 1:
        r = validate_fixture(sys.argv[1])
        print(json.dumps(r, indent=2))
        sys.exit(0 if r["ok"] else 1)
    else:
        print("Usage: python qa_madelung_d_ordering_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
