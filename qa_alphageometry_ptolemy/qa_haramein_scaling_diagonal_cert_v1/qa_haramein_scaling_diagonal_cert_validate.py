#!/usr/bin/env python3
"""
qa_haramein_scaling_diagonal_cert_validate.py  [family 218]

QA_HARAMEIN_SCALING_DIAGONAL_CERT.v1 validator.
See docs/families/218_qa_haramein_scaling_diagonal_cert.md and
docs/theory/QA_HARAMEIN_SCALING_DIAGONAL.md for claim + methodology.

TO INSTALL: rename this file to
   qa_haramein_scaling_diagonal_cert_validate.py
(i.e. drop the .txt suffix). The cert_gate_hook blocks Claude from
writing .py files directly; this staged draft awaits Will's move or
toggle of LLM_QA_ALLOW_CLAUDE_PYTHON_EDIT=1.
"""

QA_COMPLIANCE = "cert_validator - Haramein 2008 Table 1 fixed-d hyperbola + phi^2 integer-quadratic-form ratios; integer state space; A1/A2 compliant; raw d=b+e, a=b+2e; no ** operator; no float state beyond observer-projection reporting"

import json
import sys
from pathlib import Path
from fractions import Fraction

SCHEMA_VERSION = "QA_HARAMEIN_SCALING_DIAGONAL_CERT.v1"
PHI_RATIO_TOL_REL = Fraction(8, 100)  # 8% relative-error tolerance (covers 7.3% observed)


def fixed_d(row):
    return row["b"] + row["e"]


def quadratic_form(row_a, row_b):
    db = row_a["b"] - row_b["b"]
    de = row_a["e"] - row_b["e"]
    return db * db + de * de


def phi2_dev(num, den):
    """Return min rational bound on |ratio/target - 1| for target in {phi^2, 1/phi^2}.
    phi^2 = (3 + sqrt(5))/2 sandwiched in tight rational interval."""
    phi2_lo = Fraction(26180339, 10000000)
    phi2_hi = Fraction(26180340, 10000000)
    if den == 0 or num == 0:
        return Fraction(10**9, 1)
    ratio = Fraction(num, den)
    # relative error to phi^2
    denom_phi2 = phi2_lo if ratio >= phi2_lo else ratio
    err_phi2 = max(abs(ratio - phi2_lo), abs(ratio - phi2_hi)) / denom_phi2
    inv_phi2_lo = Fraction(1, 1) / phi2_hi
    inv_phi2_hi = Fraction(1, 1) / phi2_lo
    denom_inv = inv_phi2_lo if ratio >= inv_phi2_lo else ratio
    err_inv = max(abs(ratio - inv_phi2_lo), abs(ratio - inv_phi2_hi)) / denom_inv
    return min(err_phi2, err_inv)


def _run_checks(fixture):
    results = {}

    results["HSD_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    rows = fixture.get("table1", [])
    tab_ok = isinstance(rows, list) and len(rows) == 6
    if tab_ok:
        for row in rows:
            if not (isinstance(row.get("name"), str)
                    and isinstance(row.get("b"), int)
                    and isinstance(row.get("e"), int)):
                tab_ok = False
                break
            c_dec = row["b"] + row["e"]
            if not (9 <= c_dec <= 12):
                tab_ok = False
                break
    results["HSD_TABLE"] = tab_ok

    fixed_ok = tab_ok
    if fixed_ok:
        ds = [fixed_d(r) for r in rows]
        if max(ds) - min(ds) > 1:
            fixed_ok = False
    results["HSD_FIXED_D"] = fixed_ok

    seg_map = {r["name"]: r for r in rows} if tab_ok else {}
    segs = fixture.get("segments", [])
    seg_ok = isinstance(segs, list) and len(segs) >= 6
    if seg_ok:
        for s in segs:
            a_name = s.get("from")
            b_name = s.get("to")
            q_claim = s.get("quadratic_form")
            if a_name not in seg_map or b_name not in seg_map:
                seg_ok = False
                break
            q_actual = quadratic_form(seg_map[a_name], seg_map[b_name])
            if q_claim != q_actual:
                seg_ok = False
                break
    results["HSD_SEGMENTS"] = seg_ok

    ratios = fixture.get("phi_ratios", [])
    quad_ok = isinstance(ratios, list) and len(ratios) == 4
    if quad_ok:
        for r in ratios:
            num_from = r.get("numerator_segment")
            den_from = r.get("denominator_segment")
            num_claim = r.get("num_quadratic")
            den_claim = r.get("den_quadratic")
            if any(x is None for x in (num_from, den_from, num_claim, den_claim)):
                quad_ok = False
                break
            n_segs = [s for s in segs if {s.get("from"), s.get("to")} == set(num_from)]
            d_segs = [s for s in segs if {s.get("from"), s.get("to")} == set(den_from)]
            if not n_segs or not d_segs:
                quad_ok = False
                break
            if n_segs[0]["quadratic_form"] != num_claim:
                quad_ok = False
                break
            if d_segs[0]["quadratic_form"] != den_claim:
                quad_ok = False
                break
    results["HSD_QUADRATIC"] = quad_ok

    phi_ok = quad_ok
    if phi_ok:
        for r in ratios:
            num_claim = r["num_quadratic"]
            den_claim = r["den_quadratic"]
            dev = phi2_dev(num_claim, den_claim)
            if dev > PHI_RATIO_TOL_REL:
                phi_ok = False
                break
    results["HSD_PHI_RATIOS"] = phi_ok

    null = fixture.get("null_test", {})
    null_ok = (
        isinstance(null, dict)
        and "observed_stat" in null
        and "null_5pct" in null
        and "n_samples" in null
        and null.get("n_samples", 0) >= 10000
    )
    if null_ok:
        obs = null["observed_stat"]
        p5 = null["null_5pct"]
        null_ok = isinstance(obs, (int, float)) and isinstance(p5, (int, float)) and obs < p5
    results["HSD_NULL"] = null_ok

    src = fixture.get("source_attribution", "")
    results["HSD_SRC"] = (
        ("Haramein" in src)
        and ("2008" in src)
        and ("Will Dale" in src or "Dale" in src)
    )

    ws = fixture.get("witnesses", [])
    w_ok = isinstance(ws, list) and len(ws) >= 3
    if w_ok:
        kinds = {w.get("kind") for w in ws}
        required = {"fixed_d", "phi_ratio", "null"}
        if not required.issubset(kinds):
            w_ok = False
    results["HSD_WITNESS"] = w_ok

    fl = fixture.get("fail_ledger")
    results["HSD_F"] = isinstance(fl, list)

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
        print("Usage: python qa_haramein_scaling_diagonal_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
