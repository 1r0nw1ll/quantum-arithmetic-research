#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=megalithic_fixtures"
"""QA Megalithic Cert family [178] — certifies that megalithic construction
used discrete integer arithmetic consistent with QA framework.

TIER 2 (structural) + TIER 3 (MY/fathom statistical):

  1. MEGALITHIC YARD (MY = 2.72 ft ≈ 0.829 m):
     Confirmed as quantum of length: p=0.00022 (combined Thom 1962+1967, 202 circles).
     Integer MY proximity: mean fractional error 0.213 vs null 0.249, z=-3.54.
     MY=2.72 is optimal quantum (lowest variance across candidates).

  2. MEGALITHIC FATHOM (MF = 2 MY = 5.44 ft):
     74.3% of 202 circles have EVEN nearest-MY → fathom multiples.
     Binomial p < 10^-8. Primary construction unit.

  3. CONSTRUCTION TRIANGLES:
     3:4:5 → QN (1,1,2,3) = first four Fibonacci numbers
     5:12:13 → QN (1,2,3,5) = fundamental QA quantum number
     Only 19% of primitive triples (hyp≤100) are all-Fibonacci QN.
     Megalithic builders chose 2/5 from this 19% subset.

  4. HONEST NEGATIVES:
     - Fibonacci ratios in diameter pairs: p=0.155, NOT significant
     - Mod-9 distribution: uniform (chi²=6.4, p=0.60)
     - Mod-24 non-uniformity is EXPLAINED by fathom preference, not independent

  QA CONNECTION: Megalithic builders used discrete integer arithmetic
  (fathom-based measurement → integer multiples → Fibonacci QN triangles).
  This is proto-QA: discrete states, integer computation, exact geometry.
  The MY is the QA unit normalization; the fathom is the operational unit.

SOURCE: Thom, "The Megalithic Unit of Length" JRSS (1962);
        Thom, "Megalithic Sites in Britain" (1967);
        Crowhurst, Carnac alignments; cert [167] Historical Nav.

Checks
------
MG_1         schema_version == 'QA_MEGALITHIC_CERT.v1'
MG_MY        MY value and p-value documented
MG_FATHOM    fathom preference (even MY count) verified
MG_TRIANGLE  construction triangles are valid Pythagorean QN triples
MG_HONEST    negative results documented
MG_W         at least 3 evidence categories
MG_F         fail detection
"""

import json
import math
import os
import sys

SCHEMA = "QA_MEGALITHIC_CERT.v1"


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("MG_1", f"schema_version must be {SCHEMA}")

    # MG_MY — megalithic yard evidence
    my_data = cert.get("megalithic_yard")
    if my_data is not None:
        my_value = my_data.get("value_ft")
        p_value = my_data.get("p_value")
        n_circles = my_data.get("n_circles")
        if my_value is None or p_value is None:
            err("MG_MY", "missing MY value_ft or p_value")
        elif p_value > 0.05:
            err("MG_MY", f"p_value={p_value} > 0.05, not significant")
        if n_circles is not None and n_circles < 20:
            warnings.append(f"MG_MY: only {n_circles} circles, low power")
    else:
        err("MG_MY", "missing megalithic_yard section")

    # MG_FATHOM — even MY preference
    fathom_data = cert.get("megalithic_fathom")
    if fathom_data is not None:
        even_count = fathom_data.get("even_count")
        total = fathom_data.get("total")
        p_value = fathom_data.get("p_value")
        if even_count is not None and total is not None:
            frac = even_count / total if total > 0 else 0
            if frac < 0.6:
                err("MG_FATHOM", f"even fraction={frac:.2f} < 0.6, fathom not dominant")
        if p_value is not None and p_value > 0.01:
            err("MG_FATHOM", f"p_value={p_value} > 0.01")

    # MG_TRIANGLE — construction triangles
    triangles = cert.get("construction_triangles", [])
    for i, tri in enumerate(triangles):
        sides = tri.get("sides")
        qn = tri.get("qn")
        if sides is not None and len(sides) == 3:
            a, b, c = sorted(sides)
            if a * a + b * b != c * c:
                err("MG_TRIANGLE", f"triangle {i}: {a}²+{b}²={a*a+b*b} != {c}²={c*c}")
        if qn is not None:
            bv, ev, dv, av = qn.get("b",0), qn.get("e",0), qn.get("d",0), qn.get("a",0)
            if dv != bv + ev:
                err("MG_TRIANGLE", f"triangle {i}: d={dv} != b+e={bv+ev}")
            if av != bv + 2 * ev:
                err("MG_TRIANGLE", f"triangle {i}: a={av} != b+2e={bv+2*ev}")
            # Check C²+F²=G²
            C = 2 * dv * ev
            F = dv * dv - ev * ev
            G = dv * dv + ev * ev
            if C * C + F * F != G * G:
                err("MG_TRIANGLE", f"triangle {i}: C²+F²={C*C+F*F} != G²={G*G}")

    # MG_HONEST — negative results
    negatives = cert.get("honest_negatives", [])
    if len(negatives) < 1:
        warnings.append("MG_HONEST: no negative results documented — suspiciously clean")

    # MG_W — at least 3 evidence categories
    categories = 0
    if cert.get("megalithic_yard"):
        categories += 1
    if cert.get("megalithic_fathom"):
        categories += 1
    if cert.get("construction_triangles"):
        categories += 1
    if cert.get("honest_negatives"):
        categories += 1
    if categories < 3:
        err("MG_W", f"need >= 3 evidence categories, got {categories}")

    # MG_F
    declared = cert.get("result", "UNKNOWN")
    has_errors = len(errors) > 0
    fail_ledger = cert.get("fail_ledger", [])
    if has_errors and declared == "PASS":
        err("MG_F", f"declared PASS but {len(errors)-1} checks failed")
    if not has_errors and declared == "FAIL" and len(fail_ledger) == 0:
        warnings.append("MG_F: declared FAIL but no fail_ledger and all checks pass")

    return {"ok": not has_errors, "errors": errors, "warnings": warnings, "schema": SCHEMA}


def self_test():
    fixture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    results = {"pass_count": 0, "fail_count": 0, "errors": []}
    for fname in sorted(os.listdir(fixture_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(fixture_dir, fname), "r", encoding="utf-8") as f:
            cert = json.load(f)
        out = validate(cert)
        declared = cert.get("result", "UNKNOWN")
        if declared == "PASS" and out["ok"]:
            results["pass_count"] += 1
        elif declared == "FAIL" and not out["ok"]:
            results["fail_count"] += 1
        else:
            results["errors"].append({"fixture": fname, "declared": declared,
                                       "validator_ok": out["ok"], "issues": out["errors"]})
    results["ok"] = len(results["errors"]) == 0
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"{SCHEMA} validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("cert_file", nargs="?")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)
    if args.cert_file:
        with open(args.cert_file, "r", encoding="utf-8") as f:
            cert = json.load(f)
        result = validate(cert)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
