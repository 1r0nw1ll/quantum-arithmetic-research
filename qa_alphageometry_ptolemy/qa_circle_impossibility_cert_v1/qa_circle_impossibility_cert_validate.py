#!/usr/bin/env python3
"""
qa_circle_impossibility_cert_validate.py

Validator for QA_CIRCLE_IMPOSSIBILITY_CERT.v1  [family 207]

Certifies: No QA state has C=0. The circle is an observer projection
of an ellipsoid where C lies along the viewing axis (Will Dale, 2026-04-08).

Proof:
    d = b + e (raw derived coordinate, A2).
    C = 2de = 2(b+e)e.
    By A1 (No-Zero): b >= 1, e >= 1, so d = b+e >= 2, and e >= 1.
    Therefore C = 2de >= 2*2*1 = 4 for all QA states.
    Also d = b+e > e always (b >= 1), so F = d*d - e*e > 0 always.
    Every QA state defines a proper (non-degenerate) conic with C >= 4.
    A true circle (C=0) is structurally impossible.

CRITICAL: QA elements (C, F, G, etc.) are computed from RAW derived
coordinates d = b+e, a = b+2e — NOT from the mod-reduced coordinates
used by the T-operator step function. Modular reduction is for state
evolution, not element computation.

Checks:
    CI_1        — schema_version matches
    CI_C_MIN    — minimum C value is >= 4 (claimed and verified)
    CI_EXHAUSTIVE — exhaustive verification: all_C_ge_4 == true
    CI_PROJECTION — observer projection interpretation present (Will Dale)
    CI_HIERARCHY — hierarchy of impossibilities present (circle + parabola)
    CI_CHROMO   — chromogeometry connection articulated
    CI_SRC      — source attribution present
    CI_WITNESS  — at least 3 witnesses with correct element computation
    CI_F        — fail_ledger well-formed
"""

QA_COMPLIANCE = "cert_validator — validates C>=4 for all QA states; raw d=b+e for elements; integer state space; no observer, no floats"

import json
import sys
from pathlib import Path

# Canonical element computation lives in qa_alphageometry_ptolemy/qa_elements.py.
# All C/F/G derivation and structural invariant assertions (chromogeometry,
# F identity, d>e, F>0) are enforced there. Cert-local reimplementation is
# forbidden by ELEM-2; use qa_elements.qa_elements() exclusively.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qa_elements import qa_elements

SCHEMA_VERSION = "QA_CIRCLE_IMPOSSIBILITY_CERT.v1"


def verify_all_C_ge_4(m):
    """Exhaustively verify C >= 4 for all states (b,e) in {1..m} x {1..m}.

    Element computation, F identity, d>e, and chromogeometry are asserted
    inside qa_elements(); this function only checks the C >= 4 bound and
    tracks the minimizer.
    """
    min_C = None
    min_states = []
    all_ge_4 = True

    for b in range(1, m + 1):
        for e in range(1, m + 1):
            elem = qa_elements(b, e)
            C = elem.C
            if C < 4:
                all_ge_4 = False
            if min_C is None or C < min_C:
                min_C = C
                min_states = [[b, e]]
            elif C == min_C:
                min_states.append([b, e])

    return all_ge_4, min_C, min_states


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def _run_checks(fixture):
    results = {}

    # CI_1: schema version
    results["CI_1"] = fixture.get("schema_version") == SCHEMA_VERSION

    # CI_C_MIN: minimum C value >= 4
    ci = fixture.get("circle_impossibility", {})
    min_info = ci.get("minimum_C", {})
    claimed_min = min_info.get("min_C_value")
    results["CI_C_MIN"] = (
        claimed_min is not None
        and claimed_min >= 4
    )

    # Computationally verify if modulus is given
    m = min_info.get("modulus")
    if m and results["CI_C_MIN"]:
        actual_ge_4, actual_min, _ = verify_all_C_ge_4(m)
        results["CI_C_MIN"] = (
            actual_ge_4
            and actual_min == claimed_min
            and actual_min >= 4
        )

    # CI_EXHAUSTIVE: all C >= 4
    ev = fixture.get("exhaustive_verification", {})
    results["CI_EXHAUSTIVE"] = ev.get("all_C_ge_4") is True

    # Computationally verify
    ev_m = ev.get("modulus")
    if ev_m and results["CI_EXHAUSTIVE"]:
        actual_ge_4, actual_min, _ = verify_all_C_ge_4(ev_m)
        results["CI_EXHAUSTIVE"] = actual_ge_4 and actual_min >= 4

    # CI_PROJECTION: observer projection interpretation present
    proj = fixture.get("observer_projection_interpretation", {})
    results["CI_PROJECTION"] = (
        "will_dale_insight" in proj
        and len(proj.get("will_dale_insight", "")) > 10
        and "qa_view" in proj
    )

    # CI_HIERARCHY: hierarchy of impossibilities
    hier = fixture.get("hierarchy_of_impossibilities", [])
    results["CI_HIERARCHY"] = (
        isinstance(hier, list)
        and len(hier) >= 2
    )
    if results["CI_HIERARCHY"]:
        shapes = {h.get("shape") for h in hier}
        results["CI_HIERARCHY"] = "circle" in shapes and "parabola" in shapes

    # CI_CHROMO: chromogeometry connection
    chromo = fixture.get("chromogeometry_connection", {})
    results["CI_CHROMO"] = (
        "I_as_metric_difference" in chromo
        and len(chromo.get("I_as_metric_difference", "")) > 10
    )

    # CI_SRC: source attribution
    src = fixture.get("source_attribution", "")
    results["CI_SRC"] = "Will Dale" in src or "Dale" in src

    # CI_WITNESS: at least 3 witnesses with correct element computation
    witnesses = fixture.get("witnesses", [])
    valid_witnesses = [
        w for w in witnesses
        if w.get("C", 0) > 0
        and all(x >= 1 for x in w.get("state_be", [0]))
    ]
    results["CI_WITNESS"] = len(valid_witnesses) >= 3

    # Verify witness values computationally from state_be using raw derivation
    if results["CI_WITNESS"]:
        for w in valid_witnesses:
            be = w.get("state_be", [])
            if len(be) == 2:
                b_val, e_val = be
                d_claimed = w.get("d")
                e_w = w.get("e")
                C_claimed = w.get("C")
                F_claimed = w.get("F")
                # Verify d = b + e (raw, A2)
                d_actual = b_val + e_val
                if d_claimed is not None and d_actual != d_claimed:
                    results["CI_WITNESS"] = False
                    break
                # Verify C and F via canonical element computation
                if C_claimed is not None or F_claimed is not None:
                    elem = qa_elements(b_val, e_val)
                    if C_claimed is not None and elem.C != C_claimed:
                        results["CI_WITNESS"] = False
                        break
                    if F_claimed is not None and elem.F != F_claimed:
                        results["CI_WITNESS"] = False
                        break

    # CI_F: fail_ledger
    fl = fixture.get("fail_ledger")
    results["CI_F"] = isinstance(fl, list)

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
    """Run validator against bundled fixtures."""
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
        print("Usage: python qa_circle_impossibility_cert_validate.py [--self-test | fixture.json]")
        sys.exit(1)
