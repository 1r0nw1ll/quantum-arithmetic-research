#!/usr/bin/env python3
"""QA Chromogeometry Cert family [125] validator — QA_CHROMOGEOMETRY_CERT.v1

Certifies that the three QA invariants C, F, G of any generator (b,e) with
d=b+e, a=b+2e are exactly the three chromogeometric quadrances of the
direction vector (d,e):

    Green quadrance:  Q_g(d,e) = 2·d·e  = C
    Red quadrance:    Q_r(d,e) = d²−e²  = F  (= a·b, the semi-latus product)
    Blue quadrance:   Q_b(d,e) = d²+e²  = G  (= hypotenuse²)

The identity C²+F²=G² is Wildberger Chromogeometric Theorem 6:
    Q_b(d,e)² = Q_r(d,e)² + Q_g(d,e)²

Equivalently: (d²+e²)² = (d²−e²)² + (2de)²  — a Pythagorean triple identity.

Conic discriminant: I = C − F = 2e² − b²
    I < 0  →  ellipse
    I = 0  →  parabola  (requires b = e√2, no integer solution except b=e=0)
    I > 0  →  hyperbola

Validation checks:
  CG1  schema_version == 'QA_CHROMOGEOMETRY_CERT.v1'      → SCHEMA_VERSION_WRONG
  CG2  C == 2·d·e  (Green quadrance identity)             → GREEN_QUADRANCE_MISMATCH
  CG3  F == d·d − e·e  (Red quadrance identity)           → RED_QUADRANCE_MISMATCH
  CG4  G == d·d + e·e  (Blue quadrance identity)          → BLUE_QUADRANCE_MISMATCH
  CG5  C·C + F·F == G·G  (Chromogeometric Pythagoras)     → PYTHAGORAS_VIOLATED
  CG6  F == b·a  (semi-latus product identity)            → SEMI_LATUS_MISMATCH
  CG7  conic_type and I match sign of 2e²−b²              → CONIC_TYPE_MISMATCH

ok semantics (same as family [111] pk cert):
  ok=True  — cert is internally consistent: detected failures == declared fail_ledger
             AND result field matches (PASS↔empty ledger, FAIL↔nonempty ledger)
  ok=False — inconsistency between detected and declared, or result field wrong

Usage:
  python qa_chromogeometry_cert_validate.py --self-test
  python qa_chromogeometry_cert_validate.py --file fixtures/cg_pass_hyperbola_3_4_5.json
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

KNOWN_FAIL_TYPES = frozenset([
    "SCHEMA_VERSION_WRONG",
    "GREEN_QUADRANCE_MISMATCH",
    "RED_QUADRANCE_MISMATCH",
    "BLUE_QUADRANCE_MISMATCH",
    "PYTHAGORAS_VIOLATED",
    "SEMI_LATUS_MISMATCH",
    "CONIC_TYPE_MISMATCH",
])

VALID_CONIC_TYPES = frozenset(["ellipse", "parabola", "hyperbola"])


class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def _conic_from_C_F(C: int, F: int) -> tuple:
    """Compute conic type and I = |C − F| from chromogeometric quadrances.

    I is the positive difference of C and F (always ≥ 0).
    Conic type determined by which quadrance dominates:
        C > F  →  hyperbola   (C quadrance exceeds F)
        C = F  →  parabola    (equal — exact integer rarity)
        C < F  →  ellipse     (F quadrance exceeds C)
    """
    I = abs(C - F)
    if C > F:
        conic_type = "hyperbola"
    elif C < F:
        conic_type = "ellipse"
    else:
        conic_type = "parabola"
    return I, conic_type


def validate_chromogeometry_cert(cert: dict) -> dict:
    out = _Out()
    detected: set = set()

    # ── CG1: schema_version ─────────────────────────────────────────────────
    if cert.get("schema_version") != "QA_CHROMOGEOMETRY_CERT.v1":
        detected.add("SCHEMA_VERSION_WRONG")
        out.fail(
            f"CG1 SCHEMA_VERSION_WRONG: must be 'QA_CHROMOGEOMETRY_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )

    # ── required fields ──────────────────────────────────────────────────────
    for field in ["certificate_id", "cert_type", "title", "created_utc",
                  "generator", "chromogeometry", "conic_classification",
                  "validation_checks", "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    # ── extract generator ────────────────────────────────────────────────────
    gen = cert.get("generator", {})
    b = gen.get("b")
    e = gen.get("e")
    d = gen.get("d")
    a = gen.get("a")

    if not all(isinstance(v, int) for v in [b, e, d, a]):
        out.fail("generator fields b, e, d, a must all be integers")
        # Cannot continue with arithmetic checks
        return _reconcile(cert, out, detected)

    if d != b + e:
        out.fail(f"generator.d must equal b+e={b+e}, got {d}")
    if a != b + 2 * e:
        out.fail(f"generator.a must equal b+2e={b + 2*e}, got {a}")

    # ── extract claimed chromogeometry ───────────────────────────────────────
    cg = cert.get("chromogeometry", {})
    C_claimed = cg.get("C")
    F_claimed = cg.get("F")
    G_claimed = cg.get("G")

    if not all(isinstance(v, int) for v in [C_claimed, F_claimed, G_claimed]):
        out.fail("chromogeometry fields C, F, G must all be integers")
        return _reconcile(cert, out, detected)

    # ── CG2: Green quadrance C = 2·d·e ──────────────────────────────────────
    C_actual = 2 * d * e
    if C_claimed != C_actual:
        detected.add("GREEN_QUADRANCE_MISMATCH")
        out.fail(
            f"CG2 GREEN_QUADRANCE_MISMATCH: claimed C={C_claimed}, "
            f"actual Q_green(d={d},e={e})=2·{d}·{e}={C_actual}"
        )

    # ── CG3: Red quadrance F = d²−e² ────────────────────────────────────────
    F_actual = d * d - e * e
    if F_claimed != F_actual:
        detected.add("RED_QUADRANCE_MISMATCH")
        out.fail(
            f"CG3 RED_QUADRANCE_MISMATCH: claimed F={F_claimed}, "
            f"actual Q_red(d={d},e={e})=d²−e²={d*d}−{e*e}={F_actual}"
        )

    # ── CG4: Blue quadrance G = d²+e² ───────────────────────────────────────
    G_actual = d * d + e * e
    if G_claimed != G_actual:
        detected.add("BLUE_QUADRANCE_MISMATCH")
        out.fail(
            f"CG4 BLUE_QUADRANCE_MISMATCH: claimed G={G_claimed}, "
            f"actual Q_blue(d={d},e={e})=d²+e²={d*d}+{e*e}={G_actual}"
        )

    # ── CG5: Chromogeometric Pythagoras C²+F²=G² ────────────────────────────
    if C_claimed * C_claimed + F_claimed * F_claimed != G_claimed * G_claimed:
        detected.add("PYTHAGORAS_VIOLATED")
        out.fail(
            f"CG5 PYTHAGORAS_VIOLATED: C²+F²≠G²: "
            f"{C_claimed}²+{F_claimed}²={C_claimed*C_claimed + F_claimed*F_claimed} "
            f"≠ {G_claimed}²={G_claimed*G_claimed}"
        )

    # ── CG6: Semi-latus product F = b·a ─────────────────────────────────────
    F_semi_latus = b * a
    if F_claimed != F_semi_latus:
        detected.add("SEMI_LATUS_MISMATCH")
        out.fail(
            f"CG6 SEMI_LATUS_MISMATCH: claimed F={F_claimed}, "
            f"b·a={b}·{a}={F_semi_latus}. "
            f"Note: d²−e²=(d−e)(d+e)=b·a by construction."
        )

    # ── CG7: Conic classification ────────────────────────────────────────────
    # I = |C − F| (positive difference); conic type from which dominates
    conic = cert.get("conic_classification", {})
    claimed_type = conic.get("conic_type", "")
    claimed_I = conic.get("I", None)

    I_actual, expected_type = _conic_from_C_F(C_claimed, F_claimed)

    cg7_issues = []
    if claimed_I is not None and claimed_I != I_actual:
        cg7_issues.append(
            f"claimed I={claimed_I} but |C−F|=|{C_claimed}−{F_claimed}|={I_actual}"
        )
    if claimed_type not in VALID_CONIC_TYPES:
        cg7_issues.append(
            f"conic_type must be one of {sorted(VALID_CONIC_TYPES)}, got {claimed_type!r}"
        )
    elif claimed_type != expected_type:
        cg7_issues.append(
            f"claimed '{claimed_type}' but C={'>' if C_claimed > F_claimed else '<' if C_claimed < F_claimed else '='}F → '{expected_type}'"
        )

    if cg7_issues:
        detected.add("CONIC_TYPE_MISMATCH")
        out.fail("CG7 CONIC_TYPE_MISMATCH: " + "; ".join(cg7_issues))

    return _reconcile(cert, out, detected)


def _reconcile(cert: dict, out: _Out, detected: set) -> dict:
    """
    Produce the final validation result.
    ok=True when:
      - declared fail_ledger == detected failures
      - result field is consistent ('PASS' iff ledger empty, 'FAIL' iff nonempty)
    out.errors contains detection messages (informational); consistency errors are
    computed here separately.
    """
    declared_fail_types = set(cert.get("fail_ledger", []))
    declared_result = cert.get("result", "")

    # Consistency issues (these make ok=False)
    consistency_errors = []

    # Validate fail_ledger entries are known types
    for f in sorted(declared_fail_types):
        if f not in KNOWN_FAIL_TYPES:
            consistency_errors.append(f"fail_ledger contains unknown fail type: {f!r}")

    # Cross-check detected vs declared
    undeclared = detected - declared_fail_types
    for f in sorted(undeclared):
        consistency_errors.append(f"detected fail {f!r} not declared in fail_ledger")

    overclaimed = declared_fail_types - detected
    for f in sorted(overclaimed):
        consistency_errors.append(f"fail_ledger claims {f!r} but validator did not detect it")

    # result field consistency
    if declared_result == "PASS":
        if detected or declared_fail_types:
            consistency_errors.append(
                f"result='PASS' but failures present: detected={sorted(detected)}, "
                f"ledger={sorted(declared_fail_types)}"
            )
    elif declared_result == "FAIL":
        if not declared_fail_types:
            consistency_errors.append("result='FAIL' but fail_ledger is empty")
        elif detected != declared_fail_types:
            consistency_errors.append(
                f"declared ledger {sorted(declared_fail_types)} != detected {sorted(detected)}"
            )
    else:
        consistency_errors.append(
            f"result must be 'PASS' or 'FAIL', got {declared_result!r}"
        )

    ok = len(consistency_errors) == 0
    label = "PASS" if ok else "FAIL"

    return {
        "ok": ok,
        "label": label,
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "detected_fails": sorted(detected),
        "detection_messages": out.errors,   # informational: what was found
        "errors": consistency_errors,        # blocking: consistency failures
        "warnings": out.warnings,
    }


def _self_test() -> dict:
    fixtures_dir = HERE / "fixtures"
    expected = {
        "cg_pass_hyperbola_3_4_5.json":  True,
        "cg_pass_ellipse_20_21_29.json":  True,
        "cg_fail_green_mismatch.json":    True,  # FAIL fixture should be internally consistent
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        with open(fpath) as f:
            cert = json.load(f)
        r = validate_chromogeometry_cert(cert)
        passed = r["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": r["label"],
            "detected_fails": r["detected_fails"],
            "errors": r["errors"],
        })

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Chromogeometry Cert [125] validator"
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Run all fixtures and print JSON summary")
    parser.add_argument("--file", type=str,
                        help="Validate a single fixture file")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        with open(args.file) as f:
            cert = json.load(f)
        res = validate_chromogeometry_cert(cert)
        print(json.dumps(res, indent=2, sort_keys=True))
        sys.exit(0 if res["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
