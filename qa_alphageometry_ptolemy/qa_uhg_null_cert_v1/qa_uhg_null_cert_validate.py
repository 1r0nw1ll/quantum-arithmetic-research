#!/usr/bin/env python3
"""QA UHG Null Cert family [127] validator — QA_UHG_NULL_CERT.v1

Certifies that every QA triple (F, C, G) generated from (d, e) is a NULL POINT
in Universal Hyperbolic Geometry (UHG).

The UHG null condition for a projective point [F:C:G] is:

    F² + C² − G² = 0

This is identical to the QA Chromogeometric identity C² + F² = G² (family [125]).
The triple (F, C, G) = (d²−e², 2de, d²+e²) is always a null point because:

    (d²−e²)² + (2de)² = d⁴−2d²e²+e⁴ + 4d²e² = d⁴+2d²e²+e⁴ = (d²+e²)²

This is a Pythagorean triple identity, and in UHG terminology it means the
direction vector (d, e) maps to a LIGHT-LIKE (null) vector in Minkowski 3-space.

Gaussian integer interpretation:
    Z = d + e·i  (Gaussian integer)
    Z² = (d²−e²) + 2de·i
    |Z|² = d²+e²

So (F, C, G) = (Re(Z²), Im(Z²), |Z|²). The join of null points in UHG corresponds
to Gaussian integer multiplication.

Additional certification:
    The null quadrance: q([F:C:G]) = F²+C²−G² = 0  (null condition)
    The projective representation: [F:C:G] ≠ [0:0:0] (non-degenerate)

Validation checks:
  UN1  schema_version == 'QA_UHG_NULL_CERT.v1'           → SCHEMA_VERSION_WRONG
  UN2  C == 2·d·e  (Green quadrance)                     → GREEN_QUADRANCE_MISMATCH
  UN3  F == d²−e²  (Red quadrance)                       → RED_QUADRANCE_MISMATCH
  UN4  G == d²+e²  (Blue quadrance)                      → BLUE_QUADRANCE_MISMATCH
  UN5  F²+C²−G² == 0  (UHG null condition)               → NULL_CONDITION_VIOLATED
  UN6  Z² decomposition: F==Re(Z²) and C==Im(Z²) and     → GAUSSIAN_DECOMP_MISMATCH
       G==|Z|² where Z = d+e·i
  UN7  null_quadrance field == 0                          → NULL_QUADRANCE_WRONG

ok semantics (same as families [125], [126]):
  ok=True  — cert is internally consistent: detected failures == declared fail_ledger
             AND result field matches
  ok=False — inconsistency between detected and declared, or result field wrong

Usage:
  python qa_uhg_null_cert_validate.py --self-test
  python qa_uhg_null_cert_validate.py --file fixtures/un_pass_3_4_5.json
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
    "NULL_CONDITION_VIOLATED",
    "GAUSSIAN_DECOMP_MISMATCH",
    "NULL_QUADRANCE_WRONG",
])


class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def validate_uhg_null_cert(cert: dict) -> dict:
    out = _Out()
    detected: set = set()

    # ── UN1: schema_version ──────────────────────────────────────────────────
    if cert.get("schema_version") != "QA_UHG_NULL_CERT.v1":
        detected.add("SCHEMA_VERSION_WRONG")
        out.fail(
            f"UN1 SCHEMA_VERSION_WRONG: must be 'QA_UHG_NULL_CERT.v1', "
            f"got {cert.get('schema_version')!r}"
        )

    # ── required fields ──────────────────────────────────────────────────────
    for field in ["certificate_id", "cert_type", "title", "created_utc",
                  "generator", "uhg_null", "validation_checks",
                  "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    # ── extract generator ────────────────────────────────────────────────────
    gen = cert.get("generator", {})
    d = gen.get("d")
    e = gen.get("e")

    if not all(isinstance(v, int) for v in [d, e]):
        out.fail("generator fields d, e must both be integers")
        return _reconcile(cert, out, detected)

    # ── extract claimed UHG null fields ──────────────────────────────────────
    un = cert.get("uhg_null", {})
    C_claimed = un.get("C")
    F_claimed = un.get("F")
    G_claimed = un.get("G")
    nq_claimed = un.get("null_quadrance")

    if not all(isinstance(v, int) for v in [C_claimed, F_claimed, G_claimed]):
        out.fail("uhg_null fields C, F, G must all be integers")
        return _reconcile(cert, out, detected)

    # ── UN2: C == 2·d·e  (Green quadrance) ───────────────────────────────────
    C_actual = 2 * d * e
    if C_claimed != C_actual:
        detected.add("GREEN_QUADRANCE_MISMATCH")
        out.fail(
            f"UN2 GREEN_QUADRANCE_MISMATCH: claimed C={C_claimed}, "
            f"actual 2·{d}·{e}={C_actual}"
        )

    # ── UN3: F == d²−e²  (Red quadrance) ─────────────────────────────────────
    F_actual = d * d - e * e
    if F_claimed != F_actual:
        detected.add("RED_QUADRANCE_MISMATCH")
        out.fail(
            f"UN3 RED_QUADRANCE_MISMATCH: claimed F={F_claimed}, "
            f"actual d²−e²={d*d}−{e*e}={F_actual}"
        )

    # ── UN4: G == d²+e²  (Blue quadrance) ────────────────────────────────────
    G_actual = d * d + e * e
    if G_claimed != G_actual:
        detected.add("BLUE_QUADRANCE_MISMATCH")
        out.fail(
            f"UN4 BLUE_QUADRANCE_MISMATCH: claimed G={G_claimed}, "
            f"actual d²+e²={d*d}+{e*e}={G_actual}"
        )

    # ── UN5: UHG null condition F²+C²−G²=0 ───────────────────────────────────
    null_check = (F_claimed * F_claimed
                  + C_claimed * C_claimed
                  - G_claimed * G_claimed)
    if null_check != 0:
        detected.add("NULL_CONDITION_VIOLATED")
        out.fail(
            f"UN5 NULL_CONDITION_VIOLATED: F²+C²−G² = "
            f"{F_claimed}²+{C_claimed}²−{G_claimed}² = {null_check} ≠ 0"
        )

    # ── UN6: Gaussian integer decomposition Z=d+ei, Z²=(d²−e²)+2de·i ─────────
    # Z² = d²−e² + 2de·i, |Z|² = d²+e²
    re_z2 = d * d - e * e   # Re(Z²) = F_actual
    im_z2 = 2 * d * e       # Im(Z²) = C_actual
    abs2_z = d * d + e * e  # |Z|² = G_actual

    gauss_issues = []
    if F_claimed != re_z2:
        gauss_issues.append(f"F={F_claimed} ≠ Re(Z²)={re_z2}")
    if C_claimed != im_z2:
        gauss_issues.append(f"C={C_claimed} ≠ Im(Z²)={im_z2}")
    if G_claimed != abs2_z:
        gauss_issues.append(f"G={G_claimed} ≠ |Z|²={abs2_z}")

    if gauss_issues:
        detected.add("GAUSSIAN_DECOMP_MISMATCH")
        out.fail(
            "UN6 GAUSSIAN_DECOMP_MISMATCH (Z=d+e·i, Z²=(d²−e²)+2de·i): "
            + "; ".join(gauss_issues)
        )

    # ── UN7: null_quadrance field == 0 ───────────────────────────────────────
    if nq_claimed is not None:
        if not isinstance(nq_claimed, int):
            out.fail("UN7: null_quadrance must be an integer if present")
        elif nq_claimed != 0:
            detected.add("NULL_QUADRANCE_WRONG")
            out.fail(
                f"UN7 NULL_QUADRANCE_WRONG: claimed null_quadrance={nq_claimed}, "
                f"must be 0 for a null point"
            )

    return _reconcile(cert, out, detected)


def _reconcile(cert: dict, out: _Out, detected: set) -> dict:
    """Produce final validation result. ok=True iff detected==declared AND result consistent."""
    declared_fail_types = set(cert.get("fail_ledger", []))
    declared_result = cert.get("result", "")

    consistency_errors = []

    for f in sorted(declared_fail_types):
        if f not in KNOWN_FAIL_TYPES:
            consistency_errors.append(f"fail_ledger contains unknown fail type: {f!r}")

    undeclared = detected - declared_fail_types
    for f in sorted(undeclared):
        consistency_errors.append(f"detected fail {f!r} not declared in fail_ledger")

    overclaimed = declared_fail_types - detected
    for f in sorted(overclaimed):
        consistency_errors.append(f"fail_ledger claims {f!r} but validator did not detect it")

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
    return {
        "ok": ok,
        "label": "PASS" if ok else "FAIL",
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "detected_fails": sorted(detected),
        "detection_messages": out.errors,
        "errors": consistency_errors,
        "warnings": out.warnings,
    }


def _self_test() -> dict:
    fixtures_dir = HERE / "fixtures"
    expected = {
        "un_pass_3_4_5.json":               True,
        "un_pass_5_12_13.json":             True,
        "un_fail_null_violated.json":       True,
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
        r = validate_uhg_null_cert(cert)
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
        description="QA UHG Null Cert [127] validator"
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
        res = validate_uhg_null_cert(cert)
        print(json.dumps(res, indent=2, sort_keys=True))
        sys.exit(0 if res["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
