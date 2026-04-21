#!/usr/bin/env python3
"""
qa_orbit_dirac_bracket_cert_validate.py

Validator for QA_ORBIT_DIRAC_BRACKET_CERT.v1  [family 260]

Backfills the ETCR / Dirac-bracket cross-map thread (commit aa2053c,
2026-04-20) with an explicit QA-native orbit-Dirac-bracket construction on
S_9, per MC-4 of docs/theory/QA_QFT_ETCR_CROSSMAP.md.

Primary-source anchors:
    - Blaschke, D.N., Gieres, F. "On the canonical formulation of gauge
      field theories and Poincare transformations." Nucl. Phys. B 965 (2021).
      arXiv:2004.14406. Equations (5.32)-(5.39) define the Dirac-bracket
      construction from second-class constraints.
    - Mannheim, P.D. "Equivalence of light-front quantization and instant-
      time quantization." Phys. Rev. D 102 025020 (2020). arXiv:1909.03548.
      Motivates the slice/path (observer/invariant) framing.

Construction (m = 9, T_F(b,e) = (a1(b+e), b) with a1(x) = ((x-1) mod m)+1):

  Constraint functions (polynomials in Z[b, e]):
      phi_1(b, e) = b^2 - b*e - e^2 - 1        (Cassini +1 branch of I=1)
      phi_2(b, e) = b - 1                      (gauge-fix to b=1)

  Base bracket (symplectic lift of the tuple wedge; prior-art C3 recast):
      [F, G] := (d_b F)(d_e G) - (d_e F)(d_b G)

  X-matrix entries:
      X_11 = [phi_1, phi_1] = 0
      X_12 = [phi_1, phi_2] = b + 2e
      X_21 = [phi_2, phi_1] = -(b + 2e)
      X_22 = [phi_2, phi_2] = 0

  det X = -(b + 2e)^2. Invertible on the physical subspace iff gcd(b+2e, m)=1.

  Orbit-Dirac bracket:
      [F, G]_orbit := [F, G] - [F, phi_a] (X^-1)^{ab} [phi_b, G]

  Deliverables (MC-4 of cross-map):
      1. Explicit constraint family: phi_1, phi_2 above.
      2. Base bracket choice: symplectic lift of tuple wedge.
      3. X-matrix computed explicitly.
      4. Invertibility check: det X is a unit mod m at every witness.
      5. Instantiation for observable pair F=b, G=e:
             [b, e]_orbit = 0  on physical subspace.
      6. Strong-zero verification:
             [phi_a, F]_orbit = 0  for F in {b, e}, a in {1, 2}.

Scope (v1):
    - m = 9 only. m = 24 deferred to v2 (requires mixed I-level + period-n
      constraint family; see CROSSMAP §4.2 last paragraph).
    - T_F dynamics only. Other QA step operators re-checked in v2 if needed.

QA axiom compliance:
    - A1: witnesses live in {1..m}^2.
    - A2: d = b+e, a = b+2e appear only as derived expressions; never as
          independent variables.
    - T1: path time k is integer; no continuous time appears.
    - T2 (firewall): validator is pure-integer arithmetic (int throughout,
          no floats, no complex, no numpy). Modular inverse computed via
          Python's built-in pow(x, -1, m).
    - S1: no ** on runtime state; polynomial power (b_pow, e_pow) is a
          static exponent tuple, not a computed product.
    - S2: all state is int; no np.zeros, np.random, Fraction, or float.
"""

QA_COMPLIANCE = "cert_validator — QA-native orbit-Dirac-bracket construction; integer state alphabet {1..9}; no floats; symplectic lift of tuple wedge on (b,e); constraint functions are polynomials with integer coefficients; X^{-1} via modular inverse"

import json
import math
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_ORBIT_DIRAC_BRACKET_CERT.v1"


# -----------------------------------------------------------------------------
# Polynomial algebra over Z[b, e]
# Each polynomial is a dict {(b_pow, e_pow): coeff}. Zero-coefficient terms
# are pruned. All coefficients and exponents are Python ints.
# -----------------------------------------------------------------------------

def _poly_prune(p):
    return {k: v for k, v in p.items() if v != 0}


def poly_zero():
    return {}


def poly_const(c):
    if c == 0:
        return {}
    return {(0, 0): int(c)}


def poly_b():
    return {(1, 0): 1}


def poly_e():
    return {(0, 1): 1}


def poly_add(p, q):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, 0) + v
    return _poly_prune(out)


def poly_neg(p):
    return {k: -v for k, v in p.items()}


def poly_sub(p, q):
    return poly_add(p, poly_neg(q))


def poly_mul(p, q):
    out = {}
    for (i1, j1), v1 in p.items():
        for (i2, j2), v2 in q.items():
            key = (i1 + i2, j1 + j2)
            out[key] = out.get(key, 0) + v1 * v2
    return _poly_prune(out)


def poly_deriv_b(p):
    out = {}
    for (i, j), v in p.items():
        if i == 0:
            continue
        out[(i - 1, j)] = v * i
    return _poly_prune(out)


def poly_deriv_e(p):
    out = {}
    for (i, j), v in p.items():
        if j == 0:
            continue
        out[(i, j - 1)] = v * j
    return _poly_prune(out)


def poly_eval(p, b_val, e_val):
    total = 0
    for (i, j), v in p.items():
        term = v
        for _ in range(i):
            term = term * b_val
        for _ in range(j):
            term = term * e_val
        total = total + term
    return total


def poly_bracket(f, g):
    """Symplectic lift of the tuple wedge: [F,G] = (d_b F)(d_e G) - (d_e F)(d_b G).

    Equivalent to prior-art C3's tuple wedge {(b1,e1),(b2,e2)} = b1*e2 - b2*e1
    when F, G are taken as coordinate functions b, e. Generalizes to
    polynomial functions via the standard 2D Poisson structure with {b, e} = 1.
    """
    return poly_sub(
        poly_mul(poly_deriv_b(f), poly_deriv_e(g)),
        poly_mul(poly_deriv_e(f), poly_deriv_b(g)),
    )


def poly_from_terms(terms):
    """Parse a cert-declared constraint polynomial.

    Accepts list of [coeff, b_pow, e_pow] triples. Integer coefficients only.
    """
    out = {}
    for term in terms:
        if not (isinstance(term, list) and len(term) == 3):
            raise ValueError(f"malformed polynomial term {term!r}")
        coeff, b_pow, e_pow = term
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (coeff, b_pow, e_pow)):
            raise ValueError(f"non-integer polynomial term {term!r}")
        if b_pow < 0 or e_pow < 0:
            raise ValueError(f"negative exponent in {term!r}")
        key = (b_pow, e_pow)
        out[key] = out.get(key, 0) + coeff
    return _poly_prune(out)


# -----------------------------------------------------------------------------
# Canonical cert-D constraint family for m = 9 (independent reconstruction)
# -----------------------------------------------------------------------------

def canonical_phi_1():
    """phi_1 = b^2 - b*e - e^2 - 1."""
    b = poly_b()
    e = poly_e()
    bb = poly_mul(b, b)
    be = poly_mul(b, e)
    ee = poly_mul(e, e)
    return poly_sub(poly_sub(poly_sub(bb, be), ee), poly_const(1))


def canonical_phi_2():
    """phi_2 = b - 1."""
    return poly_sub(poly_b(), poly_const(1))


# -----------------------------------------------------------------------------
# Modular inverse wrapper (pure integer, Python 3.8+)
# -----------------------------------------------------------------------------

def mod_inverse(x, m):
    """Return y in {0..m-1} with (x*y) % m == 1, else None if x is not a unit."""
    x = int(x) % m
    if x == 0:
        return None
    if math.gcd(x, m) != 1:
        return None
    return pow(x, -1, m)


def is_unit_mod(x, m):
    x = int(x) % m
    return x != 0 and math.gcd(x, m) == 1


# -----------------------------------------------------------------------------
# Bracket evaluation at a witness
# -----------------------------------------------------------------------------

def _bracket_value(f, g, b_val, e_val, m):
    return poly_eval(poly_bracket(f, g), b_val, e_val) % m


def _orbit_bracket_value(f, g, phis, b_val, e_val, m):
    """Compute [F, G]_orbit at a witness.

    [F, G]_orbit = [F, G] - sum_{a,b} [F, phi_a] (X^-1)^{ab} [phi_b, G]

    Returns None if X is not invertible at the witness (hard fail).
    Otherwise returns the integer value mod m.
    """
    n = len(phis)
    # [F, phi_a] and [phi_b, G] evaluated at witness
    F_phi = [_bracket_value(f, phi_a, b_val, e_val, m) for phi_a in phis]
    phi_G = [_bracket_value(phi_b, g, b_val, e_val, m) for phi_b in phis]

    # X_ab = [phi_a, phi_b] evaluated at witness
    X = [
        [_bracket_value(phis[a], phis[b_], b_val, e_val, m) for b_ in range(n)]
        for a in range(n)
    ]

    # For n=2 antisymmetric X (which is all we need for cert v1):
    if n != 2:
        raise ValueError(f"orbit_bracket_value: cert v1 scoped to n=2, got n={n}")

    g12 = X[0][1] % m
    if g12 == 0 or math.gcd(g12, m) != 1:
        return None  # X not invertible

    # X = [[0, g12], [-g12, 0]], det X = -g12^2
    # X^-1 = [[0, -g12^-1 mod m], [g12^-1 mod m, 0]]
    g12_inv = pow(g12, -1, m)
    # For general 2x2 antisymmetric X, (X^-1)^{ab} follows the same pattern
    X_inv = [
        [0, (-g12_inv) % m],
        [g12_inv % m, 0],
    ]

    base = _bracket_value(f, g, b_val, e_val, m)
    correction = 0
    for a in range(n):
        for b_ in range(n):
            correction = (correction + F_phi[a] * X_inv[a][b_] * phi_G[b_]) % m
    return (base - correction) % m


def _evaluate_X_matrix(phis, b_val, e_val, m):
    n = len(phis)
    return [
        [_bracket_value(phis[a], phis[b_], b_val, e_val, m) for b_ in range(n)]
        for a in range(n)
    ]


def _det_2x2(X, m):
    return (X[0][0] * X[1][1] - X[0][1] * X[1][0]) % m


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # ODB_1 — schema_version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"ODB_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # ODB_F — fail_ledger structural
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("ODB_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("ODB_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    m = cert.get("modulus")
    if m != 9:
        errors.append(f"ODB_1: modulus must be 9 in v1, got {m!r}")
        return errors, warnings

    # ODB_PHI — constraint polynomials declared and parseable
    decl_phis_raw = cert.get("constraint_polynomials")
    if (not isinstance(decl_phis_raw, list)) or len(decl_phis_raw) != 2:
        errors.append("ODB_PHI: constraint_polynomials must be a list of exactly 2 entries")
        return errors, warnings

    try:
        decl_phi_1 = poly_from_terms(decl_phis_raw[0].get("terms", []))
        decl_phi_2 = poly_from_terms(decl_phis_raw[1].get("terms", []))
    except Exception as ex:
        errors.append(f"ODB_PHI: failed to parse constraint polynomials: {ex}")
        return errors, warnings

    canon_phi_1 = canonical_phi_1()
    canon_phi_2 = canonical_phi_2()

    # Canonical match required in v1 — cert-D's form is fixed for this release.
    if decl_phi_1 != canon_phi_1:
        errors.append(
            "ODB_PHI: declared phi_1 does not match canonical b^2 - b*e - e^2 - 1. "
            f"got coeffs {sorted(decl_phi_1.items())!r}"
        )
    if decl_phi_2 != canon_phi_2:
        errors.append(
            "ODB_PHI: declared phi_2 does not match canonical b - 1. "
            f"got coeffs {sorted(decl_phi_2.items())!r}"
        )

    if errors:
        return errors, warnings

    phis = [canon_phi_1, canon_phi_2]

    # ODB_WIT_A1 — witnesses in {1..m}^2
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 2:
        errors.append(f"ODB_WIT_A1: witnesses must be a list of at least 2 entries, got {len(witnesses) if isinstance(witnesses, list) else 'none'}")
        return errors, warnings

    normalized_witnesses = []
    for idx, w in enumerate(witnesses):
        pt = w.get("point") if isinstance(w, dict) else None
        if not (isinstance(pt, list) and len(pt) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in pt)):
            errors.append(f"ODB_WIT_A1: witness[{idx}] .point malformed (expected [int, int]), got {pt!r}")
            continue
        b_val, e_val = pt
        if not (1 <= b_val <= m and 1 <= e_val <= m):
            errors.append(f"ODB_WIT_A1: witness[{idx}] .point [{b_val},{e_val}] outside {{1..{m}}}^2 (A1 violation)")
            continue
        normalized_witnesses.append((idx, b_val, e_val, w))
    if errors:
        return errors, warnings

    # ODB_WIT_PHYSICAL — witnesses satisfy phi_a(w) == 0 mod m for all a
    for idx, b_val, e_val, _ in normalized_witnesses:
        for a, phi in enumerate(phis, start=1):
            val = poly_eval(phi, b_val, e_val) % m
            if val != 0:
                errors.append(
                    f"ODB_WIT_PHYSICAL: witness[{idx}] ({b_val},{e_val}) "
                    f"fails phi_{a} = 0 mod {m} (got {val})"
                )

    # ODB_X_MATRIX — X_ab recomputed symbolically, evaluated at each witness,
    # must match declared per-witness X_matrix
    for idx, b_val, e_val, w in normalized_witnesses:
        X_recomputed = _evaluate_X_matrix(phis, b_val, e_val, m)
        decl_X = w.get("X_matrix_mod_m")
        if not (isinstance(decl_X, list) and len(decl_X) == 2
                and all(isinstance(row, list) and len(row) == 2 for row in decl_X)):
            errors.append(f"ODB_X_MATRIX: witness[{idx}] X_matrix_mod_m must be a 2x2 integer list")
            continue
        decl_X_norm = [[int(v) % m for v in row] for row in decl_X]
        if X_recomputed != decl_X_norm:
            errors.append(
                f"ODB_X_MATRIX: witness[{idx}] declared X {decl_X_norm!r} "
                f"does not match recomputed {X_recomputed!r}"
            )

    # ODB_INV — det X invertible mod m (coprime to m)
    for idx, b_val, e_val, w in normalized_witnesses:
        X = _evaluate_X_matrix(phis, b_val, e_val, m)
        det = _det_2x2(X, m)
        if not is_unit_mod(det, m):
            errors.append(
                f"ODB_INV: witness[{idx}] ({b_val},{e_val}) det X = {det} mod {m} "
                f"is not a unit — X^-1 does not exist on physical subspace"
            )
            continue
        decl_det = w.get("det_X_mod_m")
        if decl_det is not None and (int(decl_det) % m) != det:
            errors.append(
                f"ODB_INV: witness[{idx}] declared det_X_mod_m = {decl_det} "
                f"does not match recomputed {det} mod {m}"
            )

    # ODB_DB_BE_ZERO — [b, e]_orbit == 0 mod m at every witness
    b_poly = poly_b()
    e_poly = poly_e()
    for idx, b_val, e_val, w in normalized_witnesses:
        val = _orbit_bracket_value(b_poly, e_poly, phis, b_val, e_val, m)
        if val is None:
            errors.append(
                f"ODB_DB_BE_ZERO: witness[{idx}] orbit bracket undefined "
                f"(X singular at this witness)"
            )
            continue
        if val != 0:
            errors.append(
                f"ODB_DB_BE_ZERO: [b, e]_orbit = {val} mod {m} at witness[{idx}] "
                f"({b_val},{e_val}); expected 0"
            )
        decl = w.get("bracket_b_e_orbit_mod_m")
        if decl is not None and (int(decl) % m) != val:
            errors.append(
                f"ODB_DB_BE_ZERO: witness[{idx}] declared bracket_b_e_orbit_mod_m "
                f"= {decl} does not match recomputed {val}"
            )

    # ODB_STRONG_ZERO — [phi_a, F]_orbit == 0 mod m for F in {b, e}, a in {1, 2}
    for idx, b_val, e_val, _ in normalized_witnesses:
        for a, phi in enumerate(phis, start=1):
            for F_name, F_poly in (("b", b_poly), ("e", e_poly)):
                val = _orbit_bracket_value(phi, F_poly, phis, b_val, e_val, m)
                if val is None:
                    errors.append(
                        f"ODB_STRONG_ZERO: witness[{idx}] [phi_{a}, {F_name}]_orbit "
                        f"undefined (X singular)"
                    )
                    continue
                if val != 0:
                    errors.append(
                        f"ODB_STRONG_ZERO: [phi_{a}, {F_name}]_orbit = {val} mod {m} "
                        f"at witness[{idx}] ({b_val},{e_val}); expected 0"
                    )

    # ODB_SRC — source_attribution references primary sources + cross-map
    src = str(cert.get("source_attribution", ""))
    for needle in ("2004.14406", "1909.03548", "Blaschke", "Mannheim",
                   "QA_QFT_ETCR_CROSSMAP", "[191]"):
        if needle not in src:
            warnings.append(f"ODB_SRC: source_attribution should reference {needle!r}")

    # ODB_WITNESS — at least 2 witnesses on distinct points
    pts = set()
    for _, b_val, e_val, _ in normalized_witnesses:
        pts.add((b_val, e_val))
    if len(pts) < 2:
        errors.append("ODB_WITNESS: require at least 2 distinct physical-subspace points")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("odb_pass_m9_b1_i1.json", True),
        ("odb_fail_wrong_physical_subspace.json", False),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QA Orbit-Dirac Bracket Cert [260] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
