#!/usr/bin/env python3
"""qa_whittaker_rational_direction_s1_cert_validate.py — cert family [266].
# RT1_OBSERVER_FILE: Whittaker rational direction — classical trig as verification for RT claims

Primary source: E. T. Whittaker (1903), "On the partial differential
equations of mathematical physics," Math. Annalen 57:333-355,
DOI 10.1007/BF01444290.

Validator for QA_WHITTAKER_RATIONAL_DIRECTION_S1_CERT.v1.

Layer 1 of the Whittaker -> QA development ladder
(docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md sec. 8).

CLAIM (narrow): the QA-rational direction set D_m on the unit circle S^1,

    D_m := { (C/G, F/G) : (b,e) in {1..m}^2, gcd(b,e)=1,
                          d = b+e, a = b+2e,
                          C = 2*d*e, F = a*b = d^2 - e^2, G = d^2 + e^2 }

is finite, exactly enumerable in integer arithmetic, and admits two structural
theorems on the closed first quadrant of S^1:

  (W2) Angular separation lower bound.
       For any two distinct directions in D_m,
           sin(angular_separation) >= 1 / (G_i * G_j) >= 1 / G_max(m)^2.
       Proven from cross-product integrality (the integer
       |C_i * F_j - C_j * F_i| is non-negative, vanishes iff vectors are
       parallel/equal in Q1, hence is at least 1 for distinct directions).
       Validator checks all i<j pairs.

  (W3) Lipschitz nearest-neighbor sampling error bound on the closed quadrant.
       D_m^+ := D_m union {E_0=(1,0), E_inf=(0,1)}.
       The boundary anchors E_0, E_inf are observer-side anchors only:
       they are NOT QA seeds, NOT counted in |D_m|, NOT in W1, NOT in W2.
       Define Delta_max^+(m) as the maximum angular gap between consecutive
       points of D_m^+ over [0, pi/2]. Then for any L-Lipschitz angular
       profile g on [0, pi/2], nearest-neighbor sampling at D_m^+ yields:
           sup_{theta in [0, pi/2]} |g(theta) - g(theta_NN(theta))|
                                                 <= L * Delta_max^+(m).

This cert does NOT prove Whittaker's wave-equation theorem, Maxwell's
equations, electromagnetism, two-scalar-potential reductions, or any
physics. Whittaker 1903 is the motivation for the rational angular net;
the cert certifies only the QA-side discretization layer.

Theorem NT compliance: integer / fractions.Fraction throughout
construction. Float operations are confined to the observer-side W3
test grid, the test function g, and pi/2 boundary float; declared
explicitly as observer projection and crossed exactly twice (input/output).

Checks:
    WRD_1     - schema_version matches
    WRD_DECL  - required fields present and well-typed; m in {9, 24, 72};
                claim block present
    WRD_W1    - bit-exact cardinality |D_m| == declared_count
    WRD_W2    - all pairs (i<j) satisfy |C_i*F_j - C_j*F_i| >= 1
    WRD_W3    - Lipschitz nearest-neighbor sup error <= L * Delta_max^+(m)
    WRD_HARD  - per-seed HARD rule: d == b+e, a == b+2e (raw, no mod);
                seeds declared as gcd-coprime are gcd-coprime in fact
    WRD_SRC   - source_attribution cites Whittaker 1903 + DOI
    WRD_WIT   - witnesses present (>= 1 for PASS fixtures with claims)
    WRD_F     - fail_ledger well-formed (where applicable)
"""

QA_COMPLIANCE = "cert_validator - QA-rational direction set on S^1; integer + fractions.Fraction construction; raw d=b+e, a=b+2e; gcd-coprime seeds; angular ordering via Fraction(F,C); floats observer-side only (Lipschitz test grid, sup-error display, pi/2 boundary)"

import json
import math
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_WHITTAKER_RATIONAL_DIRECTION_S1_CERT.v1"

ALLOWED_M = {9, 24, 72}


# -----------------------------------------------------------------------------
# QA-discrete construction (integer / Fraction only)
# -----------------------------------------------------------------------------

def enumerate_D_m(m: int) -> list[tuple[int, int, int, int, int]]:
    """Return all coprime QA seeds in {1..m}^2 as (b, e, C, F, G), ordered by angle.

    Per cert spec: gcd(b,e) = 1; d = b+e raw; a = b+2e raw;
    C = 2*d*e; F = a*b; G = d*d + e*e. C^2 + F^2 = G^2 by construction.
    """
    if not isinstance(m, int) or m < 1:
        raise ValueError(f"m must be positive int; got {m!r}")

    pts: list[tuple[int, int, int, int, int]] = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = b + 2 * e
            C = 2 * d * e
            F = a * b
            G = d * d + e * e
            assert d == b + e and a == b + 2 * e
            assert F == d * d - e * e
            assert C * C + F * F == G * G
            pts.append((b, e, C, F, G))

    pts.sort(key=lambda p: Fraction(p[3], p[2]))
    return pts


def compute_g_max(pts: list[tuple[int, int, int, int, int]]) -> int:
    return max(p[4] for p in pts)


def compute_g_min(pts: list[tuple[int, int, int, int, int]]) -> int:
    return min(p[4] for p in pts)


def consecutive_sin_gaps(pts: list[tuple[int, int, int, int, int]]) -> list[Fraction]:
    """sin(Δθ) between consecutive directions, exact rational."""
    gaps: list[Fraction] = []
    for i in range(len(pts) - 1):
        _, _, C1, F1, G1 = pts[i]
        _, _, C2, F2, G2 = pts[i + 1]
        cross = abs(C1 * F2 - C2 * F1)
        gaps.append(Fraction(cross, G1 * G2))
    return gaps


# -----------------------------------------------------------------------------
# Closed first-quadrant max gap (D_m^+ with virtual boundary anchors)
# -----------------------------------------------------------------------------

def delta_max_plus(pts: list[tuple[int, int, int, int, int]]) -> float:
    """Δ_max^+(m): max angular gap on closed Q1 [0, pi/2] of D_m^+.

    Boundary gaps via float (atan2 + pi/2). Interior gaps from exact rational
    sin(Δθ) converted via asin once for reporting. Floats observer-side only.
    """
    if not pts:
        return math.pi / 2.0

    angle0 = math.atan2(pts[0][3], pts[0][2])
    angle_last = math.atan2(pts[-1][3], pts[-1][2])

    boundary_left = angle0 - 0.0
    boundary_right = math.pi / 2.0 - angle_last

    sin_gaps = consecutive_sin_gaps(pts)
    interior_floats = [math.asin(float(g)) for g in sin_gaps]

    return max([boundary_left, boundary_right] + interior_floats)


def sample_angles_plus(pts: list[tuple[int, int, int, int, int]]) -> list[float]:
    """Angles of D_m^+ in [0, pi/2], sorted ascending. Includes 0 and pi/2."""
    if not pts:
        return [0.0, math.pi / 2.0]
    interior = [math.atan2(p[3], p[2]) for p in pts]
    return [0.0] + interior + [math.pi / 2.0]


# -----------------------------------------------------------------------------
# Observer-side W3: Lipschitz nearest-neighbor sampling error
# -----------------------------------------------------------------------------

_TEST_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "identity": lambda x: x,
}


def w3_sup_error(pts: list[tuple[int, int, int, int, int]],
                 function_name: str,
                 test_grid_size: int) -> float:
    """sup_{theta in [0, pi/2]} |g(theta) - g(NN(theta))|, observer-side."""
    if function_name not in _TEST_FUNCTIONS:
        raise ValueError(f"unknown test function {function_name!r}; "
                         f"allowed: {sorted(_TEST_FUNCTIONS)}")
    g = _TEST_FUNCTIONS[function_name]
    sample_angles = sample_angles_plus(pts)

    if test_grid_size < 2:
        test_grid_size = 2
    test_grid = [
        (math.pi / 2.0) * i / (test_grid_size - 1)
        for i in range(test_grid_size)
    ]

    def nearest(theta: float) -> float:
        best = sample_angles[0]
        best_diff = abs(theta - best)
        for s in sample_angles[1:]:
            diff = abs(theta - s)
            if diff < best_diff:
                best = s
                best_diff = diff
        return best

    sup = 0.0
    for theta in test_grid:
        nn = nearest(theta)
        err = abs(g(theta) - g(nn))
        if err > sup:
            sup = err
    return sup


# -----------------------------------------------------------------------------
# Validation gates
# -----------------------------------------------------------------------------

def _err(reasons: list[str], code: str, msg: str) -> None:
    reasons.append(f"{code}: {msg}")


def validate(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as ex:
        return [f"WRD_1: failed to read/parse {path}: {ex}"], warnings

    sv = data.get("schema_version")
    if sv != SCHEMA_VERSION:
        _err(errors, "WRD_1",
             f"expected schema_version={SCHEMA_VERSION!r}, got {sv!r}")
        return errors, warnings

    m = data.get("m")
    if not isinstance(m, int) or m not in ALLOWED_M:
        _err(errors, "WRD_DECL",
             f"m must be int in {sorted(ALLOWED_M)}; got {m!r}")
        return errors, warnings

    src = data.get("source_attribution", "")
    if not (isinstance(src, str)
            and "Whittaker" in src and "1903" in src and "10.1007/BF01444290" in src):
        _err(errors, "WRD_SRC",
             "source_attribution must mention Whittaker, 1903, and DOI 10.1007/BF01444290")

    try:
        pts = enumerate_D_m(m)
    except Exception as ex:
        _err(errors, "WRD_DECL", f"enumerate_D_m failed: {ex}")
        return errors, warnings

    actual_count = len(pts)
    actual_gmax = compute_g_max(pts) if pts else 0
    actual_dmaxp = delta_max_plus(pts)

    # ---- WRD_W1 cardinality ----
    declared_count = data.get("declared_count")
    if not isinstance(declared_count, int):
        _err(errors, "WRD_DECL", "declared_count must be int")
    elif declared_count != actual_count:
        _err(errors, "WRD_W1",
             f"|D_{m}| mismatch: declared={declared_count}, actual={actual_count}")

    # ---- WRD_HARD per-seed raw form + coprimality ----
    claimed_extra = data.get("claimed_extra_seeds", [])
    if not isinstance(claimed_extra, list):
        _err(errors, "WRD_DECL", "claimed_extra_seeds must be list (possibly empty)")
    else:
        for entry in claimed_extra:
            if not (isinstance(entry, list) and len(entry) == 2
                    and all(isinstance(x, int) for x in entry)):
                _err(errors, "WRD_HARD",
                     f"claimed_extra_seeds entry malformed: {entry!r}")
                continue
            b, e = entry
            if b < 1 or b > m or e < 1 or e > m:
                _err(errors, "WRD_HARD",
                     f"claimed_extra_seed ({b},{e}) out of range [1..{m}]")
                continue
            if gcd(b, e) != 1:
                _err(errors, "WRD_HARD",
                     f"claimed_extra_seed ({b},{e}) is not coprime "
                     f"(gcd={gcd(b, e)}); rejected from D_m")

    claimed_d_overrides = data.get("claimed_d_overrides", [])
    if not isinstance(claimed_d_overrides, list):
        _err(errors, "WRD_DECL", "claimed_d_overrides must be list (possibly empty)")
    else:
        for entry in claimed_d_overrides:
            if not (isinstance(entry, dict)
                    and "b" in entry and "e" in entry and "claimed_d" in entry):
                _err(errors, "WRD_HARD",
                     f"claimed_d_overrides entry malformed: {entry!r}")
                continue
            b, e, dc = entry["b"], entry["e"], entry["claimed_d"]
            if not (isinstance(b, int) and isinstance(e, int) and isinstance(dc, int)):
                _err(errors, "WRD_HARD",
                     f"claimed_d_overrides entry types invalid: {entry!r}")
                continue
            if dc != b + e:
                _err(errors, "WRD_HARD",
                     f"HARD-rule violation: declared d={dc} for (b={b}, e={e}); "
                     f"raw d must equal b+e={b + e}")

    # ---- WRD_W2 angular separation, all pairs ----
    if data.get("run_w2_all_pairs", True):
        if pts:
            n = len(pts)
            min_sin = None
            min_sin_pair = None
            if n * (n - 1) // 2 > 10_000_000:
                warnings.append(
                    f"WRD_W2: pair count {n * (n - 1) // 2} exceeds 10M; "
                    f"sampling consecutive pairs only")
                pair_iter = ((i, i + 1) for i in range(n - 1))
            else:
                pair_iter = ((i, j) for i in range(n) for j in range(i + 1, n))

            for i, j in pair_iter:
                _, _, C1, F1, G1 = pts[i]
                _, _, C2, F2, G2 = pts[j]
                cross = abs(C1 * F2 - C2 * F1)
                if cross < 1:
                    _err(errors, "WRD_W2",
                         f"angular separation cross product = {cross} < 1 "
                         f"for pair (idx={i}, idx={j}); seeds = "
                         f"({pts[i][:2]}, {pts[j][:2]})")
                    break
                sin_v = Fraction(cross, G1 * G2)
                if min_sin is None or sin_v < min_sin:
                    min_sin = sin_v
                    min_sin_pair = (pts[i][:2], pts[j][:2])

            declared_min_sin = data.get("declared_min_sin_separation")
            if declared_min_sin is not None and min_sin is not None:
                try:
                    dms_frac = Fraction(declared_min_sin).limit_denominator(10 ** 12)
                except Exception:
                    dms_frac = None
                if dms_frac is not None and min_sin < dms_frac:
                    _err(errors, "WRD_W2",
                         f"declared_min_sin_separation={declared_min_sin} "
                         f"overclaims; actual min sin(Δθ) ≈ {float(min_sin):.6e} "
                         f"at pair {min_sin_pair}")

    # ---- WRD_W3 Lipschitz nearest-neighbor sampling error ----
    witness = data.get("lipschitz_witness")
    if witness is not None:
        if not isinstance(witness, dict):
            _err(errors, "WRD_DECL", "lipschitz_witness must be dict")
        else:
            fname = witness.get("function_name")
            L = witness.get("L")
            grid = witness.get("test_grid_size", 1024)
            claimed_bound = witness.get("claimed_sup_error_bound")
            if fname not in _TEST_FUNCTIONS:
                _err(errors, "WRD_W3",
                     f"unknown test function {fname!r}")
            elif not isinstance(L, (int, float)) or L <= 0:
                _err(errors, "WRD_W3", f"L must be positive number; got {L!r}")
            elif not isinstance(grid, int) or grid < 16:
                _err(errors, "WRD_W3",
                     f"test_grid_size must be int >= 16; got {grid!r}")
            else:
                actual_sup = w3_sup_error(pts, fname, grid)
                theorem_bound = float(L) * actual_dmaxp
                if actual_sup > theorem_bound + 1e-12:
                    _err(errors, "WRD_W3",
                         f"theorem violation: sup={actual_sup:.6e} > "
                         f"L*Δ_max^+={theorem_bound:.6e}")
                if claimed_bound is not None:
                    if not isinstance(claimed_bound, (int, float)):
                        _err(errors, "WRD_W3",
                             f"claimed_sup_error_bound must be number; "
                             f"got {claimed_bound!r}")
                    elif actual_sup > float(claimed_bound) + 1e-12:
                        _err(errors, "WRD_W3",
                             f"declared sup_error_bound={claimed_bound} too "
                             f"strong; actual sup={actual_sup:.6e}")

    declared_dmaxp_upper = data.get("declared_delta_max_plus_upper_bound")
    if declared_dmaxp_upper is not None:
        if not isinstance(declared_dmaxp_upper, (int, float)):
            _err(errors, "WRD_DECL",
                 "declared_delta_max_plus_upper_bound must be number")
        elif actual_dmaxp > float(declared_dmaxp_upper) + 1e-9:
            _err(errors, "WRD_W3",
                 f"actual Δ_max^+={actual_dmaxp:.6f} exceeds declared "
                 f"upper bound {declared_dmaxp_upper}")

    declared_gmax = data.get("declared_g_max")
    if declared_gmax is not None and declared_gmax != actual_gmax:
        _err(errors, "WRD_W1",
             f"G_max mismatch: declared={declared_gmax}, actual={actual_gmax}")

    # ---- WRD_WIT witnesses ----
    witnesses = data.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 1:
        _err(errors, "WRD_WIT",
             f"need >= 1 witness; got {len(witnesses) if isinstance(witnesses, list) else 'malformed'}")

    # ---- WRD_F fail_ledger well-formed ----
    fl = data.get("fail_ledger")
    if fl is not None:
        if not isinstance(fl, dict):
            _err(errors, "WRD_F", "fail_ledger must be dict if present")
        else:
            for required in ("expected_failure_codes", "rationale"):
                if required not in fl:
                    _err(errors, "WRD_F",
                         f"fail_ledger missing required key {required!r}")

    return errors, warnings


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

_FIXTURES_EXPECTED = [
    ("pass_d9_canonical_net.json", True),
    ("pass_d24_canonical_net.json", True),
    ("pass_d72_canonical_net.json", True),
    ("pass_lipschitz_witness_d9_sin.json", True),
    ("fail_noncoprime_seed.json", False),
    ("fail_nonqa_seed_form.json", False),
    ("fail_w3_bound_too_strong.json", False),
    ("fail_w2_underclaimed_separation.json", False),
]


def _self_test() -> dict:
    fixtures_dir = Path(__file__).parent / "fixtures"
    results = []
    all_ok = True
    for fname, should_pass in _FIXTURES_EXPECTED:
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
            results.append({
                "fixture": fname, "ok": False,
                "error": f"expected PASS but got errors: {errs}",
            })
            all_ok = False
        elif not should_pass and passed:
            results.append({
                "fixture": fname, "ok": False,
                "error": "expected FAIL but got PASS",
            })
            all_ok = False
        else:
            results.append({
                "fixture": fname, "ok": True,
                "errors_seen": errs if not should_pass else [],
            })
    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Whittaker Rational Direction S^1 Cert [266] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True, indent=2))
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
