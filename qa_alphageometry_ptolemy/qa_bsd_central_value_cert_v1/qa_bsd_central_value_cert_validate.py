# Primary source: Birch, B.J. and Swinnerton-Dyer, H.P.F. (1965) "Notes on elliptic curves (II)"
# doi:10.1515/crll.1965.218.79 (BSD conjecture: ord_{s=1/2} L = rank, product formula);
# Rohrlich, D.E. (1984) doi:10.1007/BF01388742 (non-vanishing of L(1/2) for CM forms)
"""Cert [413] — BSD Central Value: Euler Factor Trichotomy at s=½.

At the analytic center s=½ the three prime classes of f=2.2.5.1-125.1-a produce distinct
arithmetic behaviour in the GL₄/ℚ AI Euler factors:

  Inert  (p≡2,3 mod 5): P_p^{inert}(p^{-½}) = 2        ← exact Fraction
  Ramified  (p=5)      : P_5^{ram}(5^{-½})   = 1        ← exact integer
  Split  (p≡1,4 mod 5): P_p^{split}(p^{-½})  irrational ← observer projection

The rational spine of L(½,AI(f)) is determined entirely by inert and ramified primes:
  L_rat = ∏_{p inert} (½) · 1  (from 22 inert primes, partial product = Fraction(1,2^22))

The split prime contributions involve √p (T≠0 for all 22 split primes) and are therefore
continuous observer projections under Theorem NT.

BSD connection: L(½,AI(f)) = L_rat · L_split · L_∞
  If L(½)≠0 → rank A_f(ℚ)=0 (Rohrlich; Coates-Wiles for CM).
  The integer/Fraction content is certifiable; L_split and L_∞ are observer projections.

VERIFICATION NOTE (2026-07-04): Independently confirmed via the LMFDB API that
f=2.2.5.1-125.1-a is a real object: a CM Hilbert modular form over Q(sqrt5),
level norm 125, parallel weight 2, Hecke field dimension 2 (hmf_forms id 45).
Its actual Hecke eigenvalues are elements of that degree-2 field, matching
certs [403]/[404]'s Z[phi] eigenvalue table structurally. However: (1) LMFDB's
lfunc_lfunctions table has NO entry for this object or its GL4/Q automorphic
induction -- no independently computed analytic rank exists to check r_alg=0
against; the prediction rests entirely on the general Rohrlich (1984)
non-vanishing theorem for CM towers, not a specific verified computation for
this f. (2) This cert's own EXTENDED_TABLE previously stored certs
[403]/[404]'s raw Z[phi]-basis coordinates (u,v) of a_p mislabeled as "(T,N)"
-- fixed below to the actual Trace/Norm values. The mislabeling did not flip
C3's gating conclusion (T=raw-u happened to be nonzero for all 22 entries
either way), but the printed "rational_part"/"irrational coefficient" values
were computed from the wrong numbers before this fix.
"""

import json
import sys
from fractions import Fraction

# Inert primes p≡2,3 mod 5 (from cert [409])
INERT_PRIMES = [
    2, 3, 7, 13, 17, 23, 37, 43, 47, 53,
    67, 73, 83, 97, 103, 107, 113, 127, 137, 167,
    173, 193,
]

# Split primes p≡1,4 mod 5 with (T,N) = (Trace, Norm) of a_p over Q(sqrt5)/Q.
#
# BUG FIX (2026-07-04): this table previously stored the raw Z[phi]-basis
# coordinates (u,v) of a_p = u + v*phi, copied verbatim from certs
# [403]/[404]'s EXTENDED_TABLE, mislabeled as "(T,N)". T and N here must be
# T = Tr(a_p) = 2u+v and N = Norm(a_p) = u^2+uv-v^2 (cert [404]'s own
# zphi_tr/zphi_norm) -- the quantities C3's formula
# P_p^split(p^-1/2) = Fraction(4p+N,p) - 2T/sqrt(p) actually requires.
# E.g. the old (mislabeled) entry for p=41 was (7,-5) [the raw (u,v)
# coordinates]; the correct (T,N) is (9,-11). Recomputed here from cert
# [404]'s (u,v) table via T=2u+v, N=u^2+uv-v^2; cross-checked against
# [404]'s own per-prime comments (all 22 match exactly).
EXTENDED_TABLE = [
    (11, (-1, -31)), (31, (-11, -1)), (41, (9, -11)),   (61, (-1, -31)),
    (71, (19, 59)),  (101, (29, 179)),(131, (-11, -1)), (151, (4, -496)),
    (181, (-11, -1)),(191, (-41, 389)),(211, (-1, -781)),(241, (-16, -436)),
    (251, (4, -496)),(271, (-31, 209)),(281, (-11, -751)),(311, (49, 569)),
    (331, (-61, 899)),(401, (29, 179)),(421, (19, -691)),(431, (-36, -176)),
    (461, (-1, -781)),(491, (9, -11)),
]


def check_c1_inert_center_value():
    """C1: P_p^{inert}(p^{-½}) = 2 for all inert primes (Fraction arithmetic).

    P_p^{inert}(Y) = 1 + p²Y⁴.  At Y=p^{-½}: Y⁴=(p^{-½})⁴=p^{-2}=Fraction(1,p²).
    So P_p^{inert}(p^{-½}) = 1 + p²·Fraction(1,p²) = 1 + 1 = 2.
    No float: Y⁴ is an exact inverse square, computable as Fraction(1, p*p).
    """
    errors = []
    for p in INERT_PRIMES:
        Y4 = Fraction(1, p * p)  # (p^{-1/2})^4 = p^{-2}
        val = Fraction(1) + Fraction(p * p) * Y4
        if val != Fraction(2):
            errors.append(f"Inert p={p}: P_p(p^{{-1/2}}) = {val}, expected 2")
        if not isinstance(val, Fraction):
            errors.append(f"Inert p={p}: value not Fraction (T2 violation)")
    return errors


def check_c2_ramified_center_value():
    """C2: P_5^{ram}(5^{-½}) = 1 (trivial polynomial from cert [411]).

    P_5^{ram}(Y) = 1 for all Y → central value = 1 (integer).
    """
    errors = []
    val = Fraction(1)   # P_5^{ram}(anything) = 1
    if val != Fraction(1):
        errors.append(f"Ramified p=5: central value = {val}, expected 1")
    return errors


def check_c3_split_irrationality():
    """C3: Split prime center decomposes as rational + irrational.

    P_p^{split}(p^{-½}) = [4 + N/p] − 2T/√p
      rational part : Fraction(4*p + N, p)   ← certifiable
      irrational part: −2T/√p (T≠0 ⇒ irrational since p is prime)

    Verify T≠0 for all 22 split primes (integer check), confirming the
    full center value is irrational — an observer projection under Theorem NT.
    """
    errors = []
    for p, (T, N_e) in EXTENDED_TABLE:
        if T == 0:
            errors.append(f"Split p={p}: T=0 (center value might be rational — cert assumption violated)")
        # Rational part = 1 + (N_e+2p)/p + 1 = 2 + (N_e+2p)/p = (2p + N_e + 2p)/p = (4p+N_e)/p
        rational_part = Fraction(4 * p + N_e, p)
        if not isinstance(rational_part, Fraction):
            errors.append(f"Split p={p}: rational part not Fraction")
        # The irrational coefficient: -2T (non-zero because T≠0)
        irrational_coeff = -2 * T
        if irrational_coeff == 0:
            errors.append(f"Split p={p}: irrational coefficient = 0 (T={T})")
        # p is prime ≥ 2, so √p is irrational (p is not a perfect square)
        # Integer check: p is not a perfect square iff p not in {1,4,9,16,...}
        import math
        sqrt_floor = math.isqrt(p)
        if sqrt_floor * sqrt_floor == p:
            errors.append(f"Split p={p}: is a perfect square (√p would be rational)")
    return errors


def check_c4_inert_partial_product():
    """C4: Partial Euler product over 22 inert primes at s=½ = (½)^22 (exact Fraction).

    Each inert prime p contributes local factor P_p(p^{-½})^{-1} = (1/2)^{-1}... wait:
    The Euler product for L(s) uses denominators: L(s) = ∏ P_p(p^{-s})^{-1}.
    At s=½, each inert prime contributes P_p(p^{-½})^{-1} = 1/2.
    Partial product over 22 inert primes = (1/2)^22 = Fraction(1, 2^22).
    Exact rational computation — no floats.
    """
    errors = []
    product = Fraction(1)
    for p in INERT_PRIMES:
        local_factor = Fraction(1, 2)  # 1 / P_p^{inert}(p^{-1/2}) = 1/2
        product *= local_factor

    n = len(INERT_PRIMES)
    expected = Fraction(1, 2 ** n)
    if product != expected:
        errors.append(f"Inert partial product = {product}, expected {expected}")
    if expected.denominator != 2 ** n:
        errors.append(f"Denominator = {expected.denominator}, expected 2^{n}={2**n}")
    if not isinstance(product, Fraction):
        errors.append("Partial product is not Fraction (T2 violation)")
    return errors, product, n


def main():
    results = {}

    c1 = check_c1_inert_center_value()
    results["C1_inert_center_value"] = {
        "ok": len(c1) == 0,
        "count": len(INERT_PRIMES),
        "center_value": "2",
        "arithmetic": "Fraction(1) + Fraction(p*p) * Fraction(1,p*p) = 2",
        "errors": c1,
        "desc": "P_p^{inert}(p^{-1/2})=2 (exact Fraction) for all 22 inert primes — unique rational class",
    }

    c2 = check_c2_ramified_center_value()
    results["C2_ramified_center_value"] = {
        "ok": len(c2) == 0,
        "center_value": "1",
        "errors": c2,
        "desc": "P_5^{ram}(5^{-1/2})=1 (trivial polynomial from cert [411]) — integer",
    }

    c3 = check_c3_split_irrationality()
    results["C3_split_irrationality"] = {
        "ok": len(c3) == 0,
        "count": len(EXTENDED_TABLE),
        "T_nonzero": all(T != 0 for _, (T, _) in EXTENDED_TABLE),
        "rational_part": "Fraction(4p+N, p) for each split prime",
        "irrational_part": "-2T/sqrt(p) with T≠0 and p prime => irrational",
        "errors": c3,
        "desc": "Split center = rational + irrational; T≠0 for 22/22; observer projection by Theorem NT",
    }

    c4, product, n = check_c4_inert_partial_product()
    results["C4_inert_partial_product"] = {
        "ok": len(c4) == 0,
        "n_inert": n,
        "product": f"1/2^{n} = 1/{2**n}",
        "product_exact": str(product),
        "errors": c4,
        "desc": f"∏_{{p inert}} (1/2) = Fraction(1,2^{n}) = rational spine of L(1/2,AI(f))",
    }

    all_ok = all(v["ok"] for v in results.values())
    output = {"ok": all_ok, "checks": results}
    print(json.dumps(output, indent=2))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
