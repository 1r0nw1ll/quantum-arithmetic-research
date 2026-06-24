"""
IGP24 Competition — QA-Derived Polynomials
SAIR Inverse Galois Problem degree-24, Stage 1 closes 2026-08-15.
Verification: Magma. Scoring: by discriminant (smaller = better).

Reference: qa_inverse_galois_degree24_cert_v1 [506]
Tao T. et al. (2026) SAIR IGP24 competition.
"""

from __future__ import annotations

# ── 24T1  (C_24, cyclic of order 24) ─────────────────────────────────────────
#
# Construction:
#   K24 = K3 ⊗ K8,  K3 = Q(2cos(2π/7)),  K8 = Q(2cos(2π/17))
#   Generator: α = 2cos(2π/7) + 2cos(2π/17)
#   Computed via: Res_y(p3(y), p8(x-y)) where
#     p3 = y³ + y² - 2y - 1      (min poly of 2cos(2π/7), conductor 7)
#     p8 = y⁸ + y⁷ - 7y⁶ - …   (min poly of 2cos(2π/17), conductor 17)
#
# Field discriminant: 7^16 × 17^21  (log10 ≈ 39.36)
#   — minimal discriminant for C24 totally real field by conductor-discriminant formula
#   — prime splitting confirms C3 × C8 = C24:
#       mod 7  → (degree 8, multiplicity 3)  → e=3, f=8, g=1
#       mod 17 → (degree 3, multiplicity 8)  → e=8, f=3, g=1
#
# T-number: 24T1

C24_T1_COEFFS = [
    1, 11, 17, -222, -873, 974, 9643, 6555,
    -43710, -70566, 78895, 233115, -1634, -346701, -170019,
    223085, 195174, -36603, -75227, -11626, 8317, 2638, 45, -31, 1
]
# Degree 24 to degree 0 (high-to-low)

C24_T1_DISC_FIELD = "7^16 * 17^21"
C24_T1_LOG10_DISC = 39.36


# ── 24T2  (C_12 × C_2) ───────────────────────────────────────────────────────
# Φ_{39}(x), conductor 39 = 3 × 13
# Gal ≅ (Z/39Z)* ≅ (Z/3Z)* × (Z/13Z)* ≅ C_2 × C_12

from sympy import cyclotomic_poly, symbols as _sym

_x = _sym('x')

def _to_coeffs(poly_expr):
    from sympy import Poly
    return [int(c) for c in Poly(poly_expr, _x).all_coeffs()]

C12C2_T2_COEFFS = _to_coeffs(cyclotomic_poly(39, _x))
C12C2_T2_CONDUCTOR = 39


# ── 24T3  (C_2^2 × C_6) ──────────────────────────────────────────────────────
# Φ_{56}(x), conductor 56 = 8 × 7
# Gal ≅ (Z/56Z)* ≅ (Z/8Z)* × (Z/7Z)* ≅ C_2 × C_2 × C_6

C2C2C6_T3_COEFFS = _to_coeffs(cyclotomic_poly(56, _x))
C2C2C6_T3_CONDUCTOR = 56


# ── Format helpers ────────────────────────────────────────────────────────────

def to_magma_poly(coeffs: list[int]) -> str:
    """High-to-low degree list → Magma Polynomial([low, ..., high]) string."""
    return f"Polynomial([{', '.join(map(str, reversed(coeffs)))}])"


def to_pari_poly(coeffs: list[int]) -> str:
    """High-to-low degree list → PARI/GP Pol([high, ..., low]) string."""
    return f"Pol([{', '.join(map(str, coeffs))}])"


# ── Verification ──────────────────────────────────────────────────────────────

def verify_c24() -> None:
    import numpy as np
    roots = np.roots(C24_T1_COEFFS)
    n_real = sum(abs(r.imag) < 1e-8 for r in roots)
    assert n_real == 24, f"Expected 24 real roots, got {n_real}"
    print(f"C24 ({len(roots)} roots): all real = {n_real == 24}")

    from sympy.polys.galoistools import gf_factor
    from sympy import ZZ

    def modfac(p):
        c = [coef % p for coef in C24_T1_COEFFS]
        facs = gf_factor(c, p, ZZ)
        return [(len(f[0]) - 1, f[1]) for f in facs[1]]

    fac7  = modfac(7)
    fac17 = modfac(17)
    print(f"mod 7  (deg, mult): {fac7}  → e={fac7[0][1]}, f={fac7[0][0]}")
    print(f"mod 17 (deg, mult): {fac17} → e={fac17[0][1]}, f={fac17[0][0]}")
    assert fac7  == [(8, 3)], f"mod 7 unexpected: {fac7}"
    assert fac17 == [(3, 8)], f"mod 17 unexpected: {fac17}"
    print("Prime splitting confirms C3 × C8 = C24 ✓")
    print()
    print("Magma:")
    print(f"  f := {to_magma_poly(C24_T1_COEFFS)};")
    print(f"  GaloisGroup(NumberField(f));  // expect C24")
    print(f"  Discriminant(NumberField(f)); // expect 7^16 * 17^21")


if __name__ == "__main__":
    verify_c24()
    print()
    print("T2 Magma:", f"f := {to_magma_poly(C12C2_T2_COEFFS)};")
    print("T3 Magma:", f"f := {to_magma_poly(C2C2C6_T3_COEFFS)};")
