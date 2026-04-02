"""
Quantum Arithmetic Feature Map v3.0

Produces:
    - qa21: 21 canonical invariants
    - qa27: canonical-21 + 6 canonical-expanded invariants
    - qa83: full stack (canonical + derived + modular + physical + ML)

QA tuple rules:
    d = b + e
    a = b + 2e
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def compute_qa_invariants(b: float, e: float) -> Dict[str, float]:
    """Compute canonical 21 + 6 canonical-expanded invariants from (b, e),
    then augment with derived (21), modular (12), physical (14), ML (9).
    """
    d = b + e
    a = b + 2.0 * e

    # Squares
    B = b * b
    E = e * e
    D = d * d
    A = a * a

    # Triangle legs
    X = e * d
    C = 2.0 * X           # base leg
    F = b * a             # altitude leg
    G = D + E             # hypotenuse

    # Secondary triangle/ellipse composites
    L = (C * F) / 12.0
    H = C + F
    I = abs(C - F)

    # Ellipse multipliers
    J = b * d
    K = d * a
    W = d * (e + a)       # also X + K
    Y = A - D
    Z = E + K
    h = d * math.sqrt(abs(a * b))  # semi-minor diameter

    def safe_div(num: float, den: float) -> float:
        return num / den if den != 0.0 else 0.0

    # Canonical-expanded (6)
    eps = safe_div(e, a)
    ratio_FC = safe_div(F, C)
    ratio_GC = safe_div(G, C)
    R_h = math.sqrt(B + E + D + A)
    E_QA = G * G - C * F
    theta = math.atan2(e, b)

    inv: Dict[str, float] = {
        # Canonical 21
        "b": b, "e": e, "d": d, "a": a,
        "B": B, "E": E, "D": D, "A": A,
        "X": X, "C": C, "F": F, "G": G,
        "L": L, "H": H, "I": I,
        "J": J, "K": K, "W": W, "Y": Y, "Z": Z, "h": h,
        # Expanded 6
        "eps": eps, "F_over_C": ratio_FC, "G_over_C": ratio_GC,
        "R_h": R_h, "E_QA": E_QA, "theta": theta,
    }

    # Derived algebraic (21)
    a_over_b = safe_div(a, b); inv["a_over_b"] = a_over_b
    d_over_b = safe_div(d, b); inv["d_over_b"] = d_over_b
    a_over_d = safe_div(a, d); inv["a_over_d"] = a_over_d
    d_over_e = safe_div(d, e); inv["d_over_e"] = d_over_e
    b_over_e = safe_div(b, e); inv["b_over_e"] = b_over_e
    e_over_b = safe_div(e, b); inv["e_over_b"] = e_over_b

    sum_ad = a + d; sum_be = b + e
    sum_AD = A + D; sum_BE = B + E
    inv["ratio_ad_be"] = safe_div(sum_ad, sum_be)
    inv["ratio_AD_BE"] = safe_div(sum_AD, sum_BE)
    inv["ratio_a_d_sym"] = safe_div(a - d, a + d)

    inv["X_over_J"] = safe_div(X, J)
    inv["K_over_J"] = safe_div(K, J)
    inv["G_over_F"] = safe_div(G, F)
    inv["C_over_G"] = safe_div(C, G)

    inv["AB_over_ED"] = safe_div(A * B, (E * D) + 1e-12)
    inv["cross_abed"] = a * b * e * d

    sum_CFG = C + F + G; prod_CFG = C * F * G
    inv["sum_CFG"] = sum_CFG; inv["prod_CFG"] = prod_CFG

    inv["H_over_G"] = safe_div(H, G)
    inv["I_over_G"] = safe_div(I, G)
    inv["J_over_K"] = safe_div(J, K)
    inv["Y_over_Z"] = safe_div(Y, Z)

    # Modular / resonance (12)
    def int_floor(x: float) -> int: return int(math.floor(x))
    def digital_root9(x: float) -> float:
        n = abs(int_floor(x));
        if n == 0: return 0.0
        return (1 + ((n - 1) % 9)) / 9.0
    def mod_norm(x: float, m: float) -> float: return (x % m) / m
    def parity(x: float) -> float:
        n = int_floor(abs(x)); return float(n % 2)

    inv["dr9_b"] = digital_root9(b)
    inv["dr9_e"] = digital_root9(e)
    inv["dr9_d"] = digital_root9(d)
    inv["dr9_a"] = digital_root9(a)
    inv["phi9"] = mod_norm(b + e + d + a, 9.0)
    inv["phi24"] = mod_norm(b + e + d + a, 24.0)
    n_a = abs(int_floor(a)); inv["phase24_gcd"] = (math.gcd(n_a, 24) / 24.0) if n_a != 0 else 0.0
    inv["parity_b"] = parity(b); inv["parity_e"] = parity(e); inv["parity_a"] = parity(a)
    inv["parity_mean"] = (inv["parity_b"] + inv["parity_e"] + inv["parity_a"]) / 3.0
    inv["resonance_fib"] = (inv["dr9_b"] + inv["dr9_e"]) / 2.0
    inv["resonance_lucas"] = (inv["dr9_d"] + inv["dr9_a"]) / 2.0

    # Physical / geometric (14)
    kappa = safe_div(a, e)
    sin_theta = math.sin(theta); cos_theta = math.cos(theta)
    area_tri = 0.5 * C * F
    r_in = b * e; R_circ = 0.5 * G
    slender_CF = safe_div(C - F, C + F)
    lambda_rectum = 2.0 * F
    semi_major = D; semi_minor = math.sqrt(abs(F))
    ellipse_area = math.pi * semi_major * semi_minor
    triangle_pyth_error = G * G - (C * C + F * F)
    aspect_G_over_sumCF = safe_div(G, C + F)
    inv["kappa_a_over_e"] = kappa; inv["sin_theta"] = sin_theta; inv["cos_theta"] = cos_theta
    inv["area_tri"] = area_tri; inv["r_in"] = r_in; inv["R_circ"] = R_circ
    inv["slender_CF"] = slender_CF; inv["lambda_rectum"] = lambda_rectum
    inv["semi_major"] = semi_major; inv["semi_minor"] = semi_minor; inv["ellipse_area"] = ellipse_area
    inv["triangle_pyth_error"] = triangle_pyth_error; inv["aspect_G_over_sumCF"] = aspect_G_over_sumCF
    inv["E_QA_norm"] = safe_div(E_QA, (G * G + C * F + 1e-9))

    # ML convenience (9)
    norm_triangle = math.sqrt(C * C + F * F + G * G)
    norm_ellipse = math.sqrt(J * J + X * X + K * K)
    norm_mixed = math.sqrt(H * H + I * I + W * W + Y * Y + Z * Z)
    sum_abs_tuple = abs(b) + abs(e) + abs(d) + abs(a) + 1e-9
    nb = b / sum_abs_tuple; ne = e / sum_abs_tuple; nd = d / sum_abs_tuple; na = a / sum_abs_tuple
    cos_angle_CG = safe_div(C * C + G * G - F * F, 2.0 * C * G) if (C != 0.0 and G != 0.0) else 0.0
    harmonic_density = (abs(E_QA) + abs(L) + abs(H) + abs(I)) / (1.0 + C * C + F * F + G * G)
    inv["norm_triangle"] = norm_triangle; inv["norm_ellipse"] = norm_ellipse; inv["norm_mixed"] = norm_mixed
    inv["norm_b"] = nb; inv["norm_e"] = ne; inv["norm_d"] = nd; inv["norm_a"] = na
    inv["cos_angle_CG"] = cos_angle_CG; inv["harmonic_density"] = harmonic_density

    return inv


CANONICAL_21 = [
    "b", "e", "d", "a",
    "B", "E", "D", "A",
    "X", "C", "F", "G",
    "L", "H", "I",
    "J", "K", "W", "Y", "Z", "h",
]

EXPANDED_6 = [
    "eps", "F_over_C", "G_over_C", "R_h", "E_QA", "theta",
]

DERIVED_21 = [
    "a_over_b", "d_over_b", "a_over_d", "d_over_e",
    "b_over_e", "e_over_b",
    "ratio_ad_be", "ratio_AD_BE", "ratio_a_d_sym",
    "X_over_J", "K_over_J", "G_over_F", "C_over_G",
    "AB_over_ED", "cross_abed",
    "sum_CFG", "prod_CFG",
    "H_over_G", "I_over_G", "J_over_K", "Y_over_Z",
]

MODULAR_12 = [
    "dr9_b", "dr9_e", "dr9_d", "dr9_a",
    "phi9", "phi24", "phase24_gcd",
    "parity_b", "parity_e", "parity_a", "parity_mean",
    "resonance_fib",
]

PHYSICAL_14 = [
    "kappa_a_over_e", "sin_theta", "cos_theta",
    "area_tri", "r_in", "R_circ",
    "slender_CF", "lambda_rectum",
    "semi_major", "semi_minor", "ellipse_area",
    "triangle_pyth_error", "aspect_G_over_sumCF",
    "E_QA_norm",
]

ML_9 = [
    "norm_triangle", "norm_ellipse", "norm_mixed",
    "norm_b", "norm_e", "norm_d", "norm_a",
    "cos_angle_CG", "harmonic_density",
]


def qa_feature_vector(b: float, e: float, mode: str = "qa21") -> Tuple[np.ndarray, List[str]]:
    """Return feature vector and names for a given (b, e).

    mode in {"qa21", "qa27", "qa83"}:
        - "qa21": canonical 21 invariants
        - "qa27": canonical 21 + 6 expanded invariants
        - "qa83": full extended set (canonical + derived + modular + physical + ML)
    """
    inv = compute_qa_invariants(b, e)

    if mode == "qa21":
        names = CANONICAL_21
    elif mode == "qa27":
        names = CANONICAL_21 + EXPANDED_6
    elif mode == "qa83":
        names = (
            CANONICAL_21
            + EXPANDED_6
            + DERIVED_21
            + MODULAR_12
            + PHYSICAL_14
            + ML_9
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    vec = np.array([inv[name] for name in names], dtype=float)
    return vec, names

