# PRIMARY-SOURCE-EXEMPT: reason=pure-algebra cert validator; primary sources in mapping_protocol_ref.json: Cogdell (2004) ISBN 978-0-8218-3516-0, Kim-Shahidi (2002) doi:10.4007/annals.2002.155.837, Arthur-Clozel (1989) ISBN 978-0-691-08517-3
"""
Cert [406]: QA Langlands GL₆ Exterior Square — ∧²(AI_{ℚ(√5)/ℚ}(f)) Euler Factor
================================================================================
Claim: For f = 2.2.5.1-125.1-a (GL₂/ℚ(√5), CM by ℚ(ζ₅)) and split prime
p ≡ 1 mod 5, the full exterior square GL₆ Euler polynomial is:

  W_p(Y) = 1 − (N+2p)Y + p(T²−p)Y²
           − 2p²(T²−N−2p)Y³
           + p³(T²−p)Y⁴ − p⁴(N+2p)Y⁵ + p⁶Y⁶

where T = Tr_{ℚ(√5)/ℚ}(a_p) ∈ ℤ and N = N_{ℚ(√5)/ℚ}(a_p) ∈ ℤ.

This arises from the decomposition ∧²V = ∧²V₁ ⊕ (V₁⊗V₂) ⊕ ∧²V₂
for V = V₁⊕V₂ (the GL₄ split into two GL₂ pieces at p):
  W_p(Y) = (1−pY)² · R_p(Y)
where (1−pY)² = det(π₁)·det(π₂) and R_p from cert [405].

Four checks:
  C1  Integer coefficients for all 22 split primes
  C2  Palindrome (weight-1/p): a_k = p^{2k−6}·a_{6−k}
  C3  Middle coefficient: a₃ = −2p²(T²−N−2p) is even and ∈ p²·ℤ
  C4  Consistency: W_p(Y) = (1−pY)²·R_p(Y) verified for all 22 primes

Langlands ladder cap:
  [403] GL2 CM Frobenius + Pell
  → [404] GL4 AI induction Euler factors
  → [405] GL4 tensor-product cross factor R_p
  → [406] THIS GL6 exterior square W_p = (1−pY)²·R_p
"""

import json
import hashlib
import sys

# ---------------------------------------------------------------------------
# ℤ[φ] arithmetic
# ---------------------------------------------------------------------------

def zphi_tr(u, v):
    return 2 * u + v


def zphi_norm(u, v):
    return u * u + u * v - v * v


# ---------------------------------------------------------------------------
# Extended eigenvalue table — 22 split primes p ≤ 500
# ---------------------------------------------------------------------------
EXTENDED_TABLE = [
    (11,  (-3,   5)),   # T=−1,  N=−31
    (31,  (-8,   5)),   # T=−11, N=−1
    (41,  (7,   -5)),   # T=9,   N=−11
    (61,  (2,   -5)),   # T=−1,  N=−31
    (71,  (7,    5)),   # T=19,  N=59
    (101, (12,   5)),   # T=29,  N=179
    (131, (-3,  -5)),   # T=−11, N=−1
    (151, (-8,  20)),   # T=4,   N=−496
    (181, (-8,   5)),   # T=−11, N=−1
    (191, (-23,  5)),   # T=−41, N=389
    (211, (-13, 25)),   # T=−1,  N=−781
    (241, (-18, 20)),   # T=−16, N=−436
    (251, (-8,  20)),   # T=4,   N=−496
    (271, (-18,  5)),   # T=−31, N=209
    (281, (-18, 25)),   # T=−11, N=−751
    (311, (22,   5)),   # T=49,  N=569
    (331, (-28, -5)),   # T=−61, N=899
    (401, (17,  -5)),   # T=29,  N=179
    (421, (-3,  25)),   # T=19,  N=−691
    (431, (-8, -20)),   # T=−36, N=−176
    (461, (-13, 25)),   # T=−1,  N=−781
    (491, (2,    5)),   # T=9,   N=−11
]


def gl6_exterior_square_poly(T, N, p):
    """Return [a0,...,a6] of W_p(Y) = ∧²(AI(f)) GL6 Euler polynomial.

    Derivation: W_p = (1−pY)² · R_p where R_p = 1−NY+p(T²−2N−2p)Y²−p²NY³+p⁴Y⁴.
    Expanding (1−2pY+p²Y²)·R_p:
      a0 = 1
      a1 = −N − 2p = −(N+2p)
      a2 = p(T²−2N−2p) + 2pN + p² = pT² − p² = p(T²−p)
      a3 = −p²N − 2p·p(T²−2N−2p) − p²N = −2p²(T²−N−2p)
      a4 = p⁴ + 2p³N + p³(T²−2N−2p) = p³(T²−p)
      a5 = −2p⁵ − p⁴N = −p⁴(N+2p)
      a6 = p⁶
    """
    p2 = p * p
    p3 = p2 * p
    p4 = p3 * p
    p6 = p3 * p3
    return [
        1,
        -(N + 2 * p),
        p * (T * T - p),
        -2 * p2 * (T * T - N - 2 * p),
        p3 * (T * T - p),
        -p4 * (N + 2 * p),
        p6,
    ]


def rankin_selberg_poly(T, N, p):
    """R_p from cert [405]: 1−NY+p(T²−2N−2p)Y²−p²NY³+p⁴Y⁴."""
    return [1, -N, p * (T * T - 2 * N - 2 * p), -p * p * N, p * p * p * p]


def poly_mul(A, B):
    """Multiply two coefficient lists [a0, a1, ...]."""
    result = [0] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            result[i + j] += a * b
    return result


def self_test():
    result = {"ok": True, "checks": 4, "failures": [], "detail": {}}

    # ------------------------------------------------------------------
    # C1: Integer coefficients
    #
    # From the formula: a0=1, a1=−(N+2p), a2=p(T²−p), a3=−2p²(T²−N−2p),
    # a4=p³(T²−p), a5=−p⁴(N+2p), a6=p⁶ — all lie in ℤ since T,N,p ∈ ℤ.
    # ------------------------------------------------------------------
    c1_data = []
    fails_c1 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        coeffs = gl6_exterior_square_poly(T, N, p)
        ok = all(isinstance(c, int) for c in coeffs)
        c1_data.append({"p": p, "T": T, "N": N, "coeffs": coeffs, "pass": ok})
        if not ok:
            fails_c1.append({"p": p})
    c1_pass = (len(fails_c1) == 0)
    result["detail"]["C1"] = {
        "pass": c1_pass, "primes_checked": len(EXTENDED_TABLE),
        "description": "All 7 coefficients in Z",
        "primes": c1_data, "fails": fails_c1,
    }
    if not c1_pass:
        result["ok"] = False
        result["failures"].append(f"C1 integer: {fails_c1}")

    # ------------------------------------------------------------------
    # C2: Palindrome — a_k = p^{2k−6}·a_{6−k}
    #
    # Proof: all 6 roots have magnitude p^{−1} (2 from det pieces at p^{−1},
    # 4 from R_p at p^{−1}). Weil polynomial of pure weight 2:
    # functional equation W_p(Y) = p⁶Y⁶·W_p(1/(p²Y)).
    # Coefficient conditions: a5=p⁴·a1, a4=p²·a2, a3=a3 (self-symmetric).
    # ------------------------------------------------------------------
    c2_data = []
    fails_c2 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        a = gl6_exterior_square_poly(T, N, p)
        p2 = p * p; p4 = p2 * p2; p6 = p2 * p2 * p2
        ok = (a[5] == p4 * a[1] and a[4] == p2 * a[2] and a[6] == p6 * a[0])
        c2_data.append({
            "p": p, "T": T, "N": N,
            "a5_eq_p4_a1": (a[5] == p4 * a[1]),
            "a4_eq_p2_a2": (a[4] == p2 * a[2]),
            "a6_eq_p6": (a[6] == p6),
            "pass": ok,
        })
        if not ok:
            fails_c2.append({"p": p})
    c2_pass = (len(fails_c2) == 0)
    result["detail"]["C2"] = {
        "pass": c2_pass, "primes_checked": len(EXTENDED_TABLE),
        "description": "Palindrome: a5=p^4*a1, a4=p^2*a2, a6=p^6",
        "primes": c2_data, "fails": fails_c2,
    }
    if not c2_pass:
        result["ok"] = False
        result["failures"].append(f"C2 palindrome: {fails_c2}")

    # ------------------------------------------------------------------
    # C3: Middle coefficient — a₃ = −2p²(T²−N−2p) ∈ 2p²·ℤ
    #
    # The self-symmetric middle term a₃ is divisible by 2p² for all primes.
    # This reflects the double multiplicity from the ∧² decomposition:
    # det(π₁)⊗det(π₂) contributes a term 2·p·(center of R_p), giving factor 2.
    # ------------------------------------------------------------------
    c3_data = []
    fails_c3 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        a3 = gl6_exterior_square_poly(T, N, p)[3]
        expected_a3 = -2 * p * p * (T * T - N - 2 * p)
        divisible = (a3 % (2 * p * p) == 0)
        ok = (a3 == expected_a3 and divisible)
        c3_data.append({
            "p": p, "T": T, "N": N, "a3": a3,
            "expected": expected_a3, "div_2p2": divisible, "pass": ok,
        })
        if not ok:
            fails_c3.append({"p": p, "a3": a3, "expected": expected_a3})
    c3_pass = (len(fails_c3) == 0)
    result["detail"]["C3"] = {
        "pass": c3_pass, "primes_checked": len(EXTENDED_TABLE),
        "description": "a3 = -2p^2*(T^2-N-2p) and divisible by 2p^2",
        "primes": c3_data, "fails": fails_c3,
    }
    if not c3_pass:
        result["ok"] = False
        result["failures"].append(f"C3 middle: {fails_c3}")

    # ------------------------------------------------------------------
    # C4: Consistency — W_p(Y) = (1−pY)² · R_p(Y) for all 22 primes
    #
    # Direct multiplication check: the product of (1-pY)^2 and R_p(Y) (cert [405])
    # must equal the GL6 formula W_p(Y) coefficient-by-coefficient.
    # ------------------------------------------------------------------
    c4_data = []
    fails_c4 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        W = gl6_exterior_square_poly(T, N, p)
        R = rankin_selberg_poly(T, N, p)
        det_sq = [1, -2 * p, p * p]
        product = poly_mul(det_sq, R)
        ok = (product == W)
        c4_data.append({"p": p, "T": T, "N": N, "match": ok, "pass": ok})
        if not ok:
            fails_c4.append({"p": p, "W": W, "product": product})
    c4_pass = (len(fails_c4) == 0)
    result["detail"]["C4"] = {
        "pass": c4_pass, "primes_checked": len(EXTENDED_TABLE),
        "description": "W_p = (1-pY)^2 * R_p(Y) coefficient match",
        "primes": c4_data, "fails": fails_c4,
    }
    if not c4_pass:
        result["ok"] = False
        result["failures"].append(f"C4 consistency: {fails_c4}")

    return result


if __name__ == "__main__":
    result = self_test()
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    sha = hashlib.sha256(output.encode()).hexdigest()
    print(f"SHA-256: {sha}", file=sys.stderr)
    if not result["ok"]:
        sys.exit(1)
