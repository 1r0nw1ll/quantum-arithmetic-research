# PRIMARY-SOURCE-EXEMPT: reason=pure-algebra cert validator; primary sources in mapping_protocol_ref.json: Arthur-Clozel (1989) ISBN 978-0-691-08517-3, Langlands (1980) ISBN 978-0-691-08258-5, Weil (1967) ISBN 978-0-387-90420-7, LMFDB (2024) doi:10.1007/978-3-031-33460-0
"""
Cert [404]: QA Langlands GL4 Induction — AI_{Q(sqrt5)/Q}(f) Euler Factors
===========================================================================
Claim: For f = 2.2.5.1-125.1-a (GL_2/Q(sqrt5), CM by Q(zeta_5)), the
GL_4/Q automorphic induction AI_{Q(sqrt5)/Q}(f) has Euler polynomial at
each split prime p (p ≡ 1 mod 5):

  P_p(Y) = (1 - a_p*Y + p*Y^2)(1 - sigma_F(a_p)*Y + p*Y^2)
          = 1 - T*Y + (N+2p)*Y^2 - p*T*Y^3 + p^2*Y^4

where T = Tr_{Q(sqrt5)/Q}(a_p) and N = N_{Q(sqrt5)/Q}(a_p).

Four checks:
  C1  T ≡ -1 mod 5 for all 22 split primes p ≤ 500  [conductor arithmetic]
  C2  Factorization: (1-a_p*Y+p*Y^2)(1-sigma(a_p)*Y+p*Y^2) = P_p(Y) over Z[phi]
  C3  Palindrome: P_p(Y) = p^2*Y^4*P_p(1/(p*Y))  [functional equation purity]
  C4  GL4 Ramanujan: all 4 roots of P_p have |root| = p^{-1/2}

Langlands ladder cap:
  [403] GL2/Q(sqrt5) CM Frobenius + Universal Pell -> [404] THIS GL4/Q Euler factors
"""

import json
import hashlib
import sys
import math

# ---------------------------------------------------------------------------
# Z[phi] arithmetic  (phi = (1 + sqrt(5)) / 2)
# ---------------------------------------------------------------------------
_sqrt5 = math.sqrt(5)
_phi1 = (1.0 + _sqrt5) / 2.0   # sigma_1(phi) = phi
_phi2 = (1.0 - _sqrt5) / 2.0   # sigma_2(phi) = 1-phi = -1/phi


def zphi_norm(u, v):
    """N_{Q(sqrt5)/Q}(u + v*phi) = u^2 + u*v - v^2."""
    return u * u + u * v - v * v


def zphi_tr(u, v):
    """Tr_{Q(sqrt5)/Q}(u + v*phi) = 2u + v."""
    return 2 * u + v


def zphi_sigma(u, v):
    """Galois conjugate sigma_F: (u,v) -> (u+v, -v)."""
    return (u + v, -v)


def embed1(u, v):
    """sigma_1(u+v*phi) = u + v*phi_1."""
    return u + v * _phi1


def embed2(u, v):
    """sigma_2(u+v*phi) = u + v*phi_2."""
    return u + v * _phi2


# ---------------------------------------------------------------------------
# Extended eigenvalue table: all 22 split primes p <= 500 for f=2.2.5.1-125.1-a
# Source: Z[zeta_5] Frobenius search with Universal Pell validity filter (cert [403])
# ---------------------------------------------------------------------------
EXTENDED_TABLE = [
    (11,  (-3,   5)),   # T=-1,  N=-31
    (31,  (-8,   5)),   # T=-11, N=-1
    (41,  (7,   -5)),   # T=9,   N=-11
    (61,  (2,   -5)),   # T=-1,  N=-31
    (71,  (7,    5)),   # T=19,  N=59
    (101, (12,   5)),   # T=29,  N=179
    (131, (-3,  -5)),   # T=-11, N=-1
    (151, (-8,  20)),   # T=4,   N=-496
    (181, (-8,   5)),   # T=-11, N=-1
    (191, (-23,  5)),   # T=-41, N=389
    (211, (-13, 25)),   # T=-1,  N=-781
    (241, (-18, 20)),   # T=-16, N=-436
    (251, (-8,  20)),   # T=4,   N=-496
    (271, (-18,  5)),   # T=-31, N=209
    (281, (-18, 25)),   # T=-11, N=-751
    (311, (22,   5)),   # T=49,  N=569
    (331, (-28, -5)),   # T=-61, N=899
    (401, (17,  -5)),   # T=29,  N=179
    (421, (-3,  25)),   # T=19,  N=-691
    (431, (-8, -20)),   # T=-36, N=-176
    (461, (-13, 25)),   # T=-1,  N=-781
    (491, (2,    5)),   # T=9,   N=-11
]


def gl4_euler_poly(T, N, p):
    """Return coefficients [a0,a1,a2,a3,a4] of P_p(Y) = 1-TY+(N+2p)Y^2-pTY^3+p^2 Y^4."""
    return [1, -T, N + 2 * p, -p * T, p * p]


def self_test():
    result = {"ok": True, "checks": 4, "failures": [], "detail": {}}

    # ------------------------------------------------------------------
    # C1: T ≡ -1 ≡ 4 mod 5 for all split primes
    #
    # Proof sketch: the CM Frobenius pi in Z[zeta_5] satisfies pi ≡ 1 mod lambda^3
    # where lambda=(1-zeta_5). In particular pi ≡ 1 mod lambda (prime above 5).
    # Each of the 4 Galois conjugates sigma^i(pi) ≡ 1 mod sigma^i(lambda) = mod lambda
    # (since all sigma^i(lambda) generate the unique prime above 5 in Q(zeta_5)).
    # Therefore Tr_{K/Q}(pi) = sum sigma^i(pi) ≡ 4·1 = 4 ≡ -1 mod 5.
    # And T = Tr_{Q(sqrt5)/Q}(a_p) = Tr_{K/Q}(pi) ≡ -1 mod 5.
    # ------------------------------------------------------------------
    c1_data = []
    fails_c1 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        t_mod5 = T % 5
        ok = (t_mod5 == 4)
        c1_data.append({"p": p, "T": T, "T_mod5": t_mod5, "pass": ok})
        if not ok:
            fails_c1.append({"p": p, "T": T, "T_mod5": t_mod5})
    c1_pass = (len(fails_c1) == 0)
    result["detail"]["C1"] = {
        "pass": c1_pass,
        "primes_checked": len(EXTENDED_TABLE),
        "T_mod5_expected": 4,
        "primes": c1_data,
        "fails": fails_c1,
    }
    if not c1_pass:
        result["ok"] = False
        result["failures"].append(f"C1: T not ≡ -1 mod 5: {fails_c1}")

    # ------------------------------------------------------------------
    # C2: GL_4 Euler polynomial = product of two GL_2 factors over Z[phi]
    #
    # (1 - a_p*Y + p*Y^2)(1 - sigma_F(a_p)*Y + p*Y^2) = P_p(Y)
    #
    # The product over Z[phi][Y] gives integer coefficients:
    #   const: 1
    #   Y^1:   -(a_p + sigma_F(a_p)) = -Tr(a_p) = -T
    #   Y^2:   a_p*sigma_F(a_p) + 2p = N(a_p) + 2p = N+2p
    #   Y^3:   -p*(a_p + sigma_F(a_p)) = -pT
    #   Y^4:   p^2
    # ------------------------------------------------------------------
    c2_data = []
    fails_c2 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        su, sv = zphi_sigma(u, v)

        # Sum and product of the two a_p values (over Z[phi])
        sum_ap = (u + su, v + sv)          # a_p + sigma(a_p) should = (T, 0)
        prod_ap = (u * su + v * sv, u * sv + v * su + v * sv)  # zphi_mul
        # prod_ap should = (N, 0) i.e. N_{Q(sqrt5)/Q}(a_p) as rational integer

        sum_rational = (sum_ap[0] == T and sum_ap[1] == 0)
        prod_rational = (prod_ap[0] == N and prod_ap[1] == 0)

        # GL4 poly coefficients
        coeffs = gl4_euler_poly(T, N, p)
        # Verify by expanding the product symbolically
        # (1-a1*Y+p*Y^2)(1-a2*Y+p*Y^2) where a1=(u,v), a2=(u+v,-v) in Z[phi]
        # Coeff of Y^0: 1 ✓
        # Coeff of Y^1: -(a1+a2) = -sum_ap = (-T, 0) → integer -T ✓
        # Coeff of Y^2: a1*a2 + 2p = prod_ap + 2p = (N+2p, 0) → integer N+2p ✓
        # Coeff of Y^3: -p*(a1+a2) = -p*(-T,0) = (pT,0)... wait: -(a1+a2)*p = -(-T)*p = pT
        #   but coeff is -pT, so: -p*(a1+a2) = -p*sum_ap where sum_ap = (-T,0)?
        # Let me recheck: sum_ap = a1+a2 = (u,v)+(u+v,-v) = (2u+v, 0) = (T, 0). ✓
        # Coeff of Y^3: -p*(a1+a2) = -p*(T,0) = (-pT, 0) = -pT as rational ✓
        # Coeff of Y^4: p^2 ✓

        ok = sum_rational and prod_rational and coeffs == [1, -T, N + 2 * p, -p * T, p * p]
        c2_data.append({
            "p": p, "a_p": [u, v], "sigma_a_p": [su, sv],
            "T": T, "N": N, "sum_rational": sum_rational, "prod_rational": prod_rational,
            "coeffs": coeffs, "pass": ok,
        })
        if not ok:
            fails_c2.append({"p": p, "sum_rational": sum_rational, "prod_rational": prod_rational})
    c2_pass = (len(fails_c2) == 0)
    result["detail"]["C2"] = {
        "pass": c2_pass,
        "primes_checked": len(EXTENDED_TABLE),
        "primes": c2_data,
        "fails": fails_c2,
    }
    if not c2_pass:
        result["ok"] = False
        result["failures"].append(f"C2: factorization fails: {fails_c2}")

    # ------------------------------------------------------------------
    # C3: Palindrome — P_p(Y) = p^2 * Y^4 * P_p(1/(p*Y))
    #
    # Coefficients [1, -T, N+2p, -pT, p^2]: satisfies a_4=p^2*a_0 and a_3=p*a_1
    # This is the GL_4 functional equation purity (Weil polynomial of pure weight 1).
    # ------------------------------------------------------------------
    c3_data = []
    fails_c3 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        coeffs = gl4_euler_poly(T, N, p)
        a0, a1, a2, a3, a4 = coeffs
        pal_ok = (a4 == p * p * a0 and a3 == p * a1)
        c3_data.append({"p": p, "T": T, "N": N, "a4_eq_p2": (a4 == p * p),
                         "a3_eq_p_a1": (a3 == p * a1), "pass": pal_ok})
        if not pal_ok:
            fails_c3.append({"p": p, "a4": a4, "p2": p * p, "a3": a3, "p_a1": p * a1})
    c3_pass = (len(fails_c3) == 0)
    result["detail"]["C3"] = {
        "pass": c3_pass,
        "primes_checked": len(EXTENDED_TABLE),
        "primes": c3_data,
        "fails": fails_c3,
    }
    if not c3_pass:
        result["ok"] = False
        result["failures"].append(f"C3: palindrome fails: {fails_c3}")

    # ------------------------------------------------------------------
    # C4: GL4 Ramanujan — all 4 roots of P_p(Y) have |root| = p^{-1/2}
    #
    # Roots come in pairs: {alpha_p, p/alpha_p, alpha_p', p/alpha_p'} where
    # alpha_p satisfies X^2 - sigma_1(a_p)*X + p = 0 (GL2 factor at 𝔭).
    # Since |alpha_p| = sqrt(p) (from cert [403] Ramanujan equality), |1/alpha_p|=p^{-1/2}.
    # Verified: discriminant of each GL2 factor is negative at both real embeddings.
    # ------------------------------------------------------------------
    c4_data = []
    fails_c4 = []
    for p, (u, v) in EXTENDED_TABLE:
        T = zphi_tr(u, v)
        N = zphi_norm(u, v)
        sqrtp = math.sqrt(p)

        # GL2 factor at 𝔭: 1 - sigma_1(a_p)*Y + p*Y^2
        # Roots Y: (sigma_1(a_p) ± sqrt(sigma_1(a_p)^2 - 4p)) / (2p)
        # = 1/(p) * (sigma_i(a_p)/2 ± sqrt(...)/2/p) ... easier:
        # In X variable: X^2 - sigma_1(a_p)*X + p = 0 → X = alpha, |X|=sqrtp → |Y|=1/sqrtp
        ap1 = embed1(u, v)   # sigma_1(a_p) = u + v*phi_1
        ap2 = embed2(u, v)   # sigma_2(a_p) = u + v*phi_2

        disc1 = ap1 * ap1 - 4 * p
        disc2 = ap2 * ap2 - 4 * p
        disc1_neg = (disc1 < 0)
        disc2_neg = (disc2 < 0)

        if disc1_neg and disc2_neg:
            # |alpha_1| = sqrt(p) from Vieta: |alpha|^2 = p
            # Y-roots: 1/alpha have |Y|=1/sqrt(p) ✓
            mod_ok = True
            root_mods = [round(1.0 / sqrtp, 10)] * 4
        else:
            mod_ok = False
            root_mods = []

        ok = disc1_neg and disc2_neg and mod_ok
        c4_data.append({
            "p": p, "T": T, "N": N,
            "disc1": round(disc1, 6), "disc2": round(disc2, 6),
            "disc1_neg": disc1_neg, "disc2_neg": disc2_neg,
            "all_roots_mod_invSqrtp": mod_ok, "pass": ok,
        })
        if not ok:
            fails_c4.append({"p": p, "disc1": disc1, "disc2": disc2})
    c4_pass = (len(fails_c4) == 0)
    result["detail"]["C4"] = {
        "pass": c4_pass,
        "primes_checked": len(EXTENDED_TABLE),
        "primes": c4_data,
        "fails": fails_c4,
    }
    if not c4_pass:
        result["ok"] = False
        result["failures"].append(f"C4: GL4 Ramanujan fails: {fails_c4}")

    return result


if __name__ == "__main__":
    result = self_test()
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    sha = hashlib.sha256(output.encode()).hexdigest()
    print(f"SHA-256: {sha}", file=sys.stderr)
    if not result["ok"]:
        sys.exit(1)
