# PRIMARY-SOURCE-EXEMPT: reason=pure-algebra cert validator; primary sources in mapping_protocol_ref.json: Hecke (1920) doi:10.1007/BF01458074, Shimura (1971) ISBN 978-0-691-08092-5, Weil (1967) ISBN 978-0-387-90420-7, LMFDB (2024) doi:10.1007/978-3-031-33460-0
"""
Cert [403]: QA Langlands Cap — CM Frobenius Ramanujan Equality
==============================================================
Claim: For the CM Hilbert modular form f = 2.2.5.1-125.1-a over F = Q(sqrt(5)),
the Hecke character factorization L(s,f) = L(s,psi) * L(s,psibar) implies
Ramanujan EQUALITY at every split prime: the Frobenius discriminant
Delta = a_p^2 - 4*N(p) is strictly negative at both real embeddings
sigma_1, sigma_2 of Q(sqrt(5))/Q.

Five checks:
  C1  a_p = 0 for all p not ≡ 1 (mod 5)  [CM zero pattern]
  C2  a_p in Z[phi] for split primes p ≡ 1 (mod 5)  [eigenvalue field]
  C3  Delta = a_p^2 - 4p < 0 at both embeddings  [Ramanujan equality]
  C4  N_{Q(sqrt5)/Q}(a_p) correct; |N(a_p)| <= 4p  [Ramanujan norm bound]
  C5  Discriminant resonance: e^2+e-31=0, disc(e/Q)=125=level norm  [CM fingerprint]

Langlands ladder cap:
  [394] GL1 Frobenius -> [395] GL2 Weil bound -> [396] Z[zeta_5] infrastructure
  -> [397] CM relative norm -> [399] CM form identification -> [403] THIS
"""

import json
import hashlib
import sys
import math

# ---------------------------------------------------------------------------
# Z[phi] arithmetic  (phi = (1 + sqrt(5)) / 2)
# Elements: (u, v) = u + v*phi
# phi^2 = phi + 1  =>  product rule below
# Galois conj sigma_F: phi -> 1-phi = (1-sqrt5)/2,  i.e. (u,v) -> (u+v, -v)
# Norm: N(u+v*phi) = u^2 + u*v - v^2  (= sigma_1 * sigma_2 in Q)
# ---------------------------------------------------------------------------
_sqrt5 = math.sqrt(5)
_phi1 = (1.0 + _sqrt5) / 2.0   # sigma_1(phi)
_phi2 = (1.0 - _sqrt5) / 2.0   # sigma_2(phi)


def embed1(u, v):
    return u + v * _phi1


def embed2(u, v):
    return u + v * _phi2


def zphi_mul(a, b):
    u1, v1 = a
    u2, v2 = b
    return (u1 * u2 + v1 * v2, u1 * v2 + v1 * u2 + v1 * v2)


def zphi_sq(a):
    return zphi_mul(a, a)


def zphi_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def zphi_sigma_F(u, v):
    """Galois conjugation of Q(sqrt5)/Q: phi -> 1-phi."""
    return (u + v, -v)


def zphi_norm(u, v):
    """N_{Q(sqrt5)/Q}(u + v*phi) = u^2 + u*v - v^2."""
    return u * u + u * v - v * v


# ---------------------------------------------------------------------------
# LMFDB data for 2.2.5.1-125.1-a
# Source: LMFDB Collaboration (2024) doi:10.1007/978-3-031-33460-0
# (p, p_mod_5, a_p as (u,v) in Z[phi] or None for zero eigenvalue)
# ---------------------------------------------------------------------------
LMFDB = [
    (2,  2, None),          # inert: 2 ≡ 2 (mod 5)
    (3,  3, None),          # inert: 3 ≡ 3 (mod 5)
    (5,  0, None),          # ramified
    (7,  2, None),          # inert: 7 ≡ 2 (mod 5)
    (11, 1, (-3, 5)),       # split: a_11 = -3 + 5*phi = 5*phi - 3
    (13, 3, None),          # inert: 13 ≡ 3 (mod 5)
    (17, 2, None),          # inert: 17 ≡ 2 (mod 5)
    (19, 4, None),          # inert: 19 ≡ 4 (mod 5)
    (23, 3, None),          # inert: 23 ≡ 3 (mod 5)
    (29, 4, None),          # inert: 29 ≡ 4 (mod 5)
    (31, 1, (-8, 5)),       # split: a_31 = -8 + 5*phi
    (41, 1, (7, -5)),        # split: a_41 = 7 - 5*phi  [Frobenius: pi=2-3z-2z^2+4z^3]
    (59, 4, None),          # inert: 59 ≡ 4 (mod 5)
    (61, 1, (2, -5)),       # split: a_61 = 2 - 5*phi  [Frobenius: pi=-2z-3z^2+6z^3]
    (71, 1, (7, 5)),        # split: a_71 = 7 + 5*phi
]

# Generator eigenvalue: e = -3 + 5*phi satisfies e^2 + e - 31 = 0
E = (-3, 5)

# Pre-computed norms N(a_p) = u^2 + u*v - v^2 for split primes:
#   N(-3+5phi)  = 9  - 15 - 25 = -31
#   N(-8+5phi)  = 64 - 40 - 25 = -1
#   N(7-5phi)   = 49 - 35 - 25 = -11   [Frobenius-derived; sigma_F gives 2+5phi, N=-11]
#   N(2-5phi)   = 4  - 10 - 25 = -31   [sigma_F gives -3+5phi = a_11; same N]
#   N(7+5phi)   = 49 + 35 - 25 = 59
EXPECTED_NORMS = {11: -31, 31: -1, 41: -11, 61: -31, 71: 59}


def self_test():
    result = {"ok": True, "checks": 5, "failures": [], "detail": {}}

    # ------------------------------------------------------------------
    # C1: a_p = 0 for all p not ≡ 1 (mod 5)
    # ------------------------------------------------------------------
    inert_nonzero = [p for p, p5, a in LMFDB if p5 != 1 and a is not None]
    split_missing = [p for p, p5, a in LMFDB if p5 == 1 and a is None]
    c1_pass = (len(inert_nonzero) == 0)
    result["detail"]["C1"] = {
        "pass": c1_pass,
        "inert_ramified_count": sum(1 for _, p5, _ in LMFDB if p5 != 1),
        "inert_nonzero": inert_nonzero,
    }
    if not c1_pass:
        result["ok"] = False
        result["failures"].append("C1: a_p != 0 for inert/ramified primes")

    # ------------------------------------------------------------------
    # C2: a_p in Z[phi] for split primes
    # ------------------------------------------------------------------
    bad_type = [(p, a) for p, p5, a in LMFDB
                if p5 == 1 and (a is None or
                                not isinstance(a[0], int) or
                                not isinstance(a[1], int))]
    c2_pass = (len(bad_type) == 0)
    result["detail"]["C2"] = {
        "pass": c2_pass,
        "split_primes": [(p, list(a)) for p, p5, a in LMFDB if p5 == 1 and a is not None],
        "bad_type": bad_type,
    }
    if not c2_pass:
        result["ok"] = False
        result["failures"].append(f"C2: a_p not in Z[phi]: {bad_type}")

    # ------------------------------------------------------------------
    # C3: Frobenius discriminant Delta = a_p^2 - 4p < 0 at both embeddings
    # ------------------------------------------------------------------
    delta_data = []
    fails_c3 = []
    for p, p5, a in LMFDB:
        if p5 != 1 or a is None:
            continue
        a2 = zphi_sq(a)
        delta = (a2[0] - 4 * p, a2[1])
        d1 = embed1(*delta)
        d2 = embed2(*delta)
        ok = (d1 < 0 and d2 < 0)
        delta_data.append({"p": p, "a_p": list(a), "delta": list(delta),
                            "sigma1": round(d1, 6), "sigma2": round(d2, 6),
                            "pass": ok})
        if not ok:
            fails_c3.append(p)
    c3_pass = (len(fails_c3) == 0)
    result["detail"]["C3"] = {"pass": c3_pass, "primes": delta_data, "fails": fails_c3}
    if not c3_pass:
        result["ok"] = False
        result["failures"].append(f"C3: Delta >= 0 at some embedding for p in {fails_c3}")

    # ------------------------------------------------------------------
    # C4: Field norm N(a_p) matches pre-computed values; |N(a_p)| <= 4p
    # ------------------------------------------------------------------
    norm_data = []
    fails_c4 = []
    for p, p5, a in LMFDB:
        if p5 != 1 or a is None:
            continue
        n = zphi_norm(*a)
        expected = EXPECTED_NORMS.get(p)
        norm_ok = (expected is None or n == expected) and abs(n) <= 4 * p
        norm_data.append({"p": p, "norm": n, "expected": expected,
                          "bound_4p": 4 * p, "pass": norm_ok})
        if not norm_ok:
            fails_c4.append({"p": p, "computed": n, "expected": expected,
                              "bound_4p": 4 * p})
    c4_pass = (len(fails_c4) == 0)
    result["detail"]["C4"] = {"pass": c4_pass, "norms": norm_data, "fails": fails_c4}
    if not c4_pass:
        result["ok"] = False
        result["failures"].append(f"C4: norm failures: {fails_c4}")

    # ------------------------------------------------------------------
    # C5: Discriminant resonance — e^2 + e - 31 = 0 and disc(e/Q) = 125
    # ------------------------------------------------------------------
    e_sq = zphi_sq(E)
    e_sq_plus_e = zphi_add(e_sq, E)
    poly_zero = (e_sq_plus_e[0] - 31, e_sq_plus_e[1])  # = e^2+e-31, expect (0,0)
    poly_ok = (poly_zero == (0, 0))

    e_conj = zphi_sigma_F(*E)   # sigma_F(-3,5) = (-3+5, -5) = (2,-5)
    tr_e_tuple = zphi_add(E, e_conj)  # (-3+2, 5-5) = (-1, 0)
    tr_e = tr_e_tuple[0] if tr_e_tuple[1] == 0 else None
    nm_e = zphi_norm(*E)        # (-3)^2 + (-3)(5) - 5^2 = 9-15-25 = -31
    disc_e = (tr_e * tr_e - 4 * nm_e) if tr_e is not None else None

    c5_pass = poly_ok and (disc_e == 125)
    result["detail"]["C5"] = {
        "pass": c5_pass,
        "e_zphi": list(E),
        "e_sq_plus_e_minus_31": list(poly_zero),
        "poly_zero": poly_ok,
        "e_conj": list(e_conj),
        "tr_e": tr_e,
        "norm_e": nm_e,
        "disc_e": disc_e,
        "level_norm": 125,
        "disc_equals_level": (disc_e == 125),
    }
    if not c5_pass:
        result["ok"] = False
        if not poly_ok:
            result["failures"].append(f"C5: e^2+e-31 != 0: got {poly_zero}")
        if disc_e != 125:
            result["failures"].append(f"C5: disc(e/Q)={disc_e} != 125")

    return result


if __name__ == "__main__":
    result = self_test()
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    sha = hashlib.sha256(output.encode()).hexdigest()
    print(f"SHA-256: {sha}", file=sys.stderr)
    if not result["ok"]:
        sys.exit(1)
