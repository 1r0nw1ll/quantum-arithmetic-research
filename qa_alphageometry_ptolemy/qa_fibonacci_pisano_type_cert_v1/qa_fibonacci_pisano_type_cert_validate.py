"""
Cert [427]: QA Fibonacci Pisano Type Distribution

From cert [426]: M^{alpha(p)} = epsilon(p) * I_2 mod p, where epsilon is a 4th root
of unity. The three Pisano types are:
  Type 1  (T = alpha)    :  epsilon = +1
  Type 2  (T = 2*alpha)  :  epsilon = -1
  Type 4  (T = 4*alpha)  :  epsilon = primitive 4th root (epsilon^2 = -1)

This cert certifies four structural claims:

  C1 (Type gate):
    Type 4 occurs iff p ≡ 1 mod 4 AND alpha(p) is odd.
    Proof: epsilon^2 = (-1)^alpha mod p. If alpha odd: epsilon^2 = -1.
    -1 is a QR mod p iff p ≡ 1 mod 4 (Euler's criterion).
    For p ≡ 3 mod 4: epsilon^2 = -1 has no solution, so alpha must be even.
    Verified: 92 primes in [7,500].

  C2 (Lucas bridge):
    L_{alpha(p)} ≡ 2 * epsilon(p) mod p for all primes p.
    Proof: F_{alpha+1} = F_{alpha} + F_{alpha-1} ≡ 0 + epsilon = epsilon mod p.
    L_n = F_{n+1} + F_{n-1}, so L_{alpha} = epsilon + epsilon = 2*epsilon.
    The GL_2 trace (Lucas number) equals twice the GL_2 scalar.
    Verified: 92 primes in [7,500].

  C3 (equal halves for p ≡ 3 mod 4):
    Among primes p ≡ 3 mod 4 in [7,10000], Types 1 and 2 occur with
    approximately equal frequency. Chi-squared (df=1) = 0.026 < 3.841 (alpha=0.05).
    N = 618 primes.

  C4 (equal thirds over all primes):
    Among all primes in [7,10000], Types 1, 2, 4 each occur with approximately
    equal frequency (density 1/3 each). Chi-squared (df=2) = 0.046 < 5.991 (alpha=0.05).
    N = 1226 primes.

    The equal-thirds density is explained by the following composition:
      p ≡ 3 mod 4 (density 1/2, Types 1+2 only): each type density 1/4.
      p ≡ 1 mod 4 (density 1/2, Types 1+2+4):  2/3 of this class is Type 4,
        1/6 each for Types 1 and 2.
    Combined: Type 1 = 1/4 + 1/12 = 1/3; Type 2 = 1/3; Type 4 = 0 + 1/3 = 1/3.
    The 2/3 fraction of Type 4 within p ≡ 1 mod 4 is a Chebotarev-type statement
    about the density of primes with odd alpha(p) in that class.

Primary sources:
  Lucas, E. (1878) doi:10.2307/2369308 -- Lucas numbers L_n
  Wall, D.D. (1960) doi:10.2307/2309169 -- Pisano period structure
  Euler, L. (1750) "Theoremata circa divisores numerorum" --
    Euler's criterion: -1 is QR mod p iff p ≡ 1 mod 4 (Legendre symbol (-1/p))
"""

import math
from fractions import Fraction  # noqa: F401


# ── QA LAYER: pure integer arithmetic ────────────────────────────────────────

def _fib_pair(n, m):
    """(F_n, F_{n+1}) mod m via fast doubling. Iterative O(log n). Pure integer."""
    if n == 0:
        return 0, 1
    a, b = 0, 1
    bits = n.bit_length()
    for shift in range(bits - 1, -1, -1):
        c = a * (2 * b - a) % m
        d = (a * a + b * b) % m
        if (n >> shift) & 1:
            a, b = d, (c + d) % m
        else:
            a, b = c, d
    return a, b


def fib_fast(n, m):
    """F_n mod m. Pure integer."""
    if n == 0:
        return 0
    return _fib_pair(n, m)[0]


def rank_of_apparition(p):
    """Smallest k>=1 with F_k=0 mod p. Pure integer walk."""
    a, b = 0, 1
    k = 0
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0:
            return k


def _prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def mult_order(g, p):
    """Order of g in (Z/pZ)×. Lagrange reduction. Pure integer."""
    g = g % p
    if g == 0:
        raise ValueError("g=0 has no order")
    n = p - 1
    for q in _prime_factors(n):
        while n % q == 0 and pow(g, n // q, p) == 1:
            n //= q
    return n


def sieve(n):
    """Primes up to n. Pure integer."""
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]


# ── CHECK FUNCTIONS ───────────────────────────────────────────────────────────

def check_c1_type_gate(n_bound=500):
    """C1: Type 4 iff p ≡ 1 mod 4 AND alpha(p) is odd.

    Forward: alpha odd → epsilon^2 = (-1)^{odd} = -1 mod p.
    -1 is a QR mod p iff p ≡ 1 mod 4 (Euler). For p ≡ 3 mod 4: no solution → alpha must be even.
    Backward: p ≡ 1 mod 4 AND alpha odd → epsilon^2 = -1 has solutions → ord(epsilon)=4.
    """
    fails_3mod4 = []   # p ≡ 3 mod 4 should have ord(eps) ≤ 2
    fails_type4 = []   # Type 4 primes should all satisfy p ≡ 1 mod 4 AND alpha odd
    tested = 0
    for p in sieve(n_bound):
        if p < 7:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        fn_m1, _ = _fib_pair(alpha - 1, p)
        eps = fn_m1
        ord_eps = mult_order(eps, p)
        if p % 4 == 3 and ord_eps == 4:
            fails_3mod4.append((p, alpha, eps, ord_eps))
        if ord_eps == 4 and not (p % 4 == 1 and alpha % 2 == 1):
            fails_type4.append((p, alpha, eps, ord_eps, "expected p%4=1 AND alpha%2=1"))
    return {
        "ok": len(fails_3mod4) == 0 and len(fails_type4) == 0,
        "primes_tested": tested,
        "fails_p3mod4_has_type4": fails_3mod4,
        "fails_type4_wrong_class": fails_type4,
        "desc": (
            f"Type gate: no Type 4 for p≡3 mod 4; "
            f"all Type 4 have p≡1 mod 4 AND alpha odd; "
            f"{tested} primes ≤{n_bound}"
        ),
    }


def check_c2_lucas_bridge(n_bound=500):
    """C2: L_{alpha(p)} ≡ 2 * epsilon(p) mod p.

    Proof (pure QA layer):
      F_{alpha(p)} ≡ 0 mod p                    [definition of alpha]
      F_{alpha+1}  = F_{alpha} + F_{alpha-1}
                   ≡ 0 + epsilon                [epsilon = F_{alpha-1}]
      L_{alpha} = F_{alpha+1} + F_{alpha-1}
               = epsilon + epsilon = 2*epsilon   mod p.
    The Lucas number L_{alpha(p)} equals 2 * (GL_2 scalar epsilon(p)) mod p.
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 7:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        fn_m1, fn = _fib_pair(alpha - 1, p)   # (F_{alpha-1}, F_{alpha}) = (eps, 0)
        eps = fn_m1
        f_alpha_p1 = (fn + fn_m1) % p          # F_{alpha+1} = F_{alpha} + F_{alpha-1}
        l_alpha = (f_alpha_p1 + fn_m1) % p     # L_{alpha} = F_{alpha+1} + F_{alpha-1}
        expected = (2 * eps) % p
        if l_alpha != expected:
            fails.append((p, alpha, eps, l_alpha, expected))
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": f"L_{{alpha(p)}} = 2*eps mod p: {tested-len(fails)}/{tested} primes ≤{n_bound} PASS",
    }


def check_c3_equal_halves_p3mod4(n_bound=10000):
    """C3: Among p ≡ 3 mod 4 in [7, n_bound]: Types 1 and 2 occur equally.
    Chi-squared (df=1) < 3.841 (alpha=0.05). Observer: Pearson chi^2.
    Type 4 is absent (from C1).
    """
    counts = {1: 0, 2: 0}
    n_total = 0
    for p in sieve(n_bound):
        if p < 7 or p % 4 != 3:
            continue
        alpha = rank_of_apparition(p)
        fn_m1, _ = _fib_pair(alpha - 1, p)
        eps = fn_m1
        ord_eps = mult_order(eps, p)
        counts[ord_eps] = counts.get(ord_eps, 0) + 1
        n_total += 1
    expected = n_total / 2
    # Observer layer (float — lawful statistical test):
    chi2 = sum((counts.get(t, 0) - expected) ** 2 / expected for t in [1, 2])
    critical = 3.841  # chi²(df=1, alpha=0.05)
    return {
        "ok": chi2 < critical,
        "n_primes": n_total,
        "counts": counts,
        "chi2": round(chi2, 4),
        "critical": critical,
        "desc": (
            f"p≡3 mod 4: equal halves chi²={round(chi2,4)} < {critical} "
            f"(df=1, alpha=0.05); n={n_total}; counts={counts}"
        ),
    }


def check_c4_equal_thirds_all(n_bound=10000):
    """C4: Among all primes in [7, n_bound]: Types 1, 2, 4 occur with equal density 1/3.
    Chi-squared (df=2) < 5.991 (alpha=0.05). Observer: Pearson chi^2.

    Decomposition:
      p≡3 mod 4 (density 1/2): Types 1+2 with equal density → each contributes 1/4.
      p≡1 mod 4 (density 1/2): Type 4 dominates at ~2/3 of this class;
        Types 1+2 split the remaining ~1/3 equally.
      Combined: each of Types 1, 2, 4 has density 1/3.

    The Type 4 fraction among p≡1 mod 4 (~2/3) is a Chebotarev statement:
    density of odd-alpha primes within p≡1 mod 4 equals 2/3.
    """
    counts = {1: 0, 2: 0, 4: 0}
    type4_mod4 = {1: 0, 3: 0}   # verify all Type 4 primes have p≡1 mod 4
    n_total = 0
    for p in sieve(n_bound):
        if p < 7:
            continue
        alpha = rank_of_apparition(p)
        fn_m1, _ = _fib_pair(alpha - 1, p)
        eps = fn_m1
        ord_eps = mult_order(eps, p)
        counts[ord_eps] = counts.get(ord_eps, 0) + 1
        if ord_eps == 4:
            type4_mod4[p % 4] = type4_mod4.get(p % 4, 0) + 1
        n_total += 1
    expected = n_total / 3
    # Observer layer:
    chi2 = sum((counts.get(t, 0) - expected) ** 2 / expected for t in [1, 2, 4])
    critical = 5.991  # chi²(df=2, alpha=0.05)
    all_type4_are_p1mod4 = type4_mod4.get(3, 0) == 0
    return {
        "ok": chi2 < critical and all_type4_are_p1mod4,
        "n_primes": n_total,
        "counts": counts,
        "chi2": round(chi2, 4),
        "critical": critical,
        "type4_mod4_distribution": type4_mod4,
        "all_type4_are_p1mod4": all_type4_are_p1mod4,
        "desc": (
            f"All primes: equal thirds chi²={round(chi2,4)} < {critical} "
            f"(df=2, alpha=0.05); n={n_total}; counts={counts}; "
            f"Type 4 mod 4: {type4_mod4}"
        ),
    }


# ── OBSERVER LAYER: summary (float, lawful output-only) ──────────────────────

def _run_checks():
    return {
        "c1_type_gate":         check_c1_type_gate(500),
        "c2_lucas_bridge":      check_c2_lucas_bridge(500),
        "c3_equal_halves_p3m4": check_c3_equal_halves_p3mod4(10000),
        "c4_equal_thirds_all":  check_c4_equal_thirds_all(10000),
    }


def main():
    import json
    results = _run_checks()
    all_ok = all(v["ok"] for v in results.values())
    print(json.dumps({
        "cert": "[427] QA Fibonacci Pisano Type Distribution",
        "all_checks_pass": all_ok,
        "checks": {k: {"ok": v["ok"], "desc": v["desc"]} for k, v in results.items()},
    }, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
