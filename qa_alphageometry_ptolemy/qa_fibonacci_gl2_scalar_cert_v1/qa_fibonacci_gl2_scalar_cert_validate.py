"""
Cert [426]: QA Fibonacci GL2 Scalar Identity & Pisano Period

The Fibonacci recurrence matrix M = [[1,1],[1,0]] in GL2(Z) acts on (F_{n+1}, F_n)^T.
Its GL2 invariants:
  - trace:    tr(M^n) = F_{n+1} + F_{n-1} = L_n  (Lucas numbers)
  - det:      det(M^n) = (-1)^n                    (alternating sign)

At the rank of apparition alpha(p) the matrix becomes a SCALAR in GL2(F_p):

    M^{alpha(p)} = epsilon(p) * I_2   (mod p)

where epsilon(p) = F_{alpha(p)-1} mod p satisfies epsilon(p)^2 = (-1)^{alpha(p)} mod p,
so epsilon is always a 4th root of unity in F_p×.

This gives the Pisano period formula:

    T(p) = alpha(p) * ord_{F_p×}(epsilon(p))

where ord(epsilon) in {1, 2, 4}.

Langlands ladder position (GL1 -> GL2 upgrade of cert [423]):
  [423]: (phi/psi)^{alpha(p)} = 1 in GL1(F_p)  -- eigenvalue RATIO is 1
  [426]: M^{alpha(p)} = epsilon*I_2 in GL2(F_p) -- MATRIX is scalar (eigenvalues equal)

Primary sources:
  Lucas, E. (1878) doi:10.2307/2369308 -- Lucas sequences L_n; trace of M^n
  Wall, D.D. (1960) doi:10.2307/2309169 -- Pisano period; Fibonacci mod m structure
"""

import math
from fractions import Fraction  # noqa: F401


# ── QA LAYER: pure integer arithmetic ────────────────────────────────────────

def _fib_pair(n, m):
    """Return (F_n mod m, F_{n+1} mod m) via fast doubling. Iterative O(log n)."""
    if n == 0:
        return 0, 1
    a, b = 0, 1  # (F_0, F_1)
    bits = n.bit_length()
    for shift in range(bits - 1, -1, -1):
        # Double: F_{2k} = F_k*(2*F_{k+1} - F_k), F_{2k+1} = F_k*F_k + F_{k+1}*F_{k+1}
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
    a, b = _fib_pair(n, m)
    return a


def rank_of_apparition(p):
    """Smallest k>=1 with F_k = 0 mod p. Pure integer walk."""
    a, b = 0, 1  # (F_0, F_1)
    k = 0
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0:
            return k


def pisano_period(p):
    """Pisano period T(p): smallest k>0 with (F_k, F_{k+1}) = (0, 1) mod p.
    Pure integer walk; terminates since F is periodic mod p (Lagrange)."""
    a, b = 0, 1  # (F_0, F_1)
    k = 0
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0 and b == 1:
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
    """Order of g in (Z/pZ)×. Lagrange-based reduction. Pure integer."""
    g = g % p
    if g == 0:
        raise ValueError("g=0 has no multiplicative order")
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


def lucas(n):
    """L_n (Lucas number). L_0=2, L_1=1, L_n = L_{n-1}+L_{n-2}. Pure integer."""
    if n == 0:
        return 2
    if n == 1:
        return 1
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


# ── CHECK FUNCTIONS ───────────────────────────────────────────────────────────

def check_c1_lucas_trace(n_max=50):
    """C1: tr(M^n) = F_{n+1} + F_{n-1} = L_n for n=1..n_max.

    M^n = [[F_{n+1}, F_n], [F_n, F_{n-1}]], so tr(M^n) = F_{n+1}+F_{n-1}.
    Identity: F_{n+1}+F_{n-1} = L_n (Lucas, 1878; equivalent to tr(M^n)=L_n).
    Checked in exact integer arithmetic -- no float used.
    """
    fails = []
    for n in range(1, n_max + 1):
        fn_m1, fn = _fib_pair(n - 1, 10 ** 18)   # (F_{n-1}, F_n), exact int
        fn_p1 = fn_m1 + fn                         # F_{n+1} = F_{n-1}+F_n raw (no mod)
        fn_m1_raw = fn_m1                          # F_{n-1} raw
        trace = fn_p1 + fn_m1_raw                  # tr(M^n) = F_{n+1}+F_{n-1}
        ln = lucas(n)
        if trace != ln:
            fails.append((n, trace, ln))
    return {
        "ok": len(fails) == 0,
        "n_checked": n_max,
        "fails": fails,
        "desc": f"tr(M^n)=L_n for n=1..{n_max}: {n_max-len(fails)}/{n_max} PASS",
    }


def check_c2_det(n_max=50):
    """C2: det(M^n) = (-1)^n for n=1..n_max.

    det(M) = F_1*F_{-1} - F_0^2. By Cassini: det(M^n) = det(M)^n = (-1)^n.
    Direct check: det([[F_{n+1},F_n],[F_n,F_{n-1}]]) = F_{n+1}*F_{n-1}-F_n^2 = (-1)^n.
    Pure integer arithmetic (Cassini identity, no float).
    """
    fails = []
    for n in range(1, n_max + 1):
        fn_m1, fn = _fib_pair(n - 1, 10 ** 18)   # (F_{n-1}, F_n), exact
        fn_p1 = fn_m1 + fn                         # F_{n+1}
        det = fn_p1 * fn_m1 - fn * fn              # Cassini determinant
        expected = (-1) ** n
        if det != expected:
            fails.append((n, det, expected))
    return {
        "ok": len(fails) == 0,
        "n_checked": n_max,
        "fails": fails,
        "desc": f"det(M^n)=(-1)^n for n=1..{n_max}: {n_max-len(fails)}/{n_max} PASS",
    }


def check_c3_scalar_matrix(n_bound=500):
    """C3: M^{alpha(p)} = epsilon(p)*I_2 mod p for all primes 7..n_bound.

    M^{alpha(p)} = [[F_{alpha+1}, F_alpha],[F_alpha, F_{alpha-1}]] mod p.
    F_{alpha} = 0 mod p (by definition of alpha(p)).
    F_{alpha+1} = F_alpha + F_{alpha-1} = 0 + F_{alpha-1} = F_{alpha-1} mod p.
    So M^{alpha} = [[F_{alpha-1}, 0],[0, F_{alpha-1}]] = F_{alpha-1}*I_2.
    epsilon(p) = F_{alpha(p)-1} mod p.
    """
    fails = []
    tested = 0
    epsilon_counts = {1: 0, 2: 0, 4: 0}
    for p in sieve(n_bound):
        if p < 7:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        f_alpha = fib_fast(alpha, p)
        fn_m1, fn = _fib_pair(alpha - 1, p)        # (F_{alpha-1}, F_{alpha})
        eps = fn_m1                                  # epsilon = F_{alpha-1} mod p
        fn_p1 = (fn_m1 + fn) % p                   # F_{alpha+1} mod p

        # Off-diagonal must be 0 (f_alpha = 0) and diagonal both = eps
        if f_alpha != 0:
            fails.append((p, alpha, "F_alpha!=0", f_alpha))
        elif fn_p1 != eps:
            fails.append((p, alpha, "F_{alpha+1}!=F_{alpha-1}", fn_p1, eps))
        else:
            # Track epsilon order for summary
            ord_e = mult_order(eps, p)
            if ord_e in epsilon_counts:
                epsilon_counts[ord_e] += 1

    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "epsilon_order_counts": epsilon_counts,
        "desc": (
            f"M^alpha = eps*I2 for {tested-len(fails)}/{tested} primes; "
            f"eps ord: {epsilon_counts}"
        ),
    }


def check_c4_pisano_period(n_bound=300):
    """C4: Pisano period T(p) = alpha(p) * ord_{F_p×}(epsilon(p)) for primes 7..n_bound.

    From C3: M^{alpha} = eps*I. So M^{alpha*k} = eps^k*I.
    M^T = I iff eps^{T/alpha} = 1 iff ord(eps) | T/alpha.
    Minimum T = alpha * ord(eps).
    Independently verify T via direct period walk and formula via ord(eps).
    Pure integer throughout (ord via prime factorisation of p-1).
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 7:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        fn_m1, _ = _fib_pair(alpha - 1, p)
        eps = fn_m1
        ord_eps = mult_order(eps, p)

        t_formula = alpha * ord_eps
        t_direct = pisano_period(p)

        if t_formula != t_direct:
            fails.append((p, alpha, eps, ord_eps, t_formula, t_direct))

    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"T(p)=alpha(p)*ord(eps) for {tested-len(fails)}/{tested} primes "
            f"up to {n_bound}: formula matches direct walk"
        ),
    }


# ── OBSERVER LAYER: summary statistics (float, lawful output-only) ────────────

def _run_checks():
    checks = {
        "c1_lucas_trace": check_c1_lucas_trace(50),
        "c2_det":         check_c2_det(50),
        "c3_scalar":      check_c3_scalar_matrix(500),
        "c4_pisano":      check_c4_pisano_period(300),
    }
    return checks


def main():
    import json
    results = _run_checks()
    all_ok = all(v["ok"] for v in results.values())
    print(json.dumps({
        "cert": "[426] QA Fibonacci GL2 Scalar Identity & Pisano Period",
        "all_checks_pass": all_ok,
        "checks": {k: {"ok": v["ok"], "desc": v["desc"]} for k, v in results.items()},
    }, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
