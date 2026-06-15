"""
Cert [428]: QA Fibonacci Frobenius Congruences & Wall Divisibility

The Fibonacci matrix M = [[1,1],[1,0]] in GL_2(Z) has eigenvalues phi, psi
(roots of x^2 - x - 1). The Frobenius element Frob_p acts on these as:
  Split  (p = +/-1 mod 5): phi, psi in F_p;          Frob_p fixes both.
  Inert  (p = +/-2 mod 5): phi, psi in F_{p^2}, not in F_p; Frob_p swaps phi <-> psi.
  Ramified (p = 5):        phi = psi in F_5.

Four consequences, each a pure integer check:

  C1 (Fibonacci-Legendre):  F_p = (5/p) mod p.
    Split: phi^p=phi, psi^p=psi -> F_p = (phi-psi)/(phi-psi) = +1 = (5/p).
    Inert: phi^p=psi, psi^p=phi -> F_p = (psi-phi)/(phi-psi) = -1 = (5/p).
    Ramified: F_5 = 5 = 0 = (5/5).
    (5/p) denotes the Legendre symbol: +1 (split), -1 (inert), 0 (ramified).

  C2 (Lucas-Fermat):  L_p = 1 mod p.
    L_p = phi^p + psi^p.
    Split: phi^p=phi, psi^p=psi -> phi+psi = 1 (from x^2-x-1: sum of roots = 1).
    Inert: phi^p=psi, psi^p=phi -> psi+phi = 1.
    Both give 1 regardless of splitting type. L_p = 1 mod p universally.

  C3 (Wall zero):  F_{p - (5/p)} = 0 mod p.
    Split ((5/p)=+1): F_{p-1} = 0 mod p.
      Proof: eigenvalues phi,psi in F_p; by FLT phi^{p-1}=psi^{p-1}=1;
      so M^{p-1} = I mod p; off-diagonal entry F_{p-1} = 0.
    Inert ((5/p)=-1): F_{p+1} = 0 mod p.
      Proof: phi^{p+1} = phi * phi^p = phi * psi = phi*psi = -1 (product of roots).
      Similarly psi^{p+1} = -1.
      F_{p+1} = (phi^{p+1} - psi^{p+1})/(phi-psi) = (-1 - (-1))/(phi-psi) = 0.
    Ramified (p=5): F_5 = 5 = 0 mod 5.

  C4 (Wall divisibility):  alpha(p) | p - (5/p).
    Immediate from C3: p-(5/p) is a zero of k |-> F_k mod p; alpha(p) is the
    smallest positive zero; every zero is a multiple of alpha(p) (Wall 1960).
    Thus (p - (5/p)) % alpha(p) = 0.

Primary sources:
  Wall, D.D. (1960) doi:10.2307/2309169 -- Fibonacci series mod m; alpha(p)|p-(5/p)
  Lucas, E. (1878) doi:10.2307/2369308 -- L_p = 1 mod p
  Legendre, A.M. (1798) Essai sur la Theorie des Nombres (Legendre symbol / QR)
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


def lucas_fast(n, m):
    """L_n mod m. L_n = F_{n+1} + F_{n-1}. Pure integer."""
    if n == 0:
        return 2 % m
    fn_m1, fn = _fib_pair(n - 1, m)      # (F_{n-1}, F_n)
    f_np1 = (fn + fn_m1) % m              # F_{n+1} = F_n + F_{n-1}
    return (f_np1 + fn_m1) % m            # L_n = F_{n+1} + F_{n-1}


def legendre5(p):
    """Legendre symbol (5/p): +1 (split), -1 (inert), 0 (ramified p=5).
    Pure integer: Euler criterion pow(5,(p-1)//2,p); p-1 ≡ -1 mod p."""
    if p == 5:
        return 0
    r = pow(5, (p - 1) // 2, p)
    return -1 if r == p - 1 else int(r)  # r in {1, p-1}; p-1 ≡ -1


def rank_of_apparition(p):
    """Smallest k>=1 with F_k = 0 mod p. Pure integer walk."""
    a, b = 0, 1
    k = 0
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0:
            return k


def sieve(n):
    """Primes up to n. Pure integer."""
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, math.isqrt(n) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]


# ── CHECK FUNCTIONS ───────────────────────────────────────────────────────────

def check_c1_fibonacci_legendre(n_bound=1000):
    """C1: F_p = (5/p) mod p for all primes p in [5, n_bound].
    Proof: F_p = (phi^p - psi^p)/(phi - psi).
    Split: phi^p=phi -> F_p = 1 = (5/p).
    Inert: phi^p=psi -> F_p = -1 = (5/p).
    Ramified (p=5): F_5=5=0=(5/5).
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        sym = legendre5(p)
        expected = sym % p      # -1 maps to p-1; 0 stays 0; +1 stays 1
        fp = fib_fast(p, p)
        if fp != expected:
            fails.append((p, sym, fp, expected))
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"F_p = (5/p) mod p: {tested - len(fails)}/{tested} "
            f"primes in [5,{n_bound}] PASS"
        ),
    }


def check_c2_lucas_fermat(n_bound=1000):
    """C2: L_p = 1 mod p for all primes p in [3, n_bound].
    Proof: L_p = phi^p + psi^p.
    Split: phi^p=phi, psi^p=psi -> phi+psi = 1 (sum of roots of x^2-x-1).
    Inert: phi^p=psi, psi^p=phi -> psi+phi = 1.
    Both splitting types give L_p = 1 mod p.
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 3:
            continue
        tested += 1
        lp = lucas_fast(p, p)
        if lp != 1:
            fails.append((p, lp))
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"L_p = 1 mod p: {tested - len(fails)}/{tested} "
            f"primes in [3,{n_bound}] PASS"
        ),
    }


def check_c3_wall_zero(n_bound=1000):
    """C3: F_{p - (5/p)} = 0 mod p for all primes p in [5, n_bound].
    Split: (5/p)=+1 -> F_{p-1}=0; from M^{p-1}=I (FLT on eigenvalues).
    Inert: (5/p)=-1 -> F_{p+1}=0; from phi^{p+1}=phi*psi=-1=psi^{p+1},
      so numerator = (-1)-(-1) = 0.
    Ramified: p=5 -> F_5=5=0 mod 5.
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        sym = legendre5(p)
        k = p - sym               # p-1 (split), p+1 (inert), p (ramified)
        fk = fib_fast(k, p)
        if fk != 0:
            fails.append((p, sym, k, fk))
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"F_{{p-(5/p)}} = 0 mod p: {tested - len(fails)}/{tested} "
            f"primes in [5,{n_bound}] PASS"
        ),
    }


def check_c4_wall_divisibility(n_bound=1000):
    """C4: alpha(p) | p - (5/p) for all primes p in [5, n_bound].
    From C3: p-(5/p) is a zero of k |-> F_k mod p. alpha(p) is the
    smallest positive zero. Wall (1960): zeros form a set closed under
    addition by alpha, so every zero is a multiple of alpha.
    Hence (p - (5/p)) % alpha(p) = 0.
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        sym = legendre5(p)
        target = p - sym          # p-1, p+1, or p
        alpha = rank_of_apparition(p)
        if target % alpha != 0:
            fails.append((p, sym, target, alpha, target % alpha))
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"alpha(p) | p-(5/p): {tested - len(fails)}/{tested} "
            f"primes in [5,{n_bound}] PASS"
        ),
    }


# ── OBSERVER LAYER: summary (no float state) ─────────────────────────────────

def _run_checks():
    return {
        "c1_fibonacci_legendre": check_c1_fibonacci_legendre(1000),
        "c2_lucas_fermat":       check_c2_lucas_fermat(1000),
        "c3_wall_zero":          check_c3_wall_zero(1000),
        "c4_wall_divisibility":  check_c4_wall_divisibility(1000),
    }


def main():
    import json
    results = _run_checks()
    all_ok = all(v["ok"] for v in results.values())
    print(json.dumps({
        "cert": "[428] QA Fibonacci Frobenius Congruences & Wall Divisibility",
        "all_checks_pass": all_ok,
        "checks": {k: {"ok": v["ok"], "desc": v["desc"]} for k, v in results.items()},
    }, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
