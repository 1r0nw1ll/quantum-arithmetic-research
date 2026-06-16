"""
QA Fermat Quotient Parallel Structure cert [430].

Corrects an earlier imprecise claim (closing note of cert [429]): Wall-Sun-Sun
(WSS) primes and Wieferich primes are NOT in an implication relationship.
There is no proof that a WSS prime forces 2^(p-1) = 1 (mod p^2), or vice versa.

What IS true is a structural PARALLEL: both conditions are depth-2 vanishings
of a Fermat-quotient-type residue, and Sun & Sun (1992) showed both are
independently NECESSARY (not equivalent, not sufficient on their own) for a
hypothetical failure of the first case of Fermat's Last Theorem -- a scenario
that cannot occur because FLT is proven (Wiles 1995).

    Structure         p-layer (universal)              p^2-layer (rare)
    Fibonacci         F_{p-(5/p)} = 0 (mod p)  [428]    F_{p-(5/p)} = 0 (mod p^2)  (WSS: none known)
    Powers of 2       2^(p-1) = 1 (mod p)      (FLT)    2^(p-1) = 1 (mod p^2)      (Wieferich: 1093, 3511)

Both p^2-layer conditions are equivalent to a Fermat-quotient-style residue
vanishing mod p:
    Fibonacci:    delta(p) := (F_{p-(5/p)} / p) mod p   = 0   <=>  WSS prime
    Powers of 2:  q_p(2)   := (2^(p-1) - 1) / p mod p   = 0   <=>  Wieferich prime

This cert verifies the powers-of-2 side (Fermat quotient q_p(2)) and confirms
the two known Wieferich primes are NOT WSS primes -- direct evidence the two
conditions are independent, not coupled.

Primary sources:
  Wieferich, A. (1909) "Zum letzten Fermat'schen Theorem" Journal fuer die
    reine und angewandte Mathematik 136, pp. 293-302. doi:10.1515/crll.1909.136.293
  Meissner, W. (1913) -- first known Wieferich prime 1093.
  Beeger, N.G.W.H. (1922) -- second known Wieferich prime 3511.
  Eisenstein, F.G.M. (1850) -- Fermat quotient q_p(a) = (a^(p-1)-1)/p (mod p).
  Sun, Z.H. and Sun, Z.W. (1992) "Fibonacci and Lucas numbers and powers of
    five" Acta Arithmetica 60, pp. 371-388. doi:10.4064/aa-60-4-371-388
    (parallel necessary conditions for FLT first-case failure).
  McIntosh, R.J. and Roettger, E.L. (2007) "A search for Fibonacci-Wieferich
    and Wolstenholme primes" Math. Comp. 76, pp. 2087-2094.
    doi:10.1090/S0025-5718-07-01955-2 (no WSS prime below 9.7e14).

Theorem NT factorisation:
  QA layer (pure integer): pow(2, p-1, p), pow(2, p-1, p*p), fermat_quotient_2,
    _fib_pair, fib_fast, rank_of_apparition, sieve -- all integer modular
    exponentiation / fast-doubling arithmetic.
  Observer layer: none -- no floats, no statistics, direct equality checks only.
"""

import json


def sieve(n):
    """Pure-integer sieve of Eratosthenes, primes in [2, n]."""
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]


def _fib_pair(n, m):
    """Fast doubling: returns (F_n mod m, F_{n+1} mod m). Pure integer, O(log n)."""
    if n == 0:
        return (0 % m, 1 % m)
    a, b = _fib_pair(n // 2, m)
    c = (a * ((2 * b - a) % m)) % m
    d = (a * a + b * b) % m
    if n % 2 == 0:
        return (c, d)
    else:
        return (d, (c + d) % m)


def fib_fast(n, m):
    """F_n mod m via fast doubling."""
    return _fib_pair(n, m)[0]


def rank_of_apparition(p):
    """Smallest k > 0 with F_k = 0 (mod p). Linear walk, pure integer."""
    a, b = 1 % p, 1 % p  # F1, F2
    k = 1
    if a == 0:
        return k
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0:
            return k


def fermat_quotient_2(p):
    """
    q_p(2) = (2^(p-1) - 1) / p (mod p), computed without ever materializing
    the full big integer 2^(p-1): pow(2, p-1, p*p) gives 2^(p-1) mod p^2
    directly, and (r-1) is guaranteed divisible by p when 2^(p-1) = 1 (mod p)
    (Fermat's little theorem, C1). Pure integer; S2-compliant.
    """
    r = pow(2, p - 1, p * p)
    if (r - 1) % p != 0:
        return None  # would indicate FLT failure -- never happens for prime p
    return (r - 1) // p


def check_c1_flt(n_bound=5000):
    """C1: 2^(p-1) = 1 (mod p) for all odd primes p in [3, n_bound]. FLT baseline."""
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 3:
            continue
        tested += 1
        r = pow(2, p - 1, p)
        if r != 1:
            fails.append((p, r))
    return {
        "name": "C1_flt_baseline",
        "tested": tested,
        "fails": fails,
        "pass": len(fails) == 0,
    }


def check_c2_fermat_quotient_well_defined(n_bound=5000):
    """C2: q_p(2) is a well-defined integer in [0, p-1] for all odd primes p."""
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 3:
            continue
        tested += 1
        q = fermat_quotient_2(p)
        if q is None or not (0 <= q < p):
            fails.append((p, q))
    return {
        "name": "C2_fermat_quotient_well_defined",
        "tested": tested,
        "fails": fails,
        "pass": len(fails) == 0,
    }


def check_c3_wieferich_primes(n_bound=5000):
    """C3: Wieferich primes (q_p(2) = 0) in [3, n_bound] are exactly {1093, 3511}."""
    found = []
    tested = 0
    for p in sieve(n_bound):
        if p < 3:
            continue
        tested += 1
        q = fermat_quotient_2(p)
        if q == 0:
            found.append(p)
    expected = [1093, 3511]
    return {
        "name": "C3_wieferich_primes_exact",
        "tested": tested,
        "found": found,
        "expected": expected,
        "pass": found == expected,
    }


def check_c4_wieferich_not_wss(wieferich_primes=(1093, 3511)):
    """
    C4: Independence check. The two known Wieferich primes are NOT
    Wall-Sun-Sun: F_{alpha(p)} != 0 (mod p^2) for both. Direct evidence the
    p^2-layer vanishing of q_p(2) and the p^2-layer vanishing of the Fibonacci
    Fermat quotient delta(p) are uncoupled -- one occurring says nothing about
    the other.
    """
    results = []
    fails = []
    for p in wieferich_primes:
        alpha = rank_of_apparition(p)
        fa_p2 = fib_fast(alpha, p * p)
        is_wss = (fa_p2 == 0)
        results.append({"p": p, "alpha": alpha, "F_alpha_mod_p2": fa_p2, "is_wss": is_wss})
        if is_wss:
            fails.append((p, alpha, "WOULD BE a WSS prime -- sensational, re-verify"))
    return {
        "name": "C4_wieferich_independent_of_wss",
        "results": results,
        "fails": fails,
        "pass": len(fails) == 0,
    }


def run_all_checks():
    n_bound = 5000
    c1 = check_c1_flt(n_bound)
    c2 = check_c2_fermat_quotient_well_defined(n_bound)
    c3 = check_c3_wieferich_primes(n_bound)
    c4 = check_c4_wieferich_not_wss()

    all_pass = c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"]

    return {
        "cert": "[430] QA Fermat Quotient Parallel Structure",
        "all_checks_pass": all_pass,
        "checks": {
            "c1_flt_baseline": {
                "ok": c1["pass"],
                "desc": f"2^(p-1)=1 mod p: {c1['tested']}/{c1['tested']} odd primes in [3,{n_bound}] PASS",
            },
            "c2_fermat_quotient_well_defined": {
                "ok": c2["pass"],
                "desc": f"q_p(2) well-defined in [0,p-1]: {c2['tested']}/{c2['tested']} primes in [3,{n_bound}] PASS",
            },
            "c3_wieferich_primes_exact": {
                "ok": c3["pass"],
                "desc": f"Wieferich primes in [3,{n_bound}] = {c3['found']} (expected {c3['expected']})",
            },
            "c4_wieferich_independent_of_wss": {
                "ok": c4["pass"],
                "desc": "1093 and 3511 are NOT Wall-Sun-Sun (F_alpha(p) != 0 mod p^2 for both): 2/2 PASS",
            },
        },
    }


def main():
    result = run_all_checks()
    print(json.dumps(result, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
