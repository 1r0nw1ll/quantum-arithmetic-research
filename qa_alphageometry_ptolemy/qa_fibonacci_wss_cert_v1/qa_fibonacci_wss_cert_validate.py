"""
Cert [420]: Wall-Sun-Sun Depth Zero

CLAIM: A prime p is a Wall-Sun-Sun prime iff delta(p) = 0 (i.e., p^2 | F_{alpha(p)}).
This is equivalent to the classical condition p^2 | F_{p-(5/p)}.

The equivalence follows from the Lifting-the-Exponent (LTE) identity for Fibonacci sequences:
  v_p(F_{k*alpha}) = v_p(F_alpha) + v_p(k)   for p odd, p | F_alpha, p not-divides k

Setting m = p-(5/p) and alpha = alpha(p), write m = r*alpha where r = m/alpha. Then:
  v_p(F_{p-(5/p)}) = v_p(F_{r*alpha}) = v_p(F_alpha) + v_p(r)

Key bound: r < p for all odd primes p (proved by case analysis):
  - Split  (5/p)=+1: r = (p-1)/alpha(p) <= p-1 < p
  - Inert  (5/p)=-1: r = (p+1)/alpha(p) <= (p+1)/2 < p  [since alpha(p)>=2 for p>=3]
Therefore p does not divide r, so v_p(r) = 0, giving:
  v_p(F_{p-(5/p)}) = v_p(F_{alpha(p)})

Hence p^2 | F_{p-(5/p)} iff p^2 | F_{alpha(p)} iff delta(p) = 0.

No Wall-Sun-Sun prime is known. The exhaustive search record stands at p < 9.7*10^14
(McIntosh-Roettger 2007). The heuristic (delta(p) uniform on {0,...,p-1}) predicts
O(log log X) WSS primes up to X.

Primary sources:
  Wall, D.D. (1960) "Fibonacci series modulo m"
    American Mathematical Monthly 67(6) pp. 525-532, doi:10.2307/2309169
  Sun, Z.-H. & Sun, Z.-W. (1992) "Fibonacci and Lucas congruences and their applications"
    Acta Arithmetica 61 pp. 119-129  [LTE for Fibonacci; WSS definition]
  McIntosh, R.J. & Roettger, E.L. (2007) "A search for Fibonacci-Wieferich and Wolstenholme primes"
    Mathematics of Computation 76(260) pp. 2087-2094, doi:10.1090/S0025-5718-07-01955-2
"""

import json


# --- Pure-integer arithmetic (Theorem NT: no float, no continuous state) ---

def sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]


def fib_fast(n, m):
    """F_n mod m via iterative fast doubling. O(log n) pure-integer steps.

    Uses: F_{2k} = F_k*(2*F_{k+1} - F_k), F_{2k+1} = F_k^2 + F_{k+1}^2.
    Processes bits of n from MSB to LSB.
    """
    if n == 0:
        return 0
    a, b = 0, 1  # (F_0, F_1)
    for bit in bin(n)[2:]:
        c = a * (2 * b - a) % m
        d = (a * a + b * b) % m
        if bit == "1":
            a, b = d, (c + d) % m
        else:
            a, b = c, d
    return a


def kronecker_5(p):
    """(5/p): +1 split, -1 inert, 0 ramified. Pure integer."""
    r = p % 5
    if r == 0:
        return 0
    return 1 if r in {1, 4} else -1


def p_adic_val(n, p):
    """v_p(n): p-adic valuation. Returns 0 for n=0 sentinel (unused in zero context)."""
    if n == 0:
        return 999
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def rank_of_apparition(p):
    """alpha(p): smallest n>=1 with F_n == 0 mod p. Pure integer T-step walk."""
    a, b = 0, 1
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    raise ValueError(f"rank_of_apparition not found for p={p}")


# ---------------------------------------------------------------------------
# Check C1: LTE identity — v_p(F_{k*alpha}) = v_p(F_alpha) + v_p(k)
#
# Fibonacci numbers form a strong divisibility sequence: gcd(F_m, F_n) = F_{gcd(m,n)}.
# The Lifting-the-Exponent identity (Sun-Sun 1992) strengthens this to p-adic valuations.
# We verify numerically for 10 primes and multiple k values, including k divisible by p.
# ---------------------------------------------------------------------------
def check_c1_lte_identity():
    test_primes = [5, 11, 19, 29, 31, 41, 71, 89, 101, 149]
    errors = []
    verified = []

    for p in test_primes:
        alpha = rank_of_apparition(p)
        p4 = p ** 4
        f_alpha = fib_fast(alpha, p4)
        val_alpha = p_adic_val(f_alpha, p)

        # Test k not divisible by p (LTE: v_p(F_{k*alpha}) = val_alpha + 0)
        for k in [1, 2, 3, 5, 7]:
            if k % p == 0:
                continue
            f_kalpha = fib_fast(k * alpha, p4)
            val_got = p_adic_val(f_kalpha, p)
            expected = val_alpha  # v_p(k) = 0
            if val_got != expected:
                errors.append(
                    f"p={p} alpha={alpha} k={k} (p∤k): "
                    f"v_p(F_{{{k}*alpha}})={val_got} != {expected}"
                )
            else:
                verified.append(f"p={p} k={k} v_p={val_got}")

        # Test k=p (LTE: v_p(F_{p*alpha}) = val_alpha + 1)
        f_palpha = fib_fast(p * alpha, p4)
        val_palpha = p_adic_val(f_palpha, p)
        expected_p = val_alpha + 1
        if val_palpha != expected_p:
            errors.append(
                f"p={p} alpha={alpha} k=p: "
                f"v_p(F_{{p*alpha}})={val_palpha} != {expected_p}"
            )
        else:
            verified.append(f"p={p} k=p v_p={val_palpha} (val+1)")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_verified": len(verified),
        "desc": (
            f"LTE: v_p(F_{{k*alpha}})=v_p(F_alpha)+v_p(k); "
            f"verified {len(verified)} cases (p∤k and k=p) across {len(test_primes)} primes"
        ),
    }


# ---------------------------------------------------------------------------
# Check C2: Classical equivalence
# delta(p)=0 iff p^2 | F_{p-(5/p)} for all odd primes p != 5 up to 500.
# Both sides are False for every prime in range (no WSS prime found).
# ---------------------------------------------------------------------------
def check_c2_classical_equivalence():
    primes = [p for p in sieve(500) if p not in {2, 5}]
    errors = []
    n_checked = 0

    for p in primes:
        e = p - kronecker_5(p)   # p-(5/p)
        p2 = p * p

        # Classical condition: p^2 | F_{p-(5/p)}
        classical = fib_fast(e, p2) == 0

        # Delta condition: p^2 | F_{alpha(p)}
        alpha = rank_of_apparition(p)
        delta_zero = fib_fast(alpha, p2) == 0

        if classical != delta_zero:
            errors.append(
                f"p={p}: classical(p^2|F_{{{e}}})={classical} "
                f"!= delta_zero(p^2|F_{{{alpha}}})={delta_zero}"
            )
        elif classical:
            errors.append(f"p={p}: WSS prime found in [3,500] — UNEXPECTED")
        else:
            n_checked += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_verified": n_checked,
        "desc": (
            f"delta(p)=0 iff p^2|F_{{p-(5/p)}} for all {n_checked} odd primes "
            f"!= 5 in [3,500]; no WSS prime found in range"
        ),
    }


# ---------------------------------------------------------------------------
# Check C3: No WSS prime up to N — exhaustive computational check.
# Uses the classical formulation: F_{p-(5/p)} ≢ 0 (mod p^2) for all p <= N.
# Fast doubling gives O(log p) per prime; feasible for N=500,000.
# The known record (McIntosh-Roettger 2007) extends this to p < 9.7*10^14.
# ---------------------------------------------------------------------------
def check_c3_no_wss_prime(n_bound=500_000):
    primes = [p for p in sieve(n_bound) if p not in {2, 5}]
    errors = []

    for p in primes:
        e = p - kronecker_5(p)
        if fib_fast(e, p * p) == 0:
            errors.append(f"p={p}: WSS prime found! F_{{{e}}} == 0 mod {p}^2")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_primes_checked": len(primes),
        "n_bound": n_bound,
        "desc": (
            f"No WSS prime among {len(primes)} odd primes != 5 up to {n_bound}; "
            f"computational record: no WSS prime < 9.7*10^14 (McIntosh-Roettger 2007)"
        ),
    }


# ---------------------------------------------------------------------------
# Check C4: r = (p-(5/p))/alpha(p) is coprime to p for all odd primes p!=5 up to 500.
# This is the key analytic step: r < p (split: r=(p-1)/alpha<=p-1; inert: r=(p+1)/alpha<=(p+1)/2)
# so p cannot divide r, making v_p(r)=0 and completing the LTE simplification.
# ---------------------------------------------------------------------------
def check_c4_r_coprime():
    primes = [p for p in sieve(500) if p not in {2, 5}]
    errors = []
    n_verified = 0

    for p in primes:
        alpha = rank_of_apparition(p)
        e = p - kronecker_5(p)
        kron = kronecker_5(p)

        if e % alpha != 0:
            errors.append(f"p={p}: alpha={alpha} does not divide e={e} (should by cert [416])")
            continue

        r = e // alpha

        # Analytic bound: split r=(p-1)/alpha<=p-1<p; inert r=(p+1)/alpha<=(p+1)/2<p
        if kron == 1:  # split
            bound = p - 1
        else:  # inert: kron == -1
            bound = (p + 1) // 2  # integer floor; actual (p+1)/2 <= p-1 for p>=3

        if r > bound:
            errors.append(
                f"p={p} kron={kron}: r={r} > analytic bound {bound}"
            )
        elif r % p == 0:
            errors.append(
                f"p={p}: r={r} divisible by p (impossible since r<p but verify)"
            )
        else:
            n_verified += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_verified": n_verified,
        "desc": (
            f"r=(p-(5/p))/alpha(p) < p (hence coprime to p) for all "
            f"{n_verified} odd primes != 5 in [3,500]; "
            f"proves v_p(F_{{p-(5/p)}}) = v_p(F_{{alpha(p)}}) via LTE"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    c1 = check_c1_lte_identity()
    c2 = check_c2_classical_equivalence()
    c3 = check_c3_no_wss_prime(n_bound=500_000)
    c4 = check_c4_r_coprime()

    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_lte_identity": c1,
            "C2_classical_equivalence": c2,
            "C3_no_wss_prime_up_to_500k": c3,
            "C4_r_coprime_to_p": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
