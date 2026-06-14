"""
Cert [419]: QA Fibonacci Depth Decomposition

CLAIM: The Fibonacci depth invariant delta(p) = F_{alpha(p)} / p mod p decomposes
according to the parity of alpha(p):

  Even alpha:  delta(p) = F_{alpha/2} * (L_{alpha/2} / p) mod p
               [from F_{2n} = F_n * L_n; p always divides L_{alpha/2} for even alpha]

  Odd alpha:   delta(p) = (F_k^2 + F_{k-1}^2) / p mod p, k = (alpha+1)/2
               [from F_{2k-1} = F_k^2 + F_{k-1}^2]

Consequence: ALL Fibonacci primes p = F_m satisfy delta(p) = 1 (since F_{alpha(p)} = p exactly).
Non-Fibonacci delta=1 primes up to 2000: {41, 193, 1621}, each satisfying 2*alpha(p) = p-(5/p).

Primary sources:
  Lucas (1878) "Theorie des fonctions numeriques simplement periodiques"
    American Journal of Mathematics 1(2), pp. 184-240
    [Fibonacci doubling identity F_{2n}=F_n*L_n; odd identity F_{2n-1}=F_n^2+F_{n-1}^2]
  Wall (1960) "Fibonacci series modulo m"
    doi:10.2307/2309169 [rank of apparition alpha(p)]
  Lehmer (1930) "An extended theory of Lucas functions"
    doi:10.2307/1968235 [Lucas divisibility theory]
"""

import hashlib
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


def fib_mod(n, m):
    """F_n mod m via n QA T-steps. Pure integer."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % m
    return a


def lucas_mod(n, m):
    """L_n mod m. L_0=2, L_1=1, L_{n+1}=L_n+L_{n-1}."""
    if n == 0:
        return 2 % m
    a, b = 2, 1
    for _ in range(n - 1):
        a, b = b, (a + b) % m
    return b % m


def rank_of_apparition(p):
    """alpha(p): smallest n>=1 with F_n≡0 mod p. Pure integer T-step."""
    a, b = 0, 1
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    raise ValueError(f"rank not found for p={p}")


def kronecker_5(p):
    """(5/p): +1 split, -1 inert, 0 ramified. Integer only."""
    r = p % 5
    if r == 0:
        return 0
    return 1 if r in {1, 4} else -1


def delta_p(p):
    """delta(p) = F_{alpha(p)} / p mod p. Well-defined since p | F_{alpha(p)}."""
    alpha = rank_of_apparition(p)
    fa = fib_mod(alpha, p * p)
    return (fa // p) % p


def is_fibonacci_prime(p):
    """True iff p = F_k for some k >= 1."""
    a, b = 0, 1
    while b < p:
        a, b = b, a + b
    return b == p


# ---------------------------------------------------------------------------
# Check C1: Even-alpha decomposition
# For every prime p with even alpha(p) in [2, 500]:
#   - p | L_{alpha/2}  (Lucas divisibility corollary)
#   - delta(p) = F_{alpha/2} * (L_{alpha/2}/p) mod p
# ---------------------------------------------------------------------------
def check_c1_even_alpha_decomposition():
    primes = sieve(500)
    errors = []
    results = []
    even_count = 0
    for p in primes:
        alpha = rank_of_apparition(p)
        if alpha % 2 != 0:
            continue
        even_count += 1
        half = alpha // 2

        # Lucas divisibility: p | L_{alpha/2}
        l_half_mod_p = lucas_mod(half, p)
        if l_half_mod_p != 0:
            errors.append(
                f"p={p} alpha={alpha}: L_{half} ≢ 0 mod p (= {l_half_mod_p})"
            )
            continue

        # c = L_{alpha/2}/p mod p (extract via p^2 computation)
        l_half_mod_p2 = lucas_mod(half, p * p)
        if l_half_mod_p2 % p != 0:
            errors.append(f"p={p}: L_{half} mod p^2 not divisible by p")
            continue
        c = (l_half_mod_p2 // p) % p

        # Decomposition check
        f_half = fib_mod(half, p)
        delta_decomp = (f_half * c) % p

        fa = fib_mod(alpha, p * p)
        delta_direct = (fa // p) % p

        if delta_decomp != delta_direct:
            errors.append(
                f"p={p} alpha={alpha}: decomp={delta_decomp} != direct={delta_direct}"
            )
        else:
            results.append(
                f"p={p} α={alpha} δ={delta_direct} via F_{half}*{c}={delta_decomp}"
            )

    if even_count < 40:
        errors.append(f"only {even_count} even-alpha primes in [2,500] (expected >=40)")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_even_alpha": even_count,
        "n_verified": len(results),
        "desc": "delta(p) = F_{alpha/2} * (L_{alpha/2}/p) mod p for all 62 even-alpha primes in [2,500]",
    }


# ---------------------------------------------------------------------------
# Check C2: Odd-alpha decomposition
# For every prime p with odd alpha(p) in [2, 500]:
#   - F_{2k-1} = F_k^2 + F_{k-1}^2  (identity, k = (alpha+1)/2)
#   - delta(p) = (F_k^2 + F_{k-1}^2) / p mod p
# ---------------------------------------------------------------------------
def check_c2_odd_alpha_decomposition():
    primes = sieve(500)
    errors = []
    results = []
    odd_count = 0
    for p in primes:
        alpha = rank_of_apparition(p)
        if alpha % 2 != 1:
            continue
        odd_count += 1
        k = (alpha + 1) // 2  # alpha = 2k-1

        fk = fib_mod(k, p * p)
        fk1 = fib_mod(k - 1, p * p)
        sum_sq = (fk * fk + fk1 * fk1) % (p * p)

        fa = fib_mod(alpha, p * p)
        if sum_sq != fa % (p * p):
            errors.append(
                f"p={p} alpha={alpha} k={k}: F_k^2+F_{{k-1}}^2 mod p^2 "
                f"= {sum_sq} != F_alpha mod p^2 = {fa}"
            )
            continue

        if fa % p != 0:
            errors.append(f"p={p}: F_alpha not divisible by p")
            continue

        delta_decomp = (sum_sq // p) % p
        delta_direct = (fa // p) % p

        if delta_decomp != delta_direct:
            errors.append(f"p={p}: decomp={delta_decomp} != direct={delta_direct}")
        else:
            results.append(f"p={p} α={alpha} k={k} δ={delta_direct}")

    if odd_count < 15:
        errors.append(f"only {odd_count} odd-alpha primes in [2,500] (expected >=15)")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_odd_alpha": odd_count,
        "n_verified": len(results),
        "desc": "delta(p) = (F_k^2+F_{k-1}^2)/p mod p, k=(alpha+1)/2, for all 33 odd-alpha primes in [2,500]",
    }


# ---------------------------------------------------------------------------
# Check C3: Fibonacci prime subclass — all have delta=1
# ---------------------------------------------------------------------------
def check_c3_fibonacci_primes():
    fib_primes = [p for p in sieve(2000) if is_fibonacci_prime(p)]
    EXPECTED = [2, 3, 5, 13, 89, 233, 1597]
    errors = []
    results = []

    if fib_primes != EXPECTED:
        errors.append(
            f"Fibonacci primes up to 2000 = {fib_primes}, expected {EXPECTED}"
        )
        return {"ok": False, "errors": errors, "desc": "Fibonacci prime set mismatch"}

    for p in fib_primes:
        alpha = rank_of_apparition(p)
        fa_mod_p2 = fib_mod(alpha, p * p)
        # F_{alpha(p)} = p exactly: fa mod p^2 must equal p
        if fa_mod_p2 != p:
            errors.append(
                f"p={p} alpha={alpha}: F_alpha mod p^2 = {fa_mod_p2} (expected {p})"
            )
            continue
        d = (fa_mod_p2 // p) % p
        if d != 1:
            errors.append(f"p={p}: delta={d} (expected 1)")
        else:
            results.append(f"p={p}=F_{alpha} alpha={alpha} F_alpha=p delta=1")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "fibonacci_primes": fib_primes,
        "n_verified": len(results),
        "desc": "delta=1 for all 7 Fibonacci primes up to 2000 because F_{alpha(p)} = p exactly",
    }


# ---------------------------------------------------------------------------
# Check C4: Full delta census up to 2000
# Claim: delta=1 primes up to 2000 are {2,3,5,13,41,89,193,233,1597,1621}
# Non-Fibonacci delta=1 primes {41,193,1621} each satisfy 2*alpha(p) = p-(5/p)
# ---------------------------------------------------------------------------
def check_c4_delta_census():
    EXPECTED_DELTA1 = {2, 3, 5, 13, 41, 89, 193, 233, 1597, 1621}
    EXPECTED_NON_FIB = [41, 193, 1621]
    errors = []
    results = []

    primes = sieve(2000)
    found_delta1 = []
    for p in primes:
        d = delta_p(p)
        if d == 1:
            found_delta1.append(p)

    found_set = set(found_delta1)
    if found_set != EXPECTED_DELTA1:
        extra = sorted(found_set - EXPECTED_DELTA1)
        missing = sorted(EXPECTED_DELTA1 - found_set)
        errors.append(f"delta=1 mismatch: extra={extra}, missing={missing}")
        return {"ok": False, "errors": errors, "desc": "delta census mismatch"}

    non_fib = [p for p in found_delta1 if not is_fibonacci_prime(p)]
    if non_fib != EXPECTED_NON_FIB:
        errors.append(f"non-Fibonacci delta=1 = {non_fib}, expected {EXPECTED_NON_FIB}")

    for p in non_fib:
        alpha = rank_of_apparition(p)
        target = p - kronecker_5(p)
        if 2 * alpha != target:
            errors.append(
                f"p={p}: 2*alpha={2*alpha} != p-(5/p)={target}"
            )
        else:
            results.append(
                f"p={p} alpha={alpha} 2α={2*alpha}=p-(5/p)={target} delta=1"
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "delta1_primes": sorted(found_set),
        "non_fibonacci_delta1": non_fib,
        "n_max_rank_verified": len(results),
        "desc": (
            "delta=1 primes up to 2000 = {2,3,5,13,41,89,193,233,1597,1621}; "
            "non-Fibonacci delta=1 primes {41,193,1621} each satisfy 2*alpha=p-(5/p)"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    c1 = check_c1_even_alpha_decomposition()
    c2 = check_c2_odd_alpha_decomposition()
    c3 = check_c3_fibonacci_primes()
    c4 = check_c4_delta_census()

    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_even_alpha_decomposition": c1,
            "C2_odd_alpha_decomposition": c2,
            "C3_fibonacci_prime_subclass": c3,
            "C4_full_delta_census_up_to_2000": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
