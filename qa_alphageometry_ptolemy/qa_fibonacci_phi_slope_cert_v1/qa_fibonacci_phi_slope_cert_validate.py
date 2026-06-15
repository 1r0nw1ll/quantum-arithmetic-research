"""
Cert [421]: QA Fibonacci phi-Slope Formula

For a split prime p (i.e. (5/p)=+1), Binet's formula lifts from F_p to Z/p^2 Z via
Hensel's lemma. There exist unique phi_tilde, psi_tilde in Z/p^2 Z satisfying

    x^2 - x - 1 ≡ 0  (mod p^2)
    phi_tilde ≡ (1 + s5)/2  (mod p)   [phi_tilde reduces to the golden ratio mod p]

where s5 is the canonical square root of 5 mod p. The Binet formula holds mod p^2:

    F_n ≡ (phi_tilde^n - psi_tilde^n) * s_tilde^{-1}  (mod p^2)   for all n >= 0

where s_tilde is the Hensel lift of s5 to Z/p^2 Z.

Consequence: the Fibonacci depth invariant delta(p) = F_{alpha(p)}/p mod p satisfies

    delta(p) = (phi_tilde^{alpha(p)} - psi_tilde^{alpha(p)}) * s_tilde^{-1} / p  (mod p)

i.e. delta(p) is the normalised difference of the two lifted golden-ratio branches
at the first zero of the Fibonacci sequence mod p.

Wall-Sun-Sun reformulation (sharpening cert [420]):
    delta(p) = 0  iff  phi_tilde^{alpha(p)} ≡ psi_tilde^{alpha(p)}  (mod p^2)
i.e. the two Hensel lifts of phi are indistinguishable at depth p^2.

For inert primes (5/p)=-1, phi and psi live in F_{p^2} but not in F_p; Binet still holds but
over Z/p^2 Z[sqrt 5] (the unramified degree-2 extension). The cert covers the split
case where the formula is Z/p^2 Z-valued and directly computable.

Primary sources:
  Hensel, K. (1897) "Ueber eine neue Begruendung der Theorie der algebraischen Zahlen"
    Jahresbericht der DMV 6 pp. 83-88  [p-adic Hensel lifting / Newton iteration]
  Wall, D.D. (1960) "Fibonacci series modulo m" doi:10.2307/2309169
  Sun, Z.-H. & Sun, Z.-W. (1992) "Fibonacci and Lucas congruences and their applications"
    Acta Arithmetica 61 pp. 119-129  [p^2-level Fibonacci structure; WSS definition]
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
    """F_n mod m via iterative fast doubling. O(log n) pure-integer steps."""
    if n == 0:
        return 0
    a, b = 0, 1
    for bit in bin(n)[2:]:
        c = a * (2 * b - a) % m
        d = (a * a + b * b) % m
        if bit == "1":
            a, b = d, (c + d) % m
        else:
            a, b = c, d
    return a


def kronecker_5(p):
    r = p % 5
    if r == 0:
        return 0
    return 1 if r in {1, 4} else -1


def rank_of_apparition(p):
    """alpha(p): smallest n>=1 with F_n == 0 mod p. Pure T-step walk."""
    a, b = 0, 1
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    raise ValueError(f"rank_of_apparition not found for p={p}")


def split_primes(n):
    """All split primes (5/p)=+1 in [7, n]."""
    return [p for p in sieve(n) if p > 5 and p % 5 in {1, 4}]


def sqrt5_mod_p(p):
    """Canonical (smallest) square root of 5 in {1,...,p-1} for split prime p."""
    for s in range(1, p):
        if s * s % p == 5 % p:
            return s
    raise ValueError(f"5 is not a QR mod {p}")


def hensel_lift_sqrt5(s5, p):
    """Newton step: lift s5 (s5^2 ≡ 5 mod p) to s_tilde with s_tilde^2 ≡ 5 mod p^2.

    Newton formula: s_tilde = s5 + p * t  where t = -(s5^2-5)/p * (2*s5)^{-1}  mod p.
    """
    # (5 - s5^2) is divisible by p; divide to get t
    t = ((5 - s5 * s5) // p * pow(2 * s5, -1, p)) % p
    return s5 + p * t


def phi_psi_setup(p):
    """Return (phi_tilde, psi_tilde, s_tilde, p2) in Z/p^2 Z for split prime p.

    phi_tilde satisfies phi_tilde^2 ≡ phi_tilde + 1 (mod p^2).
    psi_tilde = 1 - phi_tilde satisfies psi_tilde^2 ≡ psi_tilde + 1 (mod p^2).
    phi_tilde + psi_tilde ≡ 1, phi_tilde * psi_tilde ≡ -1  (mod p^2).
    """
    s5 = sqrt5_mod_p(p)
    s_tilde = hensel_lift_sqrt5(s5, p)
    p2 = p * p
    inv2 = pow(2, -1, p2)
    phi_tilde = (1 + s_tilde) * inv2 % p2
    psi_tilde = (1 - s_tilde) * inv2 % p2   # Python % is non-negative
    return phi_tilde, psi_tilde, s_tilde, p2


# ---------------------------------------------------------------------------
# Check C1: Hensel correctness
# For 20 split primes: verify s_tilde^2 ≡ 5, phi_tilde^2 ≡ phi_tilde+1,
# psi_tilde^2 ≡ psi_tilde+1, phi_tilde*psi_tilde ≡ -1  (all mod p^2).
# ---------------------------------------------------------------------------
def check_c1_hensel_correctness():
    primes = split_primes(200)[:20]
    errors = []
    n_ok = 0

    for p in primes:
        phi_t, psi_t, s_t, p2 = phi_psi_setup(p)

        # s_tilde^2 ≡ 5
        if s_t * s_t % p2 != 5 % p2:
            errors.append(f"p={p}: s_tilde^2 != 5 mod p^2")
            continue

        # phi_tilde^2 ≡ phi_tilde + 1
        if phi_t * phi_t % p2 != (phi_t + 1) % p2:
            errors.append(f"p={p}: phi_tilde^2 != phi_tilde+1 mod p^2")
            continue

        # psi_tilde^2 ≡ psi_tilde + 1
        if psi_t * psi_t % p2 != (psi_t + 1) % p2:
            errors.append(f"p={p}: psi_tilde^2 != psi_tilde+1 mod p^2")
            continue

        # phi_tilde * psi_tilde ≡ -1
        if phi_t * psi_t % p2 != p2 - 1:
            errors.append(f"p={p}: phi_tilde*psi_tilde != -1 mod p^2")
            continue

        # phi_tilde + psi_tilde ≡ 1
        if (phi_t + psi_t) % p2 != 1:
            errors.append(f"p={p}: phi_tilde+psi_tilde != 1 mod p^2")
            continue

        n_ok += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "primes_tested": primes,
        "n_verified": n_ok,
        "desc": (
            f"Hensel lift correct for {n_ok} split primes: "
            f"s_tilde^2=5, phi_tilde^2=phi_tilde+1, psi_tilde^2=psi_tilde+1, "
            f"phi_tilde*psi_tilde=-1, phi_tilde+psi_tilde=1  (all mod p^2)"
        ),
    }


# ---------------------------------------------------------------------------
# Check C2: Binet formula mod p^2
# F_n ≡ (phi_tilde^n - psi_tilde^n) * s_tilde^{-1}  (mod p^2)
# for n in {1,...,15} and 15 split primes.
# This is non-trivial: the recurrence F_n = F_{n-1} + F_{n-2} is NOT
# a priori the same as the Binet product mod p^2.
# ---------------------------------------------------------------------------
def check_c2_binet_formula():
    primes = split_primes(150)[:15]
    errors = []
    n_cases = 0

    for p in primes:
        phi_t, psi_t, s_t, p2 = phi_psi_setup(p)
        s_inv = pow(s_t, -1, p2)

        phi_pow, psi_pow = 1, 1   # phi^0 = psi^0 = 1
        for n in range(1, 16):
            phi_pow = phi_pow * phi_t % p2
            psi_pow = psi_pow * psi_t % p2

            fib_binet = (phi_pow - psi_pow) * s_inv % p2
            fib_direct = fib_fast(n, p2)

            if fib_binet != fib_direct:
                errors.append(
                    f"p={p} n={n}: binet={fib_binet} != direct={fib_direct} (mod {p2})"
                )
            else:
                n_cases += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_cases": n_cases,
        "desc": (
            f"Binet mod p^2: (phi^n-psi^n)/s_tilde ≡ F_n (mod p^2) "
            f"verified for {n_cases} cases (n=1..15, {len(primes)} split primes)"
        ),
    }


# ---------------------------------------------------------------------------
# Check C3: delta(p) from Binet = delta(p) from Fibonacci recurrence
# For all split primes p <= 500: the two computation paths for delta(p) agree.
# ---------------------------------------------------------------------------
def check_c3_delta_from_binet():
    primes = split_primes(500)
    errors = []
    n_ok = 0

    for p in primes:
        phi_t, psi_t, s_t, p2 = phi_psi_setup(p)
        s_inv = pow(s_t, -1, p2)
        alpha = rank_of_apparition(p)

        # Binet path: compute F_alpha from phi_tilde^alpha, psi_tilde^alpha
        phi_alpha = pow(phi_t, alpha, p2)
        psi_alpha = pow(psi_t, alpha, p2)
        fib_binet = (phi_alpha - psi_alpha) * s_inv % p2

        if fib_binet % p != 0:
            errors.append(f"p={p}: binet F_alpha not divisible by p (F_alpha mod p^2={fib_binet})")
            continue

        delta_binet = fib_binet // p % p

        # Recurrence path: direct Fibonacci computation
        fib_rec = fib_fast(alpha, p2)
        if fib_rec % p != 0:
            errors.append(f"p={p}: recurrence F_alpha not divisible by p")
            continue

        delta_rec = fib_rec // p % p

        if delta_binet != delta_rec:
            errors.append(
                f"p={p} alpha={alpha}: delta_binet={delta_binet} != delta_rec={delta_rec}"
            )
        else:
            n_ok += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_split_primes": len(primes),
        "n_verified": n_ok,
        "desc": (
            f"delta(p) from Binet = delta(p) from recurrence for all "
            f"{n_ok}/{len(primes)} split primes p <= 500"
        ),
    }


# ---------------------------------------------------------------------------
# Check C4: WSS reformulation via phi-slope
# delta(p)=0 iff phi_tilde^{alpha(p)} ≡ psi_tilde^{alpha(p)} (mod p^2).
# Proof: delta=0 iff F_alpha ≡ 0 (mod p^2) iff (phi^alpha-psi^alpha)*s_inv ≡ 0 (mod p^2)
# iff phi^alpha ≡ psi^alpha (mod p^2) [since s_inv invertible mod p^2].
# Verified for all split primes p <= 500: both sides are False (no WSS prime found).
# ---------------------------------------------------------------------------
def check_c4_wss_via_phi_slope():
    primes = split_primes(500)
    errors = []
    n_ok = 0

    for p in primes:
        phi_t, psi_t, s_t, p2 = phi_psi_setup(p)
        alpha = rank_of_apparition(p)

        phi_alpha = pow(phi_t, alpha, p2)
        psi_alpha = pow(psi_t, alpha, p2)

        # phi-slope condition: delta=0 iff phi_alpha == psi_alpha mod p^2
        phi_slope_zero = (phi_alpha == psi_alpha)

        # Direct delta check
        fib_alpha = fib_fast(alpha, p2)
        delta_zero = (fib_alpha % p2 == 0)   # p^2 | F_alpha

        if phi_slope_zero != delta_zero:
            errors.append(
                f"p={p}: phi_slope_zero={phi_slope_zero} != delta_zero={delta_zero}"
            )
        elif phi_slope_zero:
            errors.append(f"p={p}: WSS prime found among split primes <= 500 — UNEXPECTED")
        else:
            n_ok += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_verified": n_ok,
        "desc": (
            f"WSS via phi-slope: delta(p)=0 iff phi_tilde^alpha == psi_tilde^alpha (mod p^2) "
            f"for all {n_ok}/{len(primes)} split primes <= 500; no WSS prime found"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    c1 = check_c1_hensel_correctness()
    c2 = check_c2_binet_formula()
    c3 = check_c3_delta_from_binet()
    c4 = check_c4_wss_via_phi_slope()

    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_hensel_correctness": c1,
            "C2_binet_formula_mod_p2": c2,
            "C3_delta_from_binet_eq_recurrence": c3,
            "C4_wss_via_phi_slope": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
