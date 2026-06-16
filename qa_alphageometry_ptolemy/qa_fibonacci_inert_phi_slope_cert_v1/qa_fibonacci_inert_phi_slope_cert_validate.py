"""
Cert [431]: QA Fibonacci Inert phi-Slope Formula mod p^2

Closes a gap explicitly deferred by cert [421]: the Hensel-lifted Binet
formula mod p^2 was only built for split primes (5/p)=+1, where phi_tilde,
psi_tilde live directly in Z/p^2 Z. This cert does the same construction for
inert primes (5/p)=-1, where phi and psi do not live in Z/pZ at all (cert
[424]) -- so the depth-2 lift must happen inside a ring extension, not Z/p^2 Z
itself.

Construction: for inert prime p, work in the depth-2 Galois ring

    R_p = (Z/p^2 Z)[phi] / (phi^2 - phi - 1)

-- the literal mod-p^2 lift of cert [424]'s F_{p^2} = (Z/pZ)[phi]/(phi^2-phi-1).
Elements are pairs (a,b) representing a + b*phi, with a,b in Z/p^2 Z, and the
same multiplication rule as [424]:

    (a+b*phi)(c+d*phi) = (ac+bd) + (ad+bc+bd)*phi      [using phi^2=phi+1]

Set psi_tilde = 1 - phi_tilde and s_tilde = phi_tilde - psi_tilde = 2*phi_tilde-1.
Then s_tilde^2 = 5*1 (a scalar), and since p != 5 is inert, 5 is a unit mod p^2,
so s_tilde is invertible in R_p with s_tilde^{-1} = (5^{-1} mod p^2) * s_tilde.
No separate Hensel lift of sqrt(5) is needed here -- unlike the split case
([421]), R_p is a free Z/p^2 Z-module of rank 2 by construction, so phi_tilde
and its conjugate already live there exactly.

Binet's formula then holds in R_p:

    F_n = (phi_tilde^n - psi_tilde^n) * s_tilde^{-1}   (mod p^2)

and the Fibonacci depth invariant delta(p) = F_{alpha(p)}/p mod p (the same
delta used in [429]/[430]) satisfies the Frobenius reformulation:

    delta(p) = 0   iff   phi_tilde^{alpha(p)} = psi_tilde^{alpha(p)}  in R_p

This is the inert-prime analog of [421]'s split-prime phi-slope reformulation,
completing the Galois/Frobenius interpretation of delta(p) -- and hence the
WSS criterion of [429] -- for ALL primes, not just split ones. Parallels how
[424] extended [423]'s split-only Frobenius-order statement to inert primes.

Primary sources:
  Hensel, K. (1897) "Ueber eine neue Begruendung der Theorie der algebraischen
    Zahlen" Jahresbericht der DMV 6 pp. 83-88  [Galois ring / Hensel lifting]
  Wall, D.D. (1960) "Fibonacci series modulo m" doi:10.2307/2309169
  Sun, Z.-H. & Sun, Z.-W. (1992) "Fibonacci and Lucas congruences and their
    applications" Acta Arithmetica 61 pp. 119-129 [p^2-level structure; WSS]

Four checks (QA layer = pure integer pair arithmetic in R_p = (Z/p^2Z)[phi]):
  C1: ring correctness -- phi^2=phi+1, psi^2=psi+1, phi*psi=-1, phi+psi=1
      (all mod p^2, in R_p) for 20 inert primes <= 200
  C2: Binet mod p^2 -- F_n = (phi^n-psi^n)*s^{-1} (mod p^2) in R_p, n=1..15,
      for 15 inert primes <= 150
  C3: delta(p) from R_p-Binet = delta(p) from direct integer recurrence,
      for all inert primes <= 500
  C4: WSS via Frobenius -- delta(p)=0 iff phi^{alpha(p)} = psi^{alpha(p)} in
      R_p, for all inert primes <= 500 (no WSS prime found, consistent with
      [429])

Theorem NT factorisation:
  QA layer (pure integer): R_p pair arithmetic, fib_fast, rank_of_apparition,
    sieve, pow() for ring-element exponentiation via repeated multiplication.
  Observer layer: none -- direct equality checks only.
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


def inert_primes(n):
    """All inert primes (5/p)=-1 in [7, n]: p % 5 in {2, 3}."""
    return [p for p in sieve(n) if p > 5 and p % 5 in (2, 3)]


def fib_fast(n, m):
    """F_n mod m via iterative fast doubling. O(log n). Pure integer."""
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


def rank_of_apparition(p):
    """alpha(p): smallest n>=1 with F_n == 0 mod p. Pure T-step walk."""
    a, b = 0, 1
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    raise ValueError(f"rank_of_apparition not found for p={p}")


# ---------------------------------------------------------------------------
# R_p = (Z/p^2 Z)[phi] / (phi^2 - phi - 1) arithmetic.
# Elements represented as pairs (a, b) for a + b*phi, a,b in Z/p^2 Z.
# phi^2 = phi + 1  =>  (0,1)*(0,1) = (1,1)
# ---------------------------------------------------------------------------

def ring_mul(x, y, p2):
    """(a+b*phi)(c+d*phi) = (ac+bd) + (ad+bc+bd)*phi   mod p2."""
    a, b = x
    c, d = y
    return ((a * c + b * d) % p2, (a * d + b * c + b * d) % p2)


def ring_add(x, y, p2):
    a, b = x
    c, d = y
    return ((a + c) % p2, (b + d) % p2)


def ring_sub(x, y, p2):
    a, b = x
    c, d = y
    return ((a - c) % p2, (b - d) % p2)


def ring_scale(k, x, p2):
    a, b = x
    return (k * a % p2, k * b % p2)


def ring_pow(x, n, p2):
    """Exponentiation by repeated squaring within R_p. Pure integer."""
    result = (1, 0)  # multiplicative identity
    base = x
    while n > 0:
        if n & 1:
            result = ring_mul(result, base, p2)
        base = ring_mul(base, base, p2)
        n >>= 1
    return result


def phi_psi_setup(p):
    """Return (phi_tilde, psi_tilde, s_tilde, s_inv, p2) in R_p for inert prime p."""
    p2 = p * p
    phi_t = (0, 1)            # phi itself
    psi_t = (1, p2 - 1)       # 1 - phi
    s_t = ring_sub(phi_t, psi_t, p2)   # phi - psi = 2*phi - 1
    five_inv = pow(5, -1, p2)
    s_inv = ring_scale(five_inv, s_t, p2)  # s^{-1} = 5^{-1} * s  (since s^2=5)
    return phi_t, psi_t, s_t, s_inv, p2


# ---------------------------------------------------------------------------
# Check C1: Ring correctness for 20 inert primes <= 200.
# ---------------------------------------------------------------------------
def check_c1_ring_correctness():
    primes = inert_primes(200)[:20]
    errors = []
    n_ok = 0

    for p in primes:
        phi_t, psi_t, s_t, s_inv, p2 = phi_psi_setup(p)

        phi_sq = ring_mul(phi_t, phi_t, p2)
        if phi_sq != ring_add(phi_t, (1, 0), p2):
            errors.append(f"p={p}: phi^2 != phi+1")
            continue

        psi_sq = ring_mul(psi_t, psi_t, p2)
        if psi_sq != ring_add(psi_t, (1, 0), p2):
            errors.append(f"p={p}: psi^2 != psi+1")
            continue

        phi_psi = ring_mul(phi_t, psi_t, p2)
        if phi_psi != (p2 - 1, 0):
            errors.append(f"p={p}: phi*psi != -1")
            continue

        if ring_add(phi_t, psi_t, p2) != (1, 0):
            errors.append(f"p={p}: phi+psi != 1")
            continue

        # s_tilde^2 == 5
        s_sq = ring_mul(s_t, s_t, p2)
        if s_sq != (5 % p2, 0):
            errors.append(f"p={p}: s_tilde^2 != 5")
            continue

        # s_tilde * s_inv == 1
        if ring_mul(s_t, s_inv, p2) != (1, 0):
            errors.append(f"p={p}: s_tilde * s_inv != 1")
            continue

        n_ok += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "primes_tested": primes,
        "n_verified": n_ok,
        "desc": (
            f"R_p ring correctness for {n_ok} inert primes: phi^2=phi+1, "
            f"psi^2=psi+1, phi*psi=-1, phi+psi=1, s_tilde^2=5, s_tilde invertible "
            f"(all mod p^2 in R_p)"
        ),
    }


# ---------------------------------------------------------------------------
# Check C2: Binet formula mod p^2 in R_p, n=1..15, 15 inert primes <= 150.
# ---------------------------------------------------------------------------
def check_c2_binet_formula():
    primes = inert_primes(150)[:15]
    errors = []
    n_cases = 0

    for p in primes:
        phi_t, psi_t, s_t, s_inv, p2 = phi_psi_setup(p)

        phi_pow, psi_pow = (1, 0), (1, 0)
        for n in range(1, 16):
            phi_pow = ring_mul(phi_pow, phi_t, p2)
            psi_pow = ring_mul(psi_pow, psi_t, p2)

            diff = ring_sub(phi_pow, psi_pow, p2)
            fib_binet_elt = ring_mul(diff, s_inv, p2)

            # Binet's formula must land in the "scalar" (b=0) part of R_p,
            # representing the actual integer F_n mod p^2.
            if fib_binet_elt[1] != 0:
                errors.append(
                    f"p={p} n={n}: binet result not scalar: {fib_binet_elt}"
                )
                continue

            fib_binet = fib_binet_elt[0]
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
            f"Binet mod p^2 in R_p: (phi^n-psi^n)*s_inv = F_n (mod p^2) "
            f"verified for {n_cases} cases (n=1..15, {len(primes)} inert primes)"
        ),
    }


# ---------------------------------------------------------------------------
# Check C3: delta(p) from R_p-Binet = delta(p) from direct recurrence,
# for all inert primes <= 500.
# ---------------------------------------------------------------------------
def check_c3_delta_from_binet():
    primes = inert_primes(500)
    errors = []
    n_ok = 0

    for p in primes:
        phi_t, psi_t, s_t, s_inv, p2 = phi_psi_setup(p)
        alpha = rank_of_apparition(p)

        phi_alpha = ring_pow(phi_t, alpha, p2)
        psi_alpha = ring_pow(psi_t, alpha, p2)
        diff = ring_sub(phi_alpha, psi_alpha, p2)
        fib_binet_elt = ring_mul(diff, s_inv, p2)

        if fib_binet_elt[1] != 0:
            errors.append(f"p={p}: binet F_alpha not scalar: {fib_binet_elt}")
            continue
        fib_binet = fib_binet_elt[0]

        if fib_binet % p != 0:
            errors.append(f"p={p}: binet F_alpha not divisible by p (={fib_binet})")
            continue
        delta_binet = fib_binet // p % p

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
        "n_inert_primes": len(primes),
        "n_verified": n_ok,
        "desc": (
            f"delta(p) from R_p-Binet = delta(p) from recurrence for all "
            f"{n_ok}/{len(primes)} inert primes p <= 500"
        ),
    }


# ---------------------------------------------------------------------------
# Check C4: WSS via Frobenius reformulation in R_p, for all inert primes <= 500.
# delta(p)=0 iff phi_tilde^{alpha(p)} == psi_tilde^{alpha(p)} in R_p.
# ---------------------------------------------------------------------------
def check_c4_wss_via_frobenius():
    primes = inert_primes(500)
    errors = []
    n_ok = 0

    for p in primes:
        phi_t, psi_t, s_t, s_inv, p2 = phi_psi_setup(p)
        alpha = rank_of_apparition(p)

        phi_alpha = ring_pow(phi_t, alpha, p2)
        psi_alpha = ring_pow(psi_t, alpha, p2)
        frobenius_fixed = (phi_alpha == psi_alpha)

        fib_alpha = fib_fast(alpha, p2)
        delta_zero = (fib_alpha % p2 == 0)

        if frobenius_fixed != delta_zero:
            errors.append(
                f"p={p}: frobenius_fixed={frobenius_fixed} != delta_zero={delta_zero}"
            )
        elif frobenius_fixed:
            errors.append(f"p={p}: WSS prime found among inert primes <= 500 — UNEXPECTED")
        else:
            n_ok += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_verified": n_ok,
        "desc": (
            f"WSS via Frobenius: delta(p)=0 iff phi_tilde^alpha == psi_tilde^alpha "
            f"(in R_p) for all {n_ok}/{len(primes)} inert primes <= 500; no WSS prime found"
        ),
    }


def main():
    c1 = check_c1_ring_correctness()
    c2 = check_c2_binet_formula()
    c3 = check_c3_delta_from_binet()
    c4 = check_c4_wss_via_frobenius()

    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_ring_correctness": c1,
            "C2_binet_formula_mod_p2": c2,
            "C3_delta_from_binet_eq_recurrence": c3,
            "C4_wss_via_frobenius": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
