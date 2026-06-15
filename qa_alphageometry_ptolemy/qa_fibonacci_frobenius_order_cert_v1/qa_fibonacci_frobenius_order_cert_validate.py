"""
Cert [423]: QA Fibonacci Frobenius Order Identity

For a split prime p ((5/p)=+1), the rank of apparition alpha(p) equals the
multiplicative order of the Frobenius eigenvalue ratio phi_tilde/psi_tilde
in (Z/pZ)×:

    alpha(p) = ord_{(Z/pZ)×}(phi_tilde / psi_tilde)

This is the GL_1 Frobenius statement for Q(sqrt 5)/Q.

Background:
  For a split prime p, the polynomial x^2-x-1 factors as (x-phi_tilde)(x-psi_tilde)
  over F_p, where phi_tilde = (1+s5)*inv(2,p) mod p (canonical golden ratio,
  cert [421]) and psi_tilde = 1-phi_tilde (the conjugate).

  Key algebraic identities (all mod p):
    phi_tilde + psi_tilde = 1             (sum of roots = -(-1)/1 = 1)
    phi_tilde * psi_tilde = -1            (product of roots = -1/1 = -1)
    psi_tilde = -phi_tilde^{-1}           (since phi_tilde * psi_tilde = -1)
    phi_tilde / psi_tilde = -phi_tilde^2  (since psi_tilde^{-1} = -phi_tilde)

  The Frobenius eigenvalue ratio rho = phi_tilde/psi_tilde = -phi_tilde^2
  is an element of (Z/pZ)×. Its multiplicative order is alpha(p) because:

    F_n ≡ 0 (mod p)
    iff (phi_tilde^n - psi_tilde^n) * s5^{-1} ≡ 0 (mod p)   [Binet, cert [421]]
    iff phi_tilde^n ≡ psi_tilde^n (mod p)                    [s5 is a unit]
    iff (phi_tilde/psi_tilde)^n ≡ 1 (mod p)                  [divide both sides]
    iff rho^n ≡ 1 (mod p)

  So the smallest n >= 1 with F_n ≡ 0 (mod p) — i.e. alpha(p) — is exactly
  ord_{(Z/pZ)×}(rho) = ord_{(Z/pZ)×}(phi_tilde/psi_tilde).

Langlands interpretation:
  In the GL_1/Q(sqrt 5) picture: the Hecke eigenvalues at a prime P above p
  are phi_tilde (at P) and psi_tilde (at P-bar). The Frobenius at P is the
  identity (p splits), so Frob_P acts trivially on {phi_tilde, psi_tilde}.
  The "relative Frobenius" rho = phi_tilde/psi_tilde encodes how the two
  eigenvalues differ; its order in GL_1(F_p) is alpha(p). This is the GL_1
  rung of the Langlands ladder for Q(sqrt 5)/Q connecting Fibonacci dynamics
  (QA T-step orbit) to multiplicative group structure in F_p.

Primary sources:
  Wall, D.D. (1960) "Fibonacci series modulo m" doi:10.2307/2309169
    [rank of apparition alpha(p); p divides F_{p-(5/p)}]
  Lagrange, J.-L. (1771) "Demonstration d'un theoreme d'arithmetique"
    Nouveaux Memoires de l'Academie de Berlin pp. 125-173
    [Lagrange's theorem: ord(g) | #G for g in a finite group G]
  Hecke, E. (1920) "Eine neue Art von Zetafunktionen..." MZ 6 pp. 11-51
    [Hecke L-functions; GL_1 Frobenius equidistribution over Q(sqrt 5)]

Four checks (QA layer = pure integer; observer layer = float for statistics only):
  C1: phi_tilde * psi_tilde = -1 (mod p) for all 45 split primes <= 500
  C2: phi_tilde / psi_tilde = -phi_tilde^2 (mod p) for all 45 split primes <= 500
  C3: ord_{(Z/pZ)×}(phi_tilde/psi_tilde) = alpha(p) for all 45 split primes <= 500
  C4: Primitive primes (alpha(p)=p-1) among split primes <= 10000: fraction in (0.25, 0.50)
      [Artin's conjecture: density -> A = 0.3739... (Artin constant) unconditionally
       for primitive roots; empirical check validates the phi-slope element behaves as
       a "random" element of (Z/pZ)× in this range]
"""

import json


# ============================================================
# QA LAYER: pure-integer arithmetic
# ============================================================

def sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]


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


def sqrt5_mod_p(p):
    """Canonical (smallest) square root of 5 in {1,...,p-1} for split prime p."""
    for s in range(1, p):
        if s * s % p == 5 % p:
            return s
    raise ValueError(f"5 is not a QR mod {p}")


def golden_ratios_mod_p(p):
    """Return (phi_tilde, psi_tilde) in {1,...,p-1} for split prime p.

    phi_tilde = (1 + s5) * inv(2,p) mod p (canonical; s5 = sqrt(5) mod p)
    psi_tilde = 1 - phi_tilde mod p
    """
    s5 = sqrt5_mod_p(p)
    inv2 = pow(2, -1, p)
    phi = (1 + s5) * inv2 % p
    psi = (1 - phi) % p
    return phi, psi


def prime_factors(n):
    """Distinct prime factors of n."""
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
    """Order of g in (Z/pZ)×. Pure integer; uses Lagrange-based order reduction."""
    g = g % p
    if g == 0:
        raise ValueError("g=0 has no multiplicative order")
    n = p - 1
    factors = prime_factors(n)
    result = n
    for q in factors:
        while result % q == 0 and pow(g, result // q, p) == 1:
            result //= q
    return result


def split_primes_upto(n):
    """Primes p <= n with (5/p)=+1, i.e. p%5 in {1,4}. Skips p=5."""
    return [p for p in sieve(n) if p > 5 and p % 5 in {1, 4}]


# ============================================================
# CHECKS
# ============================================================

def check_c1_product(primes):
    """C1: phi_tilde * psi_tilde = -1 mod p for all split primes in list."""
    fails = []
    for p in primes:
        phi, psi = golden_ratios_mod_p(p)
        if phi * psi % p != p - 1:   # -1 mod p = p-1
            fails.append(p)
    n = len(primes)
    return {
        "ok": len(fails) == 0,
        "n": n,
        "fail_primes": fails,
        "desc": f"phi*psi=-1 mod p for {n-len(fails)}/{n} split primes <= 500",
    }


def check_c2_ratio(primes):
    """C2: phi_tilde/psi_tilde = -phi_tilde^2 mod p for all split primes in list.

    Since psi_tilde = -phi_tilde^{-1} (from phi*psi=-1),
    phi/psi = phi/(-phi^{-1}) = -phi^2 = -(phi+1) using phi^2=phi+1.
    """
    fails = []
    for p in primes:
        phi, psi = golden_ratios_mod_p(p)
        rho = phi * pow(psi, -1, p) % p         # phi/psi mod p
        neg_phi2 = (-(phi * phi % p)) % p        # -phi^2 mod p
        if rho != neg_phi2:
            fails.append(p)
    n = len(primes)
    return {
        "ok": len(fails) == 0,
        "n": n,
        "fail_primes": fails,
        "desc": f"phi/psi=-phi^2 mod p for {n-len(fails)}/{n} split primes <= 500",
    }


def check_c3_order_identity(primes):
    """C3: ord_{(Z/pZ)×}(phi_tilde/psi_tilde) = alpha(p) for all split primes.

    This is the GL_1 Frobenius Order Identity: the multiplicative order of the
    Frobenius eigenvalue ratio equals the Fibonacci rank of apparition.
    """
    fails = []
    n = len(primes)
    for p in primes:
        phi, psi = golden_ratios_mod_p(p)
        rho = phi * pow(psi, -1, p) % p    # phi/psi = Frobenius eigenvalue ratio
        ord_rho = mult_order(rho, p)
        alpha = rank_of_apparition(p)
        if ord_rho != alpha:
            fails.append((p, ord_rho, alpha))
    return {
        "ok": len(fails) == 0,
        "n": n,
        "fail_triples": fails,
        "desc": f"ord(phi/psi)=alpha(p) for {n-len(fails)}/{n} split primes <= 500",
    }


def check_c4_primitive_primes(primes):
    """C4: fraction of split primes where alpha(p)=p-1 (phi/psi is primitive root).

    Artin's conjecture: density of primitive primes for a fixed non-square g converges
    to A ~ 0.3739... (Artin constant). This check verifies the fraction lies in (0.25,0.50)
    for split primes up to 10000 — consistent with Artin's prediction.
    """
    primitive = 0
    total = len(primes)
    for p in primes:
        phi, psi = golden_ratios_mod_p(p)
        rho = phi * pow(psi, -1, p) % p
        if mult_order(rho, p) == p - 1:
            primitive += 1
    frac = primitive / total
    lo, hi = 0.25, 0.50

    return {
        "ok": lo < frac < hi,
        "n": total,
        "primitive_count": primitive,
        "fraction": round(frac, 5),
        "expected_artin": 0.3739,
        "interval": [lo, hi],
        "desc": (
            f"primitive fraction={frac:.4f} in ({lo},{hi}); "
            f"Artin prediction ~0.3739; {primitive}/{total} split primes"
        ),
    }


def main():
    split_500 = split_primes_upto(500)
    split_10k = split_primes_upto(10_000)

    c1 = check_c1_product(split_500)
    c2 = check_c2_ratio(split_500)
    c3 = check_c3_order_identity(split_500)
    c4 = check_c4_primitive_primes(split_10k)

    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "n_split_primes_500": len(split_500),
        "n_split_primes_10k": len(split_10k),
        "checks": {
            "C1_phi_psi_product": c1,
            "C2_ratio_identity": c2,
            "C3_frobenius_order_identity": c3,
            "C4_primitive_primes": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
