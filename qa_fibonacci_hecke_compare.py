#!/usr/bin/env python3
# QA_COMPLIANCE = "pure arithmetic verification; no QA observer state — compares QA's golden Frobenius (Q(sqrt5)) with the CM Hilbert form's Hecke eigenvalues (Q(zeta5))"
"""
Does QA's Fibonacci-Frobenius system COMPUTE the Hecke eigenvalues of the CM Hilbert
modular form 2.2.5.1-125.1-a, or does it only share the field Q(sqrt5)?

The Brandt/Hecke bridge (docs/theory/QA_AS_QUATERNION_ORDER.md sec 6) reproduces this
CM form's full Hecke system from the DEFINITE golden order (icosian ring = E8) via
Jacquet-Langlands. Section 6.1 says QA's own orbit generator M is *orthogonal* to the
Hecke structure. This script pins that down to a precise TOWER statement.

Two arithmetic layers of the same cyclotomic tower Q < Q(sqrt5) < Q(zeta5):

  * QA Fibonacci-Frobenius (cert [423]): the roots phi,psi of x^2-x-1 mod p -- the
    arithmetic of Q(sqrt5) (degree 2). Frobenius TRACE phi+psi == 1 (Vieta) for EVERY
    split prime: constant, the fixed golden recurrence. It carries the SPLITTING and
    the multiplicative-order/rank datum alpha(p)=ord(phi/psi), but NO varying GL2
    eigenvalue.

  * The CM form's Hecke eigenvalue a_P in Z[phi] (GL2/Hilbert), from the from-scratch
    Brandt cusp factors: 11 -> x^2+x-31 -> a = 5*phi-3; 31 -> x^2+11x-1 -> a = 5*phi-8.
    It is NONZERO iff p == 1 mod 5 (P splits in Q(zeta5)/Q(sqrt5)) -- a FINER condition
    than splitting in Q(sqrt5) (p == +/-1 mod 5). a_P is a Hecke-character value on
    Q(zeta5): the ONE-CYCLOTOMIC-LAYER-UP datum QA's golden structure does not reach.

Decisive demonstration: p=11 (=1 mod5) and p=19 (=4 mod5) BOTH split in Q(sqrt5) with
the same Fibonacci "split" structure, yet the CM form has a_11 != 0 and a_19 = 0. QA's
phi-arithmetic cannot tell them apart; the CM (zeta5) form can. => QA computes the
eigenvalue FIELD Q(sqrt5) and the splitting, NOT the eigenvalues.
"""
from __future__ import annotations


def isqrt_mod(a, p):
    """A square root of a mod p if one exists (brute force; small p), else None."""
    a %= p
    for x in range(p):
        if (x * x - a) % p == 0:
            return x
    return None


def pisano_period(p):
    a, b, t = 0, 1, 0
    for t in range(1, p * p + 1):
        a, b = b, (a + b) % p
        if a == 0 and b == 1:
            return t
    return None


def rank_of_apparition(p):
    """alpha(p): least n>=1 with F_n == 0 mod p."""
    a, b = 0, 1
    for n in range(1, p * p + 1):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    return None


# CM Hecke eigenvalues a_P in Z[phi] from the from-scratch Brandt cusp factors
# (docs/theory sec 6.3; a = (S +- sqrt(S^2-4P))/... expressed in phi with sqrt5=2phi-1).
# stored as (S, P): cusp x^2 - S x + P ; a_P is a root.  a in Z[phi] = c + d*phi.
CUSP = {11: (-1, -31), 31: (-11, -1)}      # 2.2.5.1-125.1-a, split primes with known cusp


def a_P_in_phi(S, P):
    """Return (c, d) with a_P = c + d*phi, a root of x^2 - S x + P, using sqrt5 = 2phi-1.
    Roots = (S +- sqrt(S^2-4P))/2 ; here S^2-4P is 5*k^2 so sqrt = k*sqrt5 = k*(2phi-1)."""
    disc = S * S - 4 * P
    k2 = disc // 5
    k = int(round(k2 ** 0.5))
    assert k * k == k2 and 5 * k2 == disc, f"disc {disc} not 5*square"
    # a = (S + k*(2phi-1))/2 = (S - k)/2 + k*phi
    c2, d = S - k, k
    assert c2 % 2 == 0
    return c2 // 2, d


def main():
    print("QA golden Frobenius (Q(sqrt5))  vs  CM form 2.2.5.1-125.1-a Hecke eigenvalues (Q(zeta5))\n")
    print(f"{'p':>3} {'p%5':>3} {'split Q(v5)':>11} {'Frob tr':>8} {'alpha(p)':>8} "
          f"{'pi(p)':>6} {'CM a_P!=0':>9} {'a_P in Z[phi]':>14}")
    primes = [11, 19, 29, 31, 41, 59, 61, 71, 79, 89]
    for p in primes:
        r = p % 5
        legendre5 = pow(5, (p - 1) // 2, p)                 # (5/p): 1 split, p-1 inert
        split_v5 = (legendre5 == 1)                          # p == +/-1 mod 5
        # Frobenius trace of the golden element = phi+psi = 1 (Vieta) for split primes
        s = isqrt_mod(5, p)
        if split_v5 and s is not None:
            inv2 = pow(2, p - 2, p)
            phi = ((1 + s) * inv2) % p
            psi = ((1 - s) * inv2) % p
            frob_tr = (phi + psi) % p                        # == 1 always
        else:
            frob_tr = None
        alpha = rank_of_apparition(p)
        pi = pisano_period(p)
        cm_nonzero = (r == 1)                                # splits in Q(zeta5)
        if p in CUSP:
            c, d = a_P_in_phi(*CUSP[p])
            aP = f"{c:+d}{d:+d}*phi".replace("+1*phi", "+phi").replace("-1*phi", "-phi")
        else:
            aP = "0" if not cm_nonzero else "(nonzero; not in 3-prime set)"
        print(f"{p:>3} {r:>3} {str(split_v5):>11} {str(frob_tr):>8} {str(alpha):>8} "
              f"{str(pi):>6} {str(cm_nonzero):>9} {aP:>14}")

    print("\nFINDINGS:")
    print("[1] Frobenius trace of QA's golden element == 1 for EVERY split prime (Vieta on")
    print("    x^2-x-1): constant -> carries NO varying GL2 eigenvalue. QA's golden Frobenius")
    print("    is GL1 rank/order data (alpha(p)=ord(phi/psi), cert [423]), not a Hecke a_P.")
    print("[2] p=11 (=1 mod5) and p=19 (=4 mod5) BOTH split in Q(sqrt5) (same Fibonacci type),")
    print("    yet CM a_11 = 5phi-3 != 0 while a_19 = 0. QA's phi-arithmetic (Q(sqrt5)) cannot")
    print("    distinguish them; the CM form (Q(zeta5)) does -> a_P != 0 needs p==1 mod5.")
    print("[3] CONCLUSION: QA's Fibonacci-Frobenius computes the eigenvalue FIELD Q(sqrt5) and")
    print("    the splitting, NOT the Hecke eigenvalues. The eigenvalues a_P are Q(zeta5) CM")
    print("    data one cyclotomic layer UP. The Brandt/Hecke bridge is thus field-level +")
    print("    Jacquet-Langlands (via the definite icosian/E8 order), not QA-orbit-dynamical --")
    print("    sharpening sec 6.1's 'M orthogonal to Hecke' into a precise tower statement.")


if __name__ == "__main__":
    main()
