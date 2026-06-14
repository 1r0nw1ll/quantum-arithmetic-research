# Primary source: Wall, D.D. (1960) "Fibonacci series modulo m" doi:10.2307/2309169
# Lucas, E. (1878) "Theorie des fonctions numeriques" American Journal of Mathematics 1(2)
# Lehmer, D.H. (1930) "An extended theory of Lucas' functions" doi:10.2307/1968235
# Cert [416]: QA Rank of Apparition = Unified Prime Splitting Formula
"""
Cert [416] — Rank of Apparition: unified prime splitting formula alpha(p) | p - (5/p).

Cert [415] gave TWO separate conditions:
  split (p%5 in {1,4}): pi(p) | p-1
  inert (p%5 in {2,3}): pi(p) | 2(p+1) and pi(p) does not divide p-1

Cert [416] gives ONE formula via the rank of apparition alpha(p):
  alpha(p) | p - (5/p)

where (5/p) = Kronecker symbol in {+1, -1, 0}:
  split (p%5 in {1,4}):  (5/p) = +1  =>  alpha(p) | p-1
  inert (p%5 in {2,3}):  (5/p) = -1  =>  alpha(p) | p+1
  ramified p=5:           (5/p) =  0  =>  alpha(p) | p   (=5, verified: alpha(5)=5)

alpha(p) = rank of apparition = smallest positive n with F_n ≡ 0 (mod p).
  QA T-orbit reading: alpha(p) is the first step at which the T-orbit of (F_0,F_1)=(0,1)
  under T(b,e)=(e,b+e) returns to a state (F_{n-1},F_n) with F_n ≡ 0 (mod p).

Additional structure (C2, C3):
  C2: alpha(p) | pi(p)    (rank divides Pisano period — always)
  C3: pi(p) / alpha(p) in {1, 2, 4}   (entry quotient; 1 most common, 4 rare)

The Kronecker symbol (5/p) is computed entirely by integer arithmetic:
  (5/p) = 0  if p == 5
  (5/p) = +1 if p%5 in {1,4}
  (5/p) = -1 if p%5 in {2,3}
(This follows from quadratic reciprocity + properties of the Legendre symbol mod 5.)

CHAIN: [133] T-step sign-flip -> [414] norm form -> [415] Pisano period two formulas
    -> [416] rank of apparition one formula.
"""

import json

SPLIT_PRIMES = [11, 31, 41, 61, 71, 101, 131, 151, 181, 191, 211, 241,
                251, 271, 281, 311, 331, 401, 421, 431, 461, 491]
INERT_PRIMES = [2, 3, 7, 13, 17, 23, 37, 43, 47, 53, 67, 73, 83, 97,
                103, 107, 113, 127, 137, 167, 173, 193]
RAM_PRIME = 5


def kronecker_5(p):
    """(5/p) Kronecker symbol: +1 split, -1 inert, 0 ramified. Integer only."""
    r = p % 5
    if r == 0:
        return 0
    if r == 1 or r == 4:
        return 1
    return -1


def rank_of_apparition_v2(p):
    """Clean version: smallest n >= 1 with F_n ≡ 0 mod p."""
    # F_0=0 (trivially), F_1=1, F_2=1, F_3=2, ...
    # We want smallest n >= 1 with F_n = 0 mod p.
    # F_0 = 0, so start looking from n=1.
    # alpha(p) is always >= 1 and for p>5 alpha(p) >= 5 (Wall 1960).
    a, b = 0, 1  # (F_0, F_1)
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        # Now a = F_n, b = F_{n+1}
        if a == 0:
            return n
    raise ValueError(f"rank_of_apparition not found for p={p} in 4p steps")


def pisano_period(p):
    """Pisano period pi(p): period of (F_n mod p) sequence."""
    a, b = 0, 1
    for k in range(1, 6 * p + 2):
        a, b = b, (a + b) % p
        if a == 0 and b == 1:
            return k
    raise ValueError(f"pi not found for p={p}")


def check_c1_unified_formula():
    """C1: alpha(p) | p - (5/p) for all split, inert, and ramified primes tested."""
    errors = []
    witnesses = {}
    all_primes = SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME]
    for p in sorted(all_primes):
        chi = kronecker_5(p)
        target = p - chi        # p-1 for split, p+1 for inert, p for ramified
        alpha = rank_of_apparition_v2(p)
        divides = target % alpha == 0
        witnesses[p] = {
            "chi_5_p": chi,
            "target_p_minus_chi": target,
            "alpha": alpha,
            "alpha_divides_target": divides,
        }
        if not divides:
            errors.append({"p": p, "chi": chi, "target": target, "alpha": alpha})
    return {
        "ok": len(errors) == 0,
        "count": len(all_primes),
        "errors": errors,
        "witnesses": witnesses,
        "desc": f"Unified formula alpha(p) | p-(5/p): verified for {len(all_primes)} primes (split+inert+ramified)",
    }


def check_c2_alpha_divides_pi():
    """C2: alpha(p) | pi(p) for all tested primes."""
    errors = []
    witnesses = {}
    all_primes = SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME]
    for p in sorted(all_primes):
        alpha = rank_of_apparition_v2(p)
        pi = pisano_period(p)
        divides = pi % alpha == 0
        witnesses[p] = {"alpha": alpha, "pi": pi, "alpha_divides_pi": divides}
        if not divides:
            errors.append({"p": p, "alpha": alpha, "pi": pi})
    return {
        "ok": len(errors) == 0,
        "count": len(all_primes),
        "errors": errors,
        "witnesses": witnesses,
        "desc": "alpha(p) | pi(p) for all tested primes (rank of apparition divides Pisano period)",
    }


def check_c3_entry_quotient():
    """C3: pi(p) / alpha(p) in {1, 2, 4} for all tested primes."""
    errors = []
    quotients = {}
    all_primes = SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME]
    for p in sorted(all_primes):
        alpha = rank_of_apparition_v2(p)
        pi = pisano_period(p)
        if pi % alpha != 0:
            errors.append({"p": p, "alpha": alpha, "pi": pi, "error": "alpha does not divide pi"})
            continue
        q = pi // alpha
        chi = kronecker_5(p)
        quotients[p] = {"alpha": alpha, "pi": pi, "quotient": q, "chi": chi}
        if q not in {1, 2, 4}:
            errors.append({"p": p, "alpha": alpha, "pi": pi, "quotient": q})
    return {
        "ok": len(errors) == 0,
        "count": len(all_primes),
        "errors": errors,
        "quotients": quotients,
        "desc": "pi(p)/alpha(p) in {1,2,4} for all tested primes",
    }


def check_c4_qa_orbit_reading():
    """C4: alpha(p) is the first step where the QA T-orbit of (0,1) mod p hits a zero-first component.

    T-orbit of (0,1) mod p: (0,1) -> (1,1) -> (1,2) -> (2,3) -> ...
    At step n the pair is (F_n mod p, F_{n+1} mod p).
    alpha(p) = smallest n >= 1 with F_n ≡ 0 mod p = first time pair is (0, F_{n+1}).
    The QA Kronecker symbol (5/p) = +1/-1/0 then determines alpha(p) | p-(5/p).
    Verify for sample primes.
    """
    sample = [11, 41, 3, 7, RAM_PRIME]
    results = {}
    errors = []
    for p in sample:
        alpha = rank_of_apparition_v2(p)
        chi = kronecker_5(p)
        target = p - chi
        # Verify T-orbit: at step alpha, first component (= F_alpha mod p) is 0
        a, b = 0, 1
        for n in range(1, alpha + 1):
            a, b = b, (a + b) % p
        # Now a = F_alpha mod p; should be 0
        first_comp_zero = (a == 0)
        divides = target % alpha == 0
        results[p] = {
            "alpha": alpha,
            "chi_5_p": chi,
            "target": target,
            "F_alpha_mod_p": a,
            "first_component_zero": first_comp_zero,
            "alpha_divides_target": divides,
        }
        if not first_comp_zero or not divides:
            errors.append({"p": p, "alpha": alpha, "F_alpha": a})
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "results": results,
        "desc": "QA T-orbit: step alpha(p) is first return to zero first-component; alpha(p)|p-(5/p) verified for 5 sample primes",
    }


def main():
    c1 = check_c1_unified_formula()
    c2 = check_c2_alpha_divides_pi()
    c3 = check_c3_entry_quotient()
    c4 = check_c4_qa_orbit_reading()
    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_unified_alpha_divides_p_minus_kronecker": c1,
            "C2_alpha_divides_pisano": c2,
            "C3_entry_quotient_in_1_2_4": c3,
            "C4_qa_orbit_first_zero": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
