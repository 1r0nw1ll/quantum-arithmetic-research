# Primary source: Cassini, J.-D. (1680) "Une nouvelle progression de nombres"
#   Histoire de l'Académie Royale des Sciences (original identity F_{n-1}F_{n+1} - F_n^2 = (-1)^n)
# Wall, D.D. (1960) "Fibonacci series modulo m" doi:10.2307/2309169
# Lehmer, D.H. (1930) "An extended theory of Lucas' functions" doi:10.2307/1968235
# Cert [418]: QA Cassini Alpha Parity Gate
"""
Cert [418] — The parity of the rank of apparition α(p) is constrained by the quadratic
character of -1 mod p, via the Cassini identity evaluated at the first-zero step.

Cassini identity (1680): F_{n-1} · F_{n+1} − F_n² = (−1)^n

At n = α(p) (the rank of apparition, the first step where p | F_n):
    F_{α(p)} ≡ 0 mod p
    F_{α(p)+1} ≡ F_{α(p)-1} mod p  (since F_{α+1} = F_α + F_{α-1} ≡ F_{α-1})
    ⟹  F_{α(p)-1}² ≡ (−1)^{α(p)} mod p     [Cassini Gate identity]

Consequence for parity:
    α(p) ODD  ⟹  F_{α(p)-1}² ≡ -1 mod p
                ⟹  -1 is a quadratic residue mod p
                ⟹  p ≡ 1 mod 4    (Euler's criterion)

Contrapositive: p ≡ 3 mod 4  ⟹  α(p) is EVEN for all primes p ≥ 3.

This gives a QA orbit-theoretic proof of a quadratic reciprocity consequence:
  the parity of the T-step first-zero time encodes whether p ≡ 1 or 3 mod 4.

Special case p = 2: α(2) = 3 (odd), but -1 ≡ 1 mod 2 is trivially a square; no contradiction.
Ramified p = 5: α(5) = 5 (odd), 5 ≡ 1 mod 4; consistent.

Note on δ(p) = F_{α(p)}/p mod p:
For odd-α primes where δ(p) = 1 (e.g., p = 193, α = 97):
    φ^{97} ≡ F_{96} + 193φ  mod 193²
    F_{96}² ≡ -1 mod 193   (C3 below)  ← F_{α-1} is a sqrt(-1) mod p
    δ(193) = 1 means: φ^{α(p)} ≡ i + p·φ (mod p²) where i² ≡ -1 mod p.
This is the minimum-depth lift: the QA orbit hits zero as fast as possible in the p-adic sense.
Fibonacci primes (p = F_k) also have δ = 1 but via a different mechanism (F_{α} = p exactly).

CHAIN: [416] α(p)|p-(5/p) → [417] α(p²)=p·α(p) → [418] α(p) parity ↔ p mod 4 (Cassini Gate)
"""

import json

SPLIT_PRIMES = [11, 31, 41, 61, 71, 101, 131, 151, 181, 191, 211, 241,
                251, 271, 281, 311, 331, 401, 421, 431, 461, 491]
INERT_PRIMES = [2, 3, 7, 13, 17, 23, 37, 43, 47, 53, 67, 73, 83, 97,
                103, 107, 113, 127, 137, 167, 173, 193]
RAM_PRIME = 5


def fib_at(n, m):
    """F_n mod m via n T-steps. Pure integer."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % m
    return a


def rank_of_apparition(p):
    """α(p): smallest n≥1 with F_n ≡ 0 mod p."""
    a, b = 0, 1
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    raise ValueError(f"rank not found for p={p}")


def check_c1_cassini_gate():
    """C1: F_{α(p)-1}² ≡ (-1)^{α(p)} mod p for all 45 tested primes.

    Proof sketch: Cassini gives F_{n-1}·F_{n+1} - F_n² = (-1)^n.
    At n=α(p): F_{α(p)}≡0 mod p, F_{α(p)+1}≡F_{α(p)-1} mod p.
    Substituting: F_{α(p)-1}² ≡ (-1)^{α(p)} mod p.
    This is provably true; we verify it computationally for all tested primes.
    """
    errors = []
    witnesses = {}
    all_primes = sorted(SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME])
    for p in all_primes:
        alpha = rank_of_apparition(p)
        f_prev = fib_at(alpha - 1, p)
        sq = (f_prev * f_prev) % p
        expected = pow(-1, alpha, p)     # +1 if even, p-1 if odd (all integer)
        ok = (sq == expected)
        witnesses[p] = {
            "alpha": alpha,
            "alpha_parity": "odd" if alpha % 2 else "even",
            "p_mod_4": p % 4,
            "F_alpha_minus_1_mod_p": f_prev,
            "F_prev_sq_mod_p": sq,
            "expected_minus1_power_alpha": expected,
            "cassini_gate_ok": ok,
        }
        if not ok:
            errors.append({"p": p, "alpha": alpha, "sq": sq, "expected": expected})
    return {
        "ok": len(errors) == 0,
        "count": len(all_primes),
        "errors": errors,
        "witnesses": witnesses,
        "desc": (f"F_{{alpha-1}}^2 ≡ (-1)^alpha mod p for all {len(all_primes)} primes: "
                 "Cassini identity evaluated at first-zero step"),
    }


def check_c2_parity_constraint():
    """C2: p ≡ 3 mod 4 ⟹ α(p) is even, for all tested primes p ≥ 3.

    Proof: if α(p) odd then F_{α-1}² ≡ -1 mod p (C1) → -1 is QR mod p → p ≡ 1 mod 4.
    Contrapositive: p ≡ 3 mod 4 → α(p) even. Verified for all p ≡ 3 mod 4 in test set.
    """
    errors = []
    p3_mod4 = []
    all_primes = sorted(SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME])
    for p in all_primes:
        if p % 4 == 3:
            alpha = rank_of_apparition(p)
            even = (alpha % 2 == 0)
            p3_mod4.append({"p": p, "alpha": alpha, "alpha_even": even})
            if not even:
                errors.append({"p": p, "alpha": alpha, "error": "p≡3 mod 4 but alpha ODD — Cassini violation"})
    return {
        "ok": len(errors) == 0,
        "count_p_3mod4": len(p3_mod4),
        "errors": errors,
        "witnesses_p3mod4": p3_mod4,
        "desc": (f"p≡3 mod 4 ⟹ α(p) even: verified for all {len(p3_mod4)} primes ≡ 3 mod 4 in test set; "
                 "no odd-rank prime with p≡3 mod 4 found"),
    }


def check_c3_sqrt_minus_one():
    """C3: For all odd-α primes p ≥ 3 with p ≡ 1 mod 4: F_{α(p)-1} is an explicit sqrt(-1) mod p.

    Consequence of C1 with odd α: F_{α-1}² ≡ -1 mod p.
    This gives an explicit square root of -1 in 𝔽_p computed by QA T-step iteration.
    Verified for all primes in test set with odd rank of apparition.
    """
    errors = []
    odd_alpha_witnesses = {}
    all_primes = sorted(SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME])
    for p in all_primes:
        alpha = rank_of_apparition(p)
        if alpha % 2 == 0:
            continue
        if p == 2:
            continue    # p=2: -1≡1, trivially square, not interesting
        # p must be ≡ 1 mod 4 (from C2 contrapositive)
        f_prev = fib_at(alpha - 1, p)
        sq = (f_prev * f_prev) % p
        is_sqrt_minus_1 = (sq == p - 1)
        odd_alpha_witnesses[p] = {
            "alpha": alpha,
            "p_mod_4": p % 4,
            "F_alpha_minus_1_mod_p": f_prev,
            "F_prev_squared_mod_p": sq,
            "is_sqrt_minus_1": is_sqrt_minus_1,
            "p_minus_1": p - 1,
        }
        if p % 4 != 1:
            errors.append({"p": p, "p_mod_4": p % 4, "error": "odd alpha but p≢1 mod 4"})
        if not is_sqrt_minus_1:
            errors.append({"p": p, "sq": sq, "error": "F_{alpha-1}^2 ≢ -1 mod p despite odd alpha"})
    return {
        "ok": len(errors) == 0,
        "count_odd_alpha": len(odd_alpha_witnesses),
        "errors": errors,
        "witnesses": odd_alpha_witnesses,
        "desc": (f"For all {len(odd_alpha_witnesses)} primes with odd α(p): "
                 "F_{α-1} is a QA T-step sqrt(-1) mod p; all satisfy p≡1 mod 4"),
    }


def check_c4_orbit_parity_reading():
    """C4: QA T-orbit of (0,1) mod p: the sign ((-1)^{α(p)}) is readable from the orbit alone.

    At step α(p): orbit = (F_{α(p)} mod p, F_{α(p)+1} mod p) ≡ (0, F_{α(p)-1} mod p).
    F_{α(p)-1}² mod p tells us the parity of α(p) without computing α(p) directly:
      result = p-1: α(p) is ODD  → p ≡ 1 mod 4
      result = 1:   α(p) is EVEN → p can be ≡ 1 or 3 mod 4
    Verified for 5 sample primes (one odd-alpha, one even-alpha per class).
    """
    sample = [
        (193, "inert, p≡1 mod 4, alpha ODD"),
        (37,  "inert, p≡1 mod 4, alpha ODD"),
        (41,  "split, p≡1 mod 4, alpha EVEN"),
        (11,  "split, p≡3 mod 4, alpha EVEN"),
        (7,   "inert, p≡3 mod 4, alpha EVEN"),
    ]
    errors = []
    results = {}
    for p, desc in sample:
        alpha = rank_of_apparition(p)
        # Step the orbit to step alpha(p): (0, F_{alpha+1}) but second = F_{alpha-1} mod p
        a, b = 0, 1
        for _ in range(alpha):
            a, b = b, (a + b) % p
        # Now a = F_{alpha} ≡ 0, b = F_{alpha+1} ≡ F_{alpha-1} mod p
        f_next = b         # = F_{alpha-1} mod p
        sign_sq = (f_next * f_next) % p
        expected_sign = pow(-1, alpha, p)
        parity_readable = (sign_sq == expected_sign)
        parity_decode = "ODD" if sign_sq == p - 1 else ("EVEN" if sign_sq == 1 else "ambiguous")
        results[p] = {
            "desc": desc,
            "alpha": alpha,
            "alpha_actual_parity": "ODD" if alpha % 2 else "EVEN",
            "orbit_at_alpha": (a, b),
            "F_next_sq_mod_p": sign_sq,
            "parity_decoded_from_orbit": parity_decode,
            "consistent": parity_readable and (parity_decode == ("ODD" if alpha % 2 else "EVEN")),
        }
        if not parity_readable or results[p]["parity_decoded_from_orbit"] == "ambiguous":
            errors.append({"p": p})
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "results": results,
        "desc": ("QA T-orbit: second component at step α(p) squared gives (-1)^{α(p)} mod p; "
                 "parity of α readable from orbit without computing α"),
    }


def main():
    c1 = check_c1_cassini_gate()
    c2 = check_c2_parity_constraint()
    c3 = check_c3_sqrt_minus_one()
    c4 = check_c4_orbit_parity_reading()
    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_cassini_gate_F_prev_sq_eq_minus1_power_alpha": c1,
            "C2_parity_constraint_p3mod4_implies_even_alpha": c2,
            "C3_odd_alpha_gives_explicit_sqrt_minus1": c3,
            "C4_orbit_parity_readout": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
