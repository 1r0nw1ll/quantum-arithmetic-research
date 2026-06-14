# Primary source: Wall, D.D. (1960) "Fibonacci series modulo m" doi:10.2307/2309169
# Sun, Z.-H. and Sun, Z.-W. (1992) "Fibonacci and Lucas congruences and their applications"
#   Acta Arithmetica 60(3) pp.253-261
# McIntosh, R.J. and Roettger, E.L. (2007) "A search for Fibonacci-Wieferich and Wolstenholme primes"
#   Mathematics of Computation 76(260) doi:10.1090/S0025-5718-07-01955-2
# Cert [417]: QA Pisano Lift to Prime Squares — Wall-Sun-Sun Regularity
"""
Cert [417] — The QA T-step, applied modulo p², yields period p·π(p) (not π(p)):

    π(p²) = p · π(p)    [Pisano period lifts by factor p]

Cert [416] proved p | F_{p−(5/p)} for all primes (first-order divisibility via α(p)|p−(5/p)).
Cert [417] lifts to: p² ∤ F_{p−(5/p)} for all tested primes — the second-order condition fails,
as expected under the Wall-Sun-Sun conjecture (Sun-Sun 1992).

A prime p violating π(p²) = p·π(p) — i.e., with π(p²) = π(p) instead — is called a
Wall-Sun-Sun prime. None is known; numerical search confirms none below 2×10¹⁶.

Equivalent conditions for the same prime p (Wall 1960, Sun-Sun 1992):
  (A) π(p²) = π(p)          [Pisano period does NOT lift]
  (B) α(p²) = α(p)          [rank of apparition does NOT lift]
  (C) p² | F_{α(p)}         [α(p) is also rank of apparition mod p²]
  (D) w_F(p) := F_{p−(5/p)} / p  mod p  = 0   [Wall-Sun-Sun Fermat quotient vanishes]

All four conditions hold simultaneously for any Wall-Sun-Sun prime, and none holds for any
tested prime. All computations are pure integer T-step arithmetic (Theorem NT: p² is integer).

CHAIN: [415] T mod p, period π(p) → [416] α(p)|p−(5/p), one formula →
       [417] T mod p², period p·π(p), Wall-Sun-Sun regularity for all tested primes
"""

import json

SPLIT_PRIMES = [11, 31, 41, 61, 71, 101, 131, 151, 181, 191, 211, 241,
                251, 271, 281, 311, 331, 401, 421, 431, 461, 491]
INERT_PRIMES = [2, 3, 7, 13, 17, 23, 37, 43, 47, 53, 67, 73, 83, 97,
                103, 107, 113, 127, 137, 167, 173, 193]
RAM_PRIME = 5
SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def kronecker_5(p):
    """(5/p): +1 split, -1 inert, 0 ramified. Pure integer from p%5."""
    r = p % 5
    if r == 0:
        return 0
    if r in {1, 4}:
        return 1
    return -1


def fib_at(n, m):
    """F_n mod m via n T-steps T(b,e)=(e,b+e). Pure integer arithmetic."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % m
    return a


def pisano_period(m):
    """Period of QA T-orbit of (0,1) mod m: smallest k≥1 with (F_k,F_{k+1})≡(0,1) mod m."""
    a, b = 0, 1
    for k in range(1, 6 * m + 2):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return k
    raise ValueError(f"Pisano period not found for m={m} in 6m bound")


def rank_of_apparition_mod_p(p):
    """α(p): smallest n≥1 with F_n ≡ 0 mod p. Pure integer T-step."""
    a, b = 0, 1
    for n in range(1, 4 * p + 4):
        a, b = b, (a + b) % p
        if a == 0:
            return n
    raise ValueError(f"rank of apparition not found for p={p}")


def check_c1_pisano_lift():
    """C1: π(p²) = p·π(p) for 15 small primes ≤ 47. Direct T-step computation mod p²."""
    errors = []
    witnesses = {}
    for p in SMALL_PRIMES:
        pi_p = pisano_period(p)
        pi_p2 = pisano_period(p * p)
        expected = p * pi_p
        ok = (pi_p2 == expected)
        witnesses[p] = {
            "pi_p": pi_p,
            "pi_p2": pi_p2,
            "p_pi_p": expected,
            "lift_ok": ok,
        }
        if not ok:
            errors.append({
                "p": p, "pi_p": pi_p, "pi_p2": pi_p2, "expected": expected,
                "note": "Wall-Sun-Sun prime detected" if pi_p2 == pi_p else "unexpected quotient",
            })
    return {
        "ok": len(errors) == 0,
        "count": len(SMALL_PRIMES),
        "errors": errors,
        "witnesses": witnesses,
        "desc": (f"pi(p^2) = p*pi(p) for all {len(SMALL_PRIMES)} primes ≤ 47: "
                 "Pisano period lifts cleanly by factor p; no Wall-Sun-Sun prime found"),
    }


def check_c2_wall_sun_sun_quotient():
    """C2: Wall-Sun-Sun Fermat quotient w_F(p) = F_{p−(5/p)}/p mod p ≠ 0 for all 45 primes.

    Cert [416]: p | F_{p−(5/p)} (first-order). This check: p² ∤ F_{p−(5/p)} (second-order).
    w_F(p) = (F_{p−(5/p)} mod p²) // p  mod p. Non-zero for all primes tested.
    """
    errors = []
    quotients = {}
    all_primes = sorted(SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME])
    for p in all_primes:
        chi = kronecker_5(p)
        n = p - chi
        p2 = p * p
        f_n = fib_at(n, p2)
        div_p = (f_n % p == 0)
        div_p2 = (f_n == 0)
        w_f = (f_n // p) % p if div_p else None
        quotients[p] = {
            "chi_5_p": chi, "n": n,
            "F_n_mod_p2": f_n,
            "div_by_p": div_p,
            "div_by_p2": div_p2,
            "w_F": w_f,
        }
        if not div_p:
            errors.append({"p": p, "error": "p does not divide F_{p-(5/p)} — contradicts cert [416]"})
        elif div_p2:
            errors.append({"p": p, "error": "w_F=0: Wall-Sun-Sun prime detected"})
    return {
        "ok": len(errors) == 0,
        "count": len(all_primes),
        "errors": errors,
        "quotients": quotients,
        "desc": (f"w_F(p)≠0 for all {len(all_primes)} primes: "
                 "p | F_{p-(5/p)} (cert [416]) but p^2 does not; no Wall-Sun-Sun prime in test set"),
    }


def check_c3_rank_lift():
    """C3: α(p²) = p·α(p) for all 45 primes — rank of apparition lifts by factor p.

    Equivalent: p² ∤ F_{α(p)} (condition C above). Checked by computing F_{α(p)} mod p²:
    if the result is non-zero mod p², then α(p²) = p·α(p) (not α(p)).
    This uses only α(p) T-step iterations mod p² — much cheaper than computing α(p²) directly.
    """
    errors = []
    witnesses = {}
    all_primes = sorted(SPLIT_PRIMES + INERT_PRIMES + [RAM_PRIME])
    for p in all_primes:
        alpha_p = rank_of_apparition_mod_p(p)
        p2 = p * p
        f_alpha = fib_at(alpha_p, p2)
        div_p = (f_alpha % p == 0)
        div_p2 = (f_alpha == 0)
        rank_lift_ok = not div_p2
        witnesses[p] = {
            "alpha_p": alpha_p,
            "F_alpha_mod_p2": f_alpha,
            "div_by_p": div_p,
            "div_by_p2": div_p2,
            "alpha_p2_eq_p_alpha_p": rank_lift_ok,
        }
        if not div_p:
            errors.append({"p": p, "error": "internal: F_{alpha(p)} not divisible by p"})
        elif div_p2:
            errors.append({"p": p, "error": "p^2 | F_{alpha(p)}: alpha(p^2) = alpha(p), not p*alpha(p)"})
    return {
        "ok": len(errors) == 0,
        "count": len(all_primes),
        "errors": errors,
        "witnesses": witnesses,
        "desc": (f"alpha(p^2)=p*alpha(p) for all {len(all_primes)} primes: "
                 "F_{alpha(p)} divisible by p but NOT p^2 in every case"),
    }


def check_c4_orbit_reading():
    """C4: QA T-orbit of (0,1) mod p²: first p-zero at step α(p), first p²-zero at step p·α(p).

    At step α(p): first component F_{alpha(p)} ≡ 0 mod p, ≢ 0 mod p² (C3 confirms).
    At step p·α(p): first component F_{p*alpha(p)} ≡ 0 mod p² (by definition of α(p²)=p·α(p)).
    The orbit mod p² "unwinds p times" before hitting the first exact p²-zero.
    This is the discrete analogue of a p-adic expansion: the orbit has p "sheets" over mod p.
    """
    sample = [11, 3, 5, 41, 7]
    errors = []
    results = {}
    for p in sample:
        alpha_p = rank_of_apparition_mod_p(p)
        p2 = p * p
        f_at_alpha = fib_at(alpha_p, p2)
        f_at_p_alpha = fib_at(p * alpha_p, p2)
        first_p_zero = (f_at_alpha % p == 0)
        not_p2_zero = (f_at_alpha != 0)
        p2_zero_at_p_alpha = (f_at_p_alpha == 0)
        ok = first_p_zero and not_p2_zero and p2_zero_at_p_alpha
        results[p] = {
            "alpha_p": alpha_p,
            "p_alpha_p": p * alpha_p,
            "F_at_alpha_mod_p2": f_at_alpha,
            "F_at_p_alpha_mod_p2": f_at_p_alpha,
            "first_p_zero_at_alpha": first_p_zero,
            "no_p2_zero_at_alpha": not_p2_zero,
            "p2_zero_at_p_alpha": p2_zero_at_p_alpha,
        }
        if not ok:
            errors.append({"p": p})
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "results": results,
        "desc": ("QA T-orbit mod p^2: p-zero at step alpha(p), p^2-zero at step p*alpha(p); "
                 "p sheets in the p-adic unfolding; verified for 5 sample primes"),
    }


def main():
    c1 = check_c1_pisano_lift()
    c2 = check_c2_wall_sun_sun_quotient()
    c3 = check_c3_rank_lift()
    c4 = check_c4_orbit_reading()
    all_ok = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"]
    result = {
        "ok": all_ok,
        "checks": {
            "C1_pisano_lift_pi_p2_eq_p_pi_p": c1,
            "C2_wall_sun_sun_fermat_quotient_nonzero": c2,
            "C3_rank_lift_alpha_p2_eq_p_alpha_p": c3,
            "C4_orbit_p_zero_at_alpha_p2_zero_at_p_alpha": c4,
        },
    }
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
