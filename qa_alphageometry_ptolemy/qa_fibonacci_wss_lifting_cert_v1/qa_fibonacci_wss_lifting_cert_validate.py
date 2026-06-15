"""
Cert [429]: QA Fibonacci Wall-Sun-Sun Lifting to p²

From cert [428] C3: F_{alpha(p)} ≡ 0 (mod p) for all primes p (Wall 1960).
The deeper question is the p-adic valuation v_p(F_{alpha(p)}): does p² also divide it?

A prime p where p² | F_{alpha(p)} is called a Wall-Sun-Sun prime (or
Fibonacci-Wieferich prime). Wall (1960) asked whether such primes exist; Sun & Sun
(1992) showed their existence would imply the first case of Fermat's Last Theorem for
exponent p, and conjectured no such prime exists. No Wall-Sun-Sun prime is known as
of 2024 (exhaustive search has reached p > 9.7 × 10^14).

The generic lifting behavior (non-WSS case):

  C1 (Non-WSS criterion):
    v_p(F_{alpha(p)}) = 1 for all primes p in [5, 500].
    Equivalently: F_{alpha(p)} ≡ 0 (mod p) but F_{alpha(p)} ≢ 0 (mod p²).
    Confirms no prime in [5, 500] is a Wall-Sun-Sun prime.

  C2 (LTE lifting identity):
    F_{p * alpha(p)} ≡ 0 (mod p²) for all primes p in [5, 500].
    Proof (Lifting-the-Exponent for Lucas sequences, odd p):
      v_p(F_{k * alpha}) = v_p(F_alpha) + v_p(k)  for any k ≥ 1.
    Setting k = p: v_p(F_{p*alpha}) = 1 + 1 = 2. So p^2 | F_{p*alpha}. QED.
    This is the key lifting identity: multiplying the index by p raises the valuation by 1.

  C3 (alpha lifting):
    alpha(p²) = p * alpha(p) for all primes p in [5, 500].
    Proof:
      (i)  alpha(p) | alpha(p²) and alpha(p²) | p * alpha(p)  [Wall's range theorem].
      (ii) alpha(p²) != alpha(p): would require F_{alpha(p)} ≡ 0 (mod p²), i.e., WSS;
           excluded by C1.
      (iii) p is prime, so alpha(p²)/alpha(p) ∈ {1, p}; (ii) excludes 1. QED.
    Verified directly: F_{p*alpha} ≡ 0 mod p² (from C2) and F_{alpha} ≢ 0 mod p² (C1).

  C4 (Pisano period lifting):
    T(p²) = p * T(p) for all primes p in [5, 300].
    Two sub-checks:
      (a) (F_{p*T(p)}, F_{p*T(p)+1}) ≡ (0,1) mod p²: p*T(p) is a period of F mod p².
          => T(p²) | p*T(p).
      (b) (F_{T(p)}, F_{T(p)+1}) ≢ (0,1) mod p²: T(p) is NOT a period of F mod p².
          => T(p²) > T(p), so T(p²) / T(p) > 1.
      Combined with Wall's structure theorem T(p²)/T(p) ∈ {1, p} (p prime):
      T(p²) = p * T(p). QED.
    Proof of (a): T(p²) = alpha(p²) * ord(eps(p²)). From C3: alpha(p²) = p*alpha(p).
      ord(eps(p²)) = ord(eps(p)) (Hensel: eps(p) = +-1 or prim. 4th root; eps(p)^{ord}
      lifts to 1 mod p²). So T(p²) = p*alpha(p)*ord(eps(p)) = p*T(p).

Primary sources:
  Wall, D.D. (1960) doi:10.2307/2309169 -- v_p(F_{k*alpha}) = v_p(F_alpha) + v_p(k);
    defines Wall-Sun-Sun property; proves alpha(p)|alpha(p^n) and range theorem
  Sun, Z.H. and Sun, Z.W. (1992) doi:10.1007/BF01350576 -- connection WSS <->
    Fermat's Last Theorem first case; conjectured no WSS prime exists
  Lengyel, T. (1995) "The order of the Fibonacci and Lucas numbers" -- LTE formula
    for Lucas sequences; v_p(F_{p*n}) = v_p(F_n) + 1 for odd prime p, p | F_n
"""

import math
from fractions import Fraction  # noqa: F401


# ── QA LAYER: pure integer arithmetic ────────────────────────────────────────

def _fib_pair(n, m):
    """(F_n, F_{n+1}) mod m via fast doubling. Iterative O(log n). Pure integer."""
    if n == 0:
        return 0, 1
    a, b = 0, 1
    bits = n.bit_length()
    for shift in range(bits - 1, -1, -1):
        c = a * (2 * b - a) % m
        d = (a * a + b * b) % m
        if (n >> shift) & 1:
            a, b = d, (c + d) % m
        else:
            a, b = c, d
    return a, b


def fib_fast(n, m):
    """F_n mod m. Pure integer."""
    if n == 0:
        return 0
    return _fib_pair(n, m)[0]


def rank_of_apparition(p):
    """alpha(p): smallest k>=1 with F_k ≡ 0 mod p. Pure integer walk."""
    a, b = 0, 1
    k = 0
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0:
            return k


def pisano_period(p):
    """T(p): smallest k>=1 with (F_k, F_{k+1}) ≡ (0,1) mod p. Pure integer walk."""
    a, b = 0, 1
    k = 0
    while True:
        a, b = b, (a + b) % p
        k += 1
        if a == 0 and b == 1:
            return k


def sieve(n):
    """Primes up to n. Pure integer."""
    is_p = bytearray([1]) * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, math.isqrt(n) + 1):
        if is_p[i]:
            is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]


# ── CHECK FUNCTIONS ───────────────────────────────────────────────────────────

def check_c1_non_wss(n_bound=500):
    """C1: v_p(F_{alpha(p)}) = 1 for all primes p in [5, n_bound].
    Check: F_{alpha(p)} ≡ 0 mod p (Wall, [428]) AND F_{alpha(p)} ≢ 0 mod p².
    A failure would be the first known Wall-Sun-Sun prime.
    Current computational lower bound on WSS primes: > 9.7 × 10^14.
    """
    wss_candidates = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        fa_p2 = fib_fast(alpha, p * p)
        if fa_p2 == 0:
            wss_candidates.append((p, alpha, "F_alpha(p) = 0 mod p^2: WSS candidate"))
    return {
        "ok": len(wss_candidates) == 0,
        "primes_tested": tested,
        "wss_candidates": wss_candidates,
        "desc": (
            f"v_p(F_{{alpha(p)}})=1 (non-WSS): {tested - len(wss_candidates)}/{tested} "
            f"primes in [5,{n_bound}] PASS; no Wall-Sun-Sun prime found"
        ),
    }


def check_c2_lte_lifting(n_bound=500):
    """C2: F_{p * alpha(p)} ≡ 0 (mod p²) for all primes p in [5, n_bound].
    Proof: LTE for Lucas sequences (Lengyel 1995):
      v_p(F_{k * alpha(p)}) = v_p(F_{alpha(p)}) + v_p(k)  for odd prime p, p | F_alpha.
    Setting k = p: v_p(F_{p*alpha}) = 1 + 1 = 2. So F_{p*alpha} ≡ 0 mod p^2.
    This is the key lifting identity: the p-fold repetition of the zero index
    raises the p-adic valuation by exactly 1.
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        fp_alpha_mod_p2 = fib_fast(p * alpha, p * p)
        if fp_alpha_mod_p2 != 0:
            fails.append((p, alpha, fp_alpha_mod_p2))
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"F_{{p*alpha(p)}} = 0 mod p^2 (LTE lifting): "
            f"{tested - len(fails)}/{tested} primes in [5,{n_bound}] PASS"
        ),
    }


def check_c3_alpha_lifting(n_bound=500):
    """C3: alpha(p²) = p * alpha(p) for all primes p in [5, n_bound].
    Proof uses C1 + C2 + primality of p:
      Range: alpha(p) | alpha(p²) | p * alpha(p)  (Wall 1960 Theorem 3).
      C2: p * alpha(p) is a zero of F_k mod p^2.
      C1: alpha(p) is NOT a zero of F_k mod p^2 (F_{alpha(p)} ≢ 0 mod p^2).
      Since p is prime, alpha(p²)/alpha(p) ∈ {1, p}, and it's not 1.
      => alpha(p²) = p * alpha(p).
    Verified directly: both conditions checked in a single pass.
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        alpha = rank_of_apparition(p)
        p2 = p * p
        fa_p2 = fib_fast(alpha, p2)          # F_{alpha} mod p^2: should be != 0
        fpa_p2 = fib_fast(p * alpha, p2)     # F_{p*alpha} mod p^2: should be 0
        ok = (fa_p2 != 0) and (fpa_p2 == 0)
        if not ok:
            fails.append({
                "p": p, "alpha": alpha,
                "F_alpha_mod_p2": fa_p2, "F_palpha_mod_p2": fpa_p2,
            })
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"alpha(p^2) = p*alpha(p) (lifting): "
            f"{tested - len(fails)}/{tested} primes in [5,{n_bound}] PASS"
        ),
    }


def check_c4_pisano_lifting(n_bound=300):
    """C4: T(p²) = p * T(p) for all primes p in [5, n_bound].
    Two sub-checks per prime:
      (a) p*T(p) is a period mod p²:
          (F_{p*T(p)}, F_{p*T(p)+1}) ≡ (0,1) mod p^2  =>  T(p²) | p*T(p).
      (b) T(p) is NOT a period mod p²:
          (F_{T(p)}, F_{T(p)+1}) ≢ (0,1) mod p^2  =>  T(p²) > T(p).
    Wall structure: T(p)|T(p²) and T(p²)/T(p) ∈ {1,p} (p prime, LTE + Galois ring).
    (b) excludes ratio 1; (a) gives ratio ≤ p; so ratio = p => T(p²) = p*T(p).
    Analytic proof of (a): T(p²) = alpha(p²)*ord(eps(p²)) = p*alpha(p)*ord(eps(p))
    = p*T(p), using C3 and the Hensel lifting of ord(eps).
    """
    fails = []
    tested = 0
    for p in sieve(n_bound):
        if p < 5:
            continue
        tested += 1
        T = pisano_period(p)
        p2 = p * p
        # Sub-check (a): p*T is a period mod p^2
        fn_pT, fn1_pT = _fib_pair(p * T, p2)
        ok_a = (fn_pT == 0 and fn1_pT == 1)
        # Sub-check (b): T is NOT a period mod p^2
        fn_T, fn1_T = _fib_pair(T, p2)
        ok_b = not (fn_T == 0 and fn1_T == 1)
        if not (ok_a and ok_b):
            fails.append({
                "p": p, "T": T,
                "ok_a_pT_period": ok_a, "ok_b_T_not_period": ok_b,
                "F_pT_mod_p2": fn_pT, "F_pT1_mod_p2": fn1_pT,
                "F_T_mod_p2": fn_T, "F_T1_mod_p2": fn1_T,
            })
    return {
        "ok": len(fails) == 0,
        "primes_tested": tested,
        "fails": fails,
        "desc": (
            f"T(p^2) = p*T(p) (Pisano lifting): "
            f"{tested - len(fails)}/{tested} primes in [5,{n_bound}] PASS"
        ),
    }


# ── OBSERVER LAYER: summary (no float state) ─────────────────────────────────

def _run_checks():
    return {
        "c1_non_wss":        check_c1_non_wss(500),
        "c2_lte_lifting":    check_c2_lte_lifting(500),
        "c3_alpha_lifting":  check_c3_alpha_lifting(500),
        "c4_pisano_lifting": check_c4_pisano_lifting(300),
    }


def main():
    import json
    results = _run_checks()
    all_ok = all(v["ok"] for v in results.values())
    print(json.dumps({
        "cert": "[429] QA Fibonacci Wall-Sun-Sun Lifting to p^2",
        "all_checks_pass": all_ok,
        "checks": {k: {"ok": v["ok"], "desc": v["desc"]} for k, v in results.items()},
    }, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
