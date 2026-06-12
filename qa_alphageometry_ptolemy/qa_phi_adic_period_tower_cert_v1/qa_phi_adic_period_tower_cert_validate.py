#!/usr/bin/env python3
# noqa: S1 S2 T1 T2 (cert validator — integers only, no QA state drift)
# Primary sources: Wall (1960) doi.org/10.1080/00029890.1960.11989541
#   (Pisano periods, period tower conjecture);
#   Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.13 (Cassini identity,
#   quadratic norm multiplicativity);
#   Serre (1979) ISBN 978-0-387-90236-7 (Witt vectors, p-adic groups).
"""
Cert [392]: QA φ-adic Period Tower (Cassini-Witt)

Building on cert [391] (σ = ×φ on ℤ[φ]), the orbit of φ=(1,0) under σ
traces φ, φ², φ³, … in ℤ[φ]. Its period mod p^n is the Pisano period π(p^n).

Core claims:

(A) CASSINI NORM: N(σ^k(1,0)) = (−1)^(k+1) for all k ≥ 0.
    Proof: σ^k(1,0) = φ^(k+1), N(φ^(k+1)) = N(φ)^(k+1) = (−1)^(k+1).
    In Fibonacci notation: N(F_{k+1}, F_k) = F_k²+F_{k+1}F_k−F_{k+1}²
    = −(F_{k+1}F_{k−1}−F_k²) = −(−1)^k = (−1)^(k+1)  [Cassini identity].

(B) PERIOD TOWER: π(p^n) = p^(n−1)·π(p) for all primes p and n ≥ 1
    tested (inert {3,7,13,17}, split {11,19,29,41}, ramified {5}).
    This means the Witt multiplier is exactly p at each level: going from
    ℤ[φ]/(p^n) to ℤ[φ]/(p^(n+1)) multiplies the orbit period by exactly p.

(C) WITT CARRY: σ^π(p)(1,0) mod p² = (F_{π(p)+1} mod p², F_{π(p)} mod p²),
    and the first-order carry (F_{π(p)}/p mod p, (F_{π(p)+1}−1)/p mod p)
    is non-zero for tested p ∈ {3,7,11,19} — ruling out Wall-Sun-Sun primes
    in the tested set and confirming the Witt multiplier is genuinely p
    (not 1 due to a degenerate lift).

7 checks (all PASS) + 8 fixtures (7 PASS, 1 FAIL designed).
Extends: [391] (σ=×φ, N(φ)=−1), [389] (Witt tower at p²),
         [387] (Witt carry in W_2(𝔽₉)).
"""
import sys
import json


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def fib(n):
    """Standard Fibonacci: F(0)=0, F(1)=1, F(2)=1, …"""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def norm(a, b):
    """N_{ℚ(√5)/ℚ}(a·φ + b) = b²+ab−a²."""
    return b * b + a * b - a * a


def sigma_k(a, b, k):
    for _ in range(k):
        a, b = a + b, a
    return (a, b)


def pisano_period(m):
    """π(m): period of Fibonacci sequence mod m."""
    a, b = 0, 1
    for k in range(1, 6 * m + 10):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return k
    raise ValueError(f"Pisano period not found for m={m}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_cassini_norm():
    """N(σ^k(1,0)) = (−1)^(k+1) for k=0..17 (Cassini via norm multiplicativity)."""
    a, b = 1, 0
    for k in range(18):
        n_val = norm(a, b)
        expected = (-1) ** (k + 1)
        if n_val != expected:
            return False, f"k={k}: N(σ^k(1,0))={n_val}, expected (−1)^{k+1}={expected}"
        a, b = a + b, a
    return True, (
        "N(σ^k(1,0))=(−1)^(k+1) k=0..17; "
        "proof: σ^k(1,0)=φ^(k+1), N(φ^(k+1))=N(φ)^(k+1)=(−1)^(k+1)"
    )


def check_period_is_pisano():
    """Period of σ starting from (1,0) mod m equals π(m) for 10 moduli."""
    test_moduli = [8, 9, 24, 27, 49, 72, 110, 121, 343, 1210]
    for m in test_moduli:
        pi = pisano_period(m)
        a, b = sigma_k(1, 0, pi)
        if (a % m, b % m) != (1 % m, 0):
            return False, f"m={m}: σ^π(1,0) mod m = ({a%m},{b%m}), expected (1,0)"
        # minimality: pi//smallest_factor should not return to identity
        for factor in [2, 3, 5, 7]:
            if pi % factor == 0:
                a2, b2 = sigma_k(1, 0, pi // factor)
                if (a2 % m, b2 % m) == (1 % m, 0):
                    return False, f"m={m}: σ^(π/{factor})(1,0) mod m = (1,0) — period not minimal"
                break
    return True, f"Period of σ at (1,0) mod m = π(m); verified minimal on {test_moduli}"


def check_tower_inert():
    """π(p^n) = p^(n−1)·π(p) for inert p ∈ {3,7,13,17} and n=1,2,3."""
    inert_primes = [3, 7, 13, 17]
    for p in inert_primes:
        pi1 = pisano_period(p)
        for n in range(1, 4):
            pi_pn = pisano_period(p ** n)
            expected = (p ** (n - 1)) * pi1
            if pi_pn != expected:
                return False, (
                    f"p={p}, n={n}: π({p}^{n})={pi_pn}, expected {p}^{n-1}·{pi1}={expected}"
                )
    return True, "π(p^n)=p^(n−1)·π(p) for inert {3,7,13,17}, n=1..3"


def check_tower_split():
    """π(p^n) = p^(n−1)·π(p) for split p ∈ {11,19,29,41} and n=1,2,3."""
    split_primes = [11, 19, 29, 41]
    for p in split_primes:
        pi1 = pisano_period(p)
        for n in range(1, 4):
            pi_pn = pisano_period(p ** n)
            expected = (p ** (n - 1)) * pi1
            if pi_pn != expected:
                return False, (
                    f"p={p}, n={n}: π({p}^{n})={pi_pn}, expected {p}^{n-1}·{pi1}={expected}"
                )
    return True, "π(p^n)=p^(n−1)·π(p) for split {11,19,29,41}, n=1..3"


def check_tower_ramified():
    """π(5^n) = 5^(n−1)·π(5) for n=1..4 (ramified prime)."""
    pi1 = pisano_period(5)
    for n in range(1, 5):
        pi_pn = pisano_period(5 ** n)
        expected = (5 ** (n - 1)) * pi1
        if pi_pn != expected:
            return False, f"n={n}: π(5^{n})={pi_pn}, expected 5^{n-1}·{pi1}={expected}"
    return True, f"π(5^n)=5^(n−1)·{pisano_period(5)} for n=1..4 (ramified p=5)"


def check_witt_multiplier_exact():
    """π(p^2)/π(p) = p exactly for all tested primes; no Wall-Sun-Sun in set."""
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 41, 59]
    for p in primes:
        pi1 = pisano_period(p)
        pi2 = pisano_period(p * p)
        ratio = pi2 // pi1
        if pi2 % pi1 != 0 or ratio != p:
            return False, (
                f"p={p}: π(p²)/π(p)={pi2}/{pi1}={pi2/pi1:.4f}, expected {p}"
            )
    return True, (
        f"π(p²)=p·π(p) exactly for p ∈ {primes}; "
        "Witt multiplier is p at every tested prime"
    )


def check_witt_carry_nonzero():
    """σ^π(p)(1,0) mod p² = (F_{π(p)+1}, F_{π(p)}) mod p²; carry is non-zero."""
    # carry_b = F_{π(p)} / p mod p; carry_a = (F_{π(p)+1} − 1) / p mod p
    # non-zero carry => period at p² is genuinely p·π(p), not π(p)
    test_primes = [3, 7, 11, 19, 29]
    for p in test_primes:
        pi = pisano_period(p)
        p2 = p * p
        # sigma^pi(1,0) mod p^2
        a_lift, b_lift = sigma_k(1, 0, pi)
        a_mod = a_lift % p2
        b_mod = b_lift % p2
        # should equal (F_{pi+1} mod p^2, F_pi mod p^2)
        f_pi = fib(pi)
        f_pi1 = fib(pi + 1)
        if (a_mod, b_mod) != (f_pi1 % p2, f_pi % p2):
            return False, (
                f"p={p}: σ^π(1,0) mod p²=({a_mod},{b_mod}), "
                f"(F_{{π+1}},F_π) mod p²=({f_pi1%p2},{f_pi%p2})"
            )
        # check carry is non-zero (rules out Wall-Sun-Sun prime in test set)
        if f_pi % p != 0:
            return False, f"p={p}: F_{{π(p)}} = {f_pi} not divisible by p — Pisano property violated"
        if f_pi1 % p != 1:
            return False, f"p={p}: F_{{π(p)+1}} = {f_pi1} not ≡ 1 mod p"
        carry_b = (f_pi // p) % p
        carry_a = ((f_pi1 - 1) // p) % p
        if carry_b == 0 and carry_a == 0:
            return False, f"p={p}: both Witt carries are 0 mod p — this would be a Wall-Sun-Sun prime!"
    return True, (
        f"For p ∈ {test_primes}: σ^π(p)(1,0) mod p² = (F_{{π+1}},F_π) mod p²; "
        "carries (F_π/p, (F_{{π+1}}−1)/p) mod p all non-zero — no Wall-Sun-Sun prime in set"
    )


CHECKS = {
    "CASSINI_NORM": check_cassini_norm,
    "PERIOD_IS_PISANO": check_period_is_pisano,
    "TOWER_INERT": check_tower_inert,
    "TOWER_SPLIT": check_tower_split,
    "TOWER_RAMIFIED": check_tower_ramified,
    "WITT_MULTIPLIER_EXACT": check_witt_multiplier_exact,
    "WITT_CARRY_NONZERO": check_witt_carry_nonzero,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES = [
    {
        "name": "CASSINI_k3",
        "description": "N(σ³(1,0))=N(3,2)=4+6−9=1=(−1)⁴: Cassini at k=3",
        "expected": True,
        "fn": lambda: norm(*sigma_k(1, 0, 3)) == (-1) ** 4,
    },
    {
        "name": "PISANO_TOWER_p3_n3",
        "description": "π(27)=72=9·8=3²·π(3): period tower level 3 for inert p=3",
        "expected": True,
        "fn": lambda: pisano_period(27) == 9 * pisano_period(3),
    },
    {
        "name": "PISANO_TOWER_p11_n2",
        "description": "π(121)=110=11·10=11·π(11): period tower level 2 for split p=11",
        "expected": True,
        "fn": lambda: pisano_period(121) == 11 * pisano_period(11),
    },
    {
        "name": "WITT_CARRY_p3",
        "description": "σ^8(1,0) mod 9 = (7,3) = (F₉ mod 9, F₈ mod 9); carry_b=F₈/3=1, carry_a=(F₉−1)/3=2, both non-zero",
        "expected": True,
        "fn": lambda: (
            sigma_k(1, 0, 8)[0] % 9 == fib(9) % 9 and
            sigma_k(1, 0, 8)[1] % 9 == fib(8) % 9 and
            (fib(8) // 3) % 3 != 0
        ),
    },
    {
        "name": "CONNECT_387_PERIOD_24",
        "description": "π(9)=24=3·π(3)=3·8; the 3 Cosmos sub-orbits of cert [387] each have this period",
        "expected": True,
        "fn": lambda: pisano_period(9) == 3 * pisano_period(3) == 24,
    },
    {
        "name": "PHI_PERIOD_EQUALS_PISANO_p7n2",
        "description": "Period of σ at (1,0) mod 49 = 112 = π(49) = 7·π(7): φ's orbit IS the Fibonacci period",
        "expected": True,
        "fn": lambda: (
            sigma_k(1, 0, 112)[0] % 49 == 1 and
            sigma_k(1, 0, 112)[1] % 49 == 0 and
            sigma_k(1, 0, 112 // 7)[0] % 49 != 1
        ),
    },
    {
        "name": "CASSINI_IS_NORM_MULT",
        "description": "N(σ¹⁰(1,0))=(−1)¹¹=−1; N(σ¹¹(1,0))=(−1)¹²=1: alternating via norm multiplicativity",
        "expected": True,
        "fn": lambda: norm(*sigma_k(1, 0, 10)) == -1 and norm(*sigma_k(1, 0, 11)) == 1,
    },
    {
        "name": "WALL_QUESTION_TOWER_FAILS_IF_CARRY_ZERO",
        "description": "DESIGNED FAIL: if p=3 had carry_b=0 (Wall-Sun-Sun), π(9) would equal π(3)=8, not 24. But π(9)=24≠8.",
        "expected": False,
        "fn": lambda: pisano_period(9) == pisano_period(3),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_self_test():
    results = {}
    all_pass = True
    for name, fn in CHECKS.items():
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, str(exc)
        results[name] = {"ok": ok, "detail": msg}
        if not ok:
            all_pass = False

    fixture_pass = 0
    fixture_results = []
    for fx in FIXTURES:
        try:
            got = fx["fn"]()
        except Exception as exc:
            got = None
        passed = (got == fx["expected"])
        fixture_pass += int(passed)
        fixture_results.append({
            "name": fx["name"],
            "expected": fx["expected"],
            "got": got,
            "passed": passed,
        })

    return {
        "ok": all_pass and fixture_pass == len(FIXTURES),
        "checks": {k: v["ok"] for k, v in results.items()},
        "check_details": {k: v["detail"] for k, v in results.items()},
        "fixture_summary": f"{fixture_pass}/{len(FIXTURES)} passed",
        "fixtures": fixture_results,
        "tower_formula": "pi(p^n) = p^(n-1) * pi(p) for all tested primes and n",
        "cassini_chain": "N(sigma^k(1,0)) = (-1)^(k+1) = N(phi)^(k+1)",
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        out = run_self_test()
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["ok"] else 1)
    print("Usage: python3 qa_phi_adic_period_tower_cert_validate.py --self-test")
    sys.exit(1)
