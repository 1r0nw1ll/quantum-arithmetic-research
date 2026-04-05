"""Verify the dual extremality of m=24: simultaneously the minimum Pisano-period
fixed point and the maximum Carmichael-lambda=2 modulus.

Facts to verify (all classical, none original):
  1. pi(24) = 24 (Pisano period fixed point)
  2. pi(9)  = 24 (QA theoretical modulus maps to QA applied modulus)
  3. For m in 1..200, pi(m) = m iff m in {24, 120} (within the test range);
     classified completely in OEIS A235702 as {24 * 5^k : k >= 0}.
  4. 24 is the MINIMUM of the Pisano fixed point set.
  5. For m in 1..100, lambda(m) = 2 iff m in {1, 2, 3, 4, 6, 8, 12, 24}.
  6. 24 is the MAXIMUM of the Carmichael-lambda=2 set.
  7. Cannonball identity: 1^2 + 2^2 + ... + 24^2 = 4900 = 70^2 (Watson 1918, unique).
  8. 24 theorem: p^2 - 1 is divisible by 24 for every prime p >= 5.

The JOINT observation — that 24 is simultaneously minimum under pi and maximum
under lambda=2 — is the original contribution of the QA project.
"""

QA_COMPLIANCE = "theory_verification — classical number-theoretic facts about Pisano periods, Carmichael lambda, cannonball identity, and 24-theorem; integer-only, no observer, no floats"

from math import gcd


def qa_mod(x, m):
    """A1-compliant."""
    return ((int(x) - 1) % m) + 1


def pisano_period(m: int) -> int:
    """Pisano period: length of Fibonacci sequence mod m before repeating 0,1."""
    if m <= 0:
        return 0
    if m == 1:
        return 1
    prev, curr = 0, 1
    for k in range(1, 6 * m + 1):
        prev, curr = curr, (prev + curr) % m
        if prev == 0 and curr == 1:
            return k
    return 0  # unreachable for m >= 1


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def carmichael_lambda(n: int) -> int:
    """Carmichael function lambda(n): exponent of (Z/nZ)*."""
    if n == 1:
        return 1
    # Factorize
    factors = {}
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            factors[p] = factors.get(p, 0) + 1
            x //= p
        p += 1
    if x > 1:
        factors[x] = factors.get(x, 0) + 1

    # lambda for each prime power, then lcm
    result = 1
    for p, k in factors.items():
        if p == 2:
            if k == 1:
                lam_pk = 1
            elif k == 2:
                lam_pk = 2
            else:
                lam_pk = 2 ** (k - 2)
        else:
            lam_pk = (p - 1) * p ** (k - 1)
        result = lcm(result, lam_pk)
    return result


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def main():
    print("=== Pisano period facts ===")
    print(f"  pi(9)  = {pisano_period(9)}   (expected 24)")
    print(f"  pi(12) = {pisano_period(12)}   (expected 24)")
    print(f"  pi(24) = {pisano_period(24)}   (expected 24 — FIXED POINT)")
    print(f"  pi(120) = {pisano_period(120)} (expected 120 — next fixed point)")
    print()

    print("=== Pisano fixed points in [1, 200] ===")
    pisano_fp = [m for m in range(1, 201) if pisano_period(m) == m]
    print(f"  {pisano_fp}")
    print(f"  minimum = {min(pisano_fp)} (expected 24)")
    # OEIS A235702: fixed points are 24 * 5^k for k >= 0
    print(f"  Match to OEIS A235702 (24 * 5^k): {[24 * 5**k for k in range(3) if 24 * 5**k <= 200]}")
    print()

    print("=== Carmichael lambda facts ===")
    lambda_two_set = [m for m in range(1, 101) if carmichael_lambda(m) == 2]
    print(f"  {{m in [1,100] : lambda(m) = 2}} = {lambda_two_set}")
    print(f"  Expected: [1, 2, 3, 4, 6, 8, 12, 24]")
    print(f"  maximum = {max(lambda_two_set)} (expected 24)")
    print()

    print("=== Joint extremality at 24 ===")
    print(f"  pi(24) = 24: {pisano_period(24) == 24}")
    print(f"  lambda(24) = 2: {carmichael_lambda(24) == 2}")
    print(f"  24 = min Pisano fixed point: {min(pisano_fp) == 24}")
    print(f"  24 = max lambda=2 modulus (in [1,100]): {max(lambda_two_set) == 24}")
    print()

    print("=== Cannonball identity ===")
    square_sum = sum(k * k for k in range(1, 25))
    print(f"  1^2 + 2^2 + ... + 24^2 = {square_sum}")
    print(f"  70^2 = {70 * 70}")
    print(f"  Match: {square_sum == 70 * 70}")
    print()

    print("=== 24 theorem: p^2 - 1 divisible by 24 for primes p >= 5 ===")
    primes_to_test = [p for p in range(5, 50) if is_prime(p)]
    for p in primes_to_test[:10]:
        print(f"  p={p}: p^2-1 = {p*p - 1}, mod 24 = {(p*p - 1) % 24}")
    all_ok = all((p*p - 1) % 24 == 0 for p in primes_to_test)
    print(f"  All {len(primes_to_test)} primes checked: {all_ok}")
    print()

    print("=== QA theoretical -> applied bridge ===")
    print(f"  pi(9) = 24: m=9 (theoretical) maps to m=24 (applied) under Pisano")
    print(f"  pi(24) = 24: m=24 is self-stable")
    print(f"  pre-image set pi^-1(24) in [1,30] = {[m for m in range(1, 31) if pisano_period(m) == 24]}")


if __name__ == "__main__":
    main()
