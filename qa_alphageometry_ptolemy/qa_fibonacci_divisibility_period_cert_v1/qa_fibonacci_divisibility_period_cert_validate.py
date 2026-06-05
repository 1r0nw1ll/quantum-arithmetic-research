# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Fibonacci divisibility periods: "
    "every 3rd even, every 4th mod 3, every 5th mod 5, every 6th mod 4; "
    "coprime pair counts); "
    "Theorem NT: 'par', 'tri', 'pent' are observer classification labels; "
    "causal structure is modular arithmetic over integer Fibonacci sequence; "
    "no float state, no QA orbit evolution"
)

"""
Cert [348] — QA Fibonacci Divisibility Period Laws

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol II, Chapter XV pp.107-110
  p.107 (text): "One can see that every third number is an even number. Of these,
   every other one is a 2-par number and the ones in between are all 4-par numbers.
   So every sixth number of the extended Fibonacci series is a 4-par number.
   Beginning with any number which is divisible by 3, every fourth integer is a
   3-tri number. In the same way, every fifth integer is a 5-pent number."
  p.110 (Q&A):
   Q3: "Every third number." [even numbers in Fibonacci]
   Q4: "Every fourth number." [3-tri = divisible by 3]
   Q5: "Every sixth." [4-par = divisible by 4]
   Q6: "Every fifth number." [5-pent = divisible by 5]
   Q7: Answers: "15 pairs; 13 pairs; 35 pairs."
   Q8: Answer: "18."

Five claims:
  C1: Every 3rd Fibonacci number is even (divisible by 2): F(3), F(6), F(9), ...
  C2: Every 4th Fibonacci number is divisible by 3: F(4), F(8), F(12), ...
  C3: Every 5th Fibonacci number is divisible by 5: F(5), F(10), F(15), ...
  C4: Every 6th Fibonacci number is divisible by 4: F(6), F(12), F(18), ...
  C5: Coprime pair counts: (b odd, both≤6) = 15; (b odd, both≤5) = 13;
      (b any, both≤7) = 35; (e,d with e≤d<8, gcd=1) = 18
"""

from math import gcd


def _fibonacci(n: int) -> list[int]:
    """Return first n Fibonacci numbers: 1,1,2,3,5,8,..."""
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def check_c1() -> tuple[bool, str]:
    """Every 3rd Fibonacci number is even (divisible by 2)."""
    F = _fibonacci(60)
    # F(3k) for k=1..20: F(3)=2, F(6)=8, F(9)=34, F(12)=144,...
    threes = [(k, F[3*k - 1]) for k in range(1, 21)]
    for k, fk in threes:
        assert fk % 2 == 0, f"F({3*k})={fk} is not even"
    # Conversely, F(3k+1) and F(3k+2) are odd (not even)
    for k in range(1, 20):
        assert F[3*k] % 2 == 1, f"F({3*k+1})={F[3*k]} is even (should be odd)"
        assert F[3*k + 1] % 2 == 1, f"F({3*k+2})={F[3*k+1]} is even (should be odd)"
    return True, (
        f"Every 3rd Fibonacci is even: F(3)={F[2]}, F(6)={F[5]}, F(9)={F[8]}, ...; "
        f"verified for k=1..20; no F(3k±1) is even in range"
    )


def check_c2() -> tuple[bool, str]:
    """Every 4th Fibonacci number is divisible by 3 (3-tri)."""
    F = _fibonacci(80)
    # F(4k): F(4)=3, F(8)=21, F(12)=144,...
    fours = [(k, F[4*k - 1]) for k in range(1, 21)]
    for k, fk in fours:
        assert fk % 3 == 0, f"F({4*k})={fk} not divisible by 3"
    # F(4k±1) and F(4k+2) not divisible by 3
    for k in range(1, 20):
        assert F[4*k] % 3 != 0, f"F({4*k+1})={F[4*k]} divisible by 3 (should not be)"
        assert F[4*k + 1] % 3 != 0, f"F({4*k+2})={F[4*k+1]} divisible by 3"
        assert F[4*k - 2] % 3 != 0, f"F({4*k-1})={F[4*k-2]} divisible by 3"
    return True, (
        f"Every 4th Fibonacci divisible by 3: F(4)={F[3]}, F(8)={F[7]}, F(12)={F[11]}, ...; "
        f"verified for k=1..20; no F(4k±1) or F(4k+2) divisible by 3 in range"
    )


def check_c3() -> tuple[bool, str]:
    """Every 5th Fibonacci number is divisible by 5 (5-pent)."""
    F = _fibonacci(100)
    # F(5k): F(5)=5, F(10)=55, F(15)=610,...
    fives = [(k, F[5*k - 1]) for k in range(1, 21)]
    for k, fk in fives:
        assert fk % 5 == 0, f"F({5*k})={fk} not divisible by 5"
    # F(5k±1,2) not divisible by 5
    for k in range(1, 20):
        assert F[5*k] % 5 != 0, f"F({5*k+1})={F[5*k]} divisible by 5"
        assert F[5*k + 1] % 5 != 0, f"F({5*k+2})={F[5*k+1]} divisible by 5"
        assert F[5*k - 2] % 5 != 0, f"F({5*k-1})={F[5*k-2]} divisible by 5"
        assert F[5*k - 3] % 5 != 0, f"F({5*k-2})={F[5*k-3]} divisible by 5"
    return True, (
        f"Every 5th Fibonacci divisible by 5: F(5)={F[4]}, F(10)={F[9]}, F(15)={F[14]}, ...; "
        f"verified for k=1..20; no non-multiple-of-5 index is divisible by 5 in range"
    )


def check_c4() -> tuple[bool, str]:
    """Every 6th Fibonacci number is divisible by 4 (4-par)."""
    F = _fibonacci(120)
    # F(6k): F(6)=8, F(12)=144, F(18)=2584,...
    sixes = [(k, F[6*k - 1]) for k in range(1, 21)]
    for k, fk in sixes:
        assert fk % 4 == 0, f"F({6*k})={fk} not divisible by 4"
    # F(3k) for k not mult of 2 is divisible by 2 but NOT by 4 (i.e., is 2-par)
    for k in range(1, 20):
        if k % 2 == 1:  # F(6*(2k-1)+3) = F(3*(2k+1)) — odd multiple of 3
            # F(3*(2k-1)) divisible by 2 but not 4
            idx = 3 * (2*k - 1)
            if idx <= len(F):
                assert F[idx - 1] % 2 == 0, f"F({idx}) not even"
                assert F[idx - 1] % 4 != 0, f"F({idx})={F[idx-1]} is 4-par (should be 2-par)"
    return True, (
        f"Every 6th Fibonacci divisible by 4: F(6)={F[5]}, F(12)={F[11]}, F(18)={F[17]}, ...; "
        f"F(3k) for k odd is 2-par (divisible by 2, not 4); verified for k=1..20"
    )


def check_c5() -> tuple[bool, str]:
    """Coprime pair counts: 15, 13, 35 (from Q7); 18 (from Q8)."""
    # Q7: pairs (b,e) b odd, both < 7 (≤ 6), gcd=1
    count_15 = sum(1 for b in range(1, 7, 2) for e in range(1, 7) if gcd(b, e) == 1)
    assert count_15 == 15, f"b odd, both ≤ 6: {count_15} != 15"

    # Q7: pairs (b,e) b odd, both < 6 (≤ 5), gcd=1
    count_13 = sum(1 for b in range(1, 6, 2) for e in range(1, 6) if gcd(b, e) == 1)
    assert count_13 == 13, f"b odd, both ≤ 5: {count_13} != 13"

    # Q7: pairs (b,e) b any, both < 8 (≤ 7), gcd=1
    count_35 = sum(1 for b in range(1, 8) for e in range(1, 8) if gcd(b, e) == 1)
    assert count_35 == 35, f"b any, both ≤ 7: {count_35} != 35"

    # Q8: pairs (e,d) with 1≤e≤d<8, gcd(e,d)=1 (allows e=d with gcd(d,d)=1 → d=1)
    count_18 = sum(
        1 for e in range(1, 8) for d in range(e, 8) if gcd(e, d) == 1
    )
    assert count_18 == 18, f"(e,d) with 1≤e≤d<8, gcd=1: {count_18} != 18"

    return True, (
        f"Coprime pair counts: b-odd both≤6 → {count_15}; "
        f"b-odd both≤5 → {count_13}; b-any both≤7 → {count_35}; "
        f"(e,d) 1≤e≤d<8 gcd=1 (incl e=d=1) → {count_18}"
    )


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise RuntimeError(f"cert [348] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
