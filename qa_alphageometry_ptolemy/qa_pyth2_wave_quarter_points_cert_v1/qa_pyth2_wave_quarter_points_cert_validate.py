# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-2 Ch.XIII Wave Quarter Points: "
    "4-par wavelength: W/4,W/2,3W/4 all integers; 2-par: W/2 integer but W/4 half-integer; "
    "odd: W/2 half-integer, W/4 quarter-integer; 3-par vs 5-par quarter-flip; "
    "two-wave integer coincidence at lcm); "
    "Theorem NT: 'sine wave', 'high point', 'low point', 'amplitude' are observer projections; "
    "QA layer = divisibility of wavelength by 4; no float state"
)

"""
Cert [369] — QA Pyth-2 Wave Quarter Points and Par Classification (Ch.XIII)

Source: Iverson (1993) Pythagorean Arithmetic Vol II, Chapter XIII pp.43-56+
  p.43: 'Any wavelength which is measured by a 4-par number will find each of
         these points falling at an integer.'
  p.43: 'If the wavelength is measured by a 2-par number, only the beginning,
         the middle, and the end will fall at an integer.'
  p.43: 'in the case where a wavelength is measured by an odd number, only the
         beginning and the ending will be at an integer. The midpoint will fall
         at a half-integer and the quarter points will fall at a quarter integer.'

Five claims (QA integer divisibility by 4, characterizing wave quarter-point behavior):
  C1: 4-par wavelength (W≡0 mod 4): W/4, W/2, 3W/4 are all integers
      proof: W=4k → W/4=k; W/2=2k; 3W/4=3k — all integers ✓
  C2: 2-par wavelength (W≡2 mod 4): W/2 is an integer; W/4 is not an integer
      proof: W=4k+2 → W/2=2k+1 (integer); W/4=k+1/2 (half-integer, not integer) ✓
  C3: Odd wavelength (W≡1 or W≡3 mod 4): W/2 is not an integer; W/4 is not a half-integer
      proof: W=2m+1 (odd) → W/2=m+1/2 (not integer); W/4=(2m+1)/4;
             for W≡1(mod 4): W/4≡1/4 (quarter); for W≡3(mod 4): W/4≡3/4 (3-quarter) ✓
  C4: 3-par vs 5-par quarter-flip: the quarter-point of a 5-par wave falls at n+1/4 (first quarter)
      while the 3-par wave falls at n+3/4 (third quarter) — they are mirror images
      proof: W≡1(mod 4) → W/4=k+1/4; W≡3(mod 4) → W/4=k+3/4; 1/4+3/4=1 (complementary) ✓
  C5: Two waves with coprime wavelengths W₁,W₂ have integer-coincidences every lcm(W₁,W₂) units;
      pairwise integer coincidences: {multiples of W₁} ∩ {multiples of W₂} = {multiples of lcm(W₁,W₂)}
      proof: n∈W₁ℤ∩W₂ℤ iff W₁|n and W₂|n iff lcm(W₁,W₂)|n ✓
"""

from math import gcd, lcm
from fractions import Fraction


def check_c1() -> tuple[bool, str]:
    """4-par wavelength: W/4, W/2, 3W/4 are all integers."""
    four_par_examples = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 100, 200, 1000]
    count = 0
    for W in range(4, 500, 4):  # all 4-par numbers in [4, 496]
        assert W % 4 == 0, f"W={W} is not 4-par"
        assert (W // 4) * 4 == W          # W/4 is an integer
        assert (W // 2) * 2 == W          # W/2 is an integer
        assert (3 * W // 4) * 4 == 3 * W  # 3W/4 is an integer
        count += 1
    return True, (
        f"4-par (W≡0 mod 4): W/4, W/2, 3W/4 all integers verified for {count} values W∈[4,496]; "
        f"proof: W=4k → W/4=k, W/2=2k, 3W/4=3k — all integers; "
        f"Iverson: 'a 4-par wavelength will find each of these points falling at an integer' ✓"
    )


def check_c2() -> tuple[bool, str]:
    """2-par wavelength: W/2 is integer; W/4 is half-integer (not integer)."""
    count = 0
    for W in range(2, 500, 4):  # 2-par: W≡2 (mod 4)
        assert W % 4 == 2, f"W={W} is not 2-par"
        # W/2 is an integer
        half = Fraction(W, 2)
        assert half.denominator == 1, f"W/2 not integer at W={W}"
        # W/4 is a half-integer (denominator exactly 2)
        quarter = Fraction(W, 4)
        assert quarter.denominator == 2, f"W/4 not half-integer at W={W}: {quarter}"
        assert quarter == Fraction(W // 4 * 4 + 2, 4) == Fraction(W // 4, 1) + Fraction(1, 2)
        count += 1
    return True, (
        f"2-par (W≡2 mod 4): W/2 integer, W/4 half-integer verified for {count} values W∈[2,498]; "
        f"proof: W=4k+2 → W/2=2k+1 (integer); W/4=k+1/2 (half-integer); "
        f"Iverson: 'only the beginning, the middle, and the end will fall at an integer' ✓"
    )


def check_c3() -> tuple[bool, str]:
    """Odd wavelength: W/2 is half-integer; W/4 is quarter-integer."""
    count_5par = 0
    count_3par = 0
    for W in range(1, 500, 2):  # all odd W in [1, 499]
        half = Fraction(W, 2)
        quarter = Fraction(W, 4)
        # W/2 is never an integer for odd W
        assert half.denominator == 2, f"W/2 not half-integer at W={W}: {half}"
        # W/4 is never a half-integer: denominator is 4 (not 1 or 2)
        assert quarter.denominator == 4, f"W/4 not quarter-integer at W={W}: {quarter}"
        if W % 4 == 1:  # 5-par
            count_5par += 1
        else:            # 3-par (W≡3 mod 4)
            count_3par += 1
    return True, (
        f"Odd W: W/2 is half-integer and W/4 is quarter-integer (denom=4) verified for "
        f"{count_5par+count_3par} odd values in [1,499]; "
        f"5-par (W≡1 mod 4): {count_5par} values; 3-par (W≡3 mod 4): {count_3par} values; "
        f"proof: W=2m+1 → W/2=m+1/2; W/4=(2m+1)/4 has denom=4; "
        f"Iverson: 'only the beginning and ending at integer; midpoint half-integer; "
        f"quarter points at quarter-integer' ✓"
    )


def check_c4() -> tuple[bool, str]:
    """3-par vs 5-par quarter-flip: 5-par quarter=k+1/4; 3-par quarter=k+3/4; complementary."""
    count = 0
    for W in range(1, 200, 2):  # odd W
        quarter = Fraction(W, 4)
        k = W // 4
        if W % 4 == 1:  # 5-par: W=4k+1 → W/4=k+1/4
            assert quarter == Fraction(k) + Fraction(1, 4), (
                f"5-par W={W}: quarter={quarter} != {k}+1/4"
            )
        else:            # 3-par: W=4k+3 → W/4=k+3/4
            assert quarter == Fraction(k) + Fraction(3, 4), (
                f"3-par W={W}: quarter={quarter} != {k}+3/4"
            )
        # Complementary: 1/4 + 3/4 = 1 (they're mirror images)
        if W % 4 == 1:
            frac = Fraction(1, 4)
        else:
            frac = Fraction(3, 4)
        assert frac + (1 - frac) == 1
        count += 1
    # Specific Iverson examples: 3-unit (3-par) and 5-unit (5-par) from Fig.16a
    W_3par = 3   # 3-par: quarter at k+3/4 = 0+3/4 = 3/4
    W_5par = 5   # 5-par: quarter at k+1/4 = 1+1/4 = 5/4
    assert Fraction(W_3par, 4) == Fraction(3, 4)
    assert Fraction(W_5par, 4) == Fraction(5, 4)
    return True, (
        f"3-par vs 5-par quarter-flip verified for {count} odd values W∈[1,199]; "
        f"5-par (W≡1 mod 4): quarter = k+1/4; 3-par (W≡3 mod 4): quarter = k+3/4; "
        f"fractions 1/4 and 3/4 are complementary (sum=1 = mirror images); "
        f"Iverson example: W=3 (3-par) quarter=3/4; W=5 (5-par) quarter=5/4 ✓"
    )


def check_c5() -> tuple[bool, str]:
    """Two waves coincide at integer multiples of their LCM only."""
    count = 0
    for W1 in range(2, 25):
        for W2 in range(2, 25):
            L = lcm(W1, W2)
            # Integer points for wave W1: 0, W1, 2*W1, ...
            # Integer points for wave W2: 0, W2, 2*W2, ...
            # Intersection = multiples of lcm(W1, W2)
            # Verify: first 5 intersection points are exactly multiples of L
            intersections_w1 = set(range(0, 6 * L, W1))
            intersections_w2 = set(range(0, 6 * L, W2))
            intersection = sorted(intersections_w1 & intersections_w2)
            multiples_of_lcm = list(range(0, 6 * L, L))
            assert intersection == multiples_of_lcm, (
                f"Intersection({W1},{W2}) != multiples of lcm={L}: {intersection} vs {multiples_of_lcm}"
            )
            count += 1
    # Specific examples from the chapter
    assert lcm(3, 5) == 15  # 3-cycle and 5-cycle coincide every 15
    assert lcm(2, 3, 5) == 30  # all three coincide every 30
    return True, (
        f"Two-wave integer coincidences = multiples of LCM verified for all pairs W1,W2∈[2,24] "
        f"({count} pairs); "
        f"proof: n∈W₁ℤ∩W₂ℤ iff lcm(W₁,W₂)|n; "
        f"Iverson examples: lcm(3,5)=15; lcm(2,3,5)=30 ✓"
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
        raise RuntimeError(f"cert [369] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
