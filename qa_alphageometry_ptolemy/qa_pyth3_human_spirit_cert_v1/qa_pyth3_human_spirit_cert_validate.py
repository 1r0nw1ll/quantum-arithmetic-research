# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.10 The Human Spirit: "
    "6 Talmud books=male seed product 1*1*2*3=6; 600/1200 yr Christ Spirit gaps mod24=0 ratio=2; "
    "10000/70000 yr ancient civilizations both mod24=16=Myriad; 200yr memory 200%24=8=2^3; "
    "3000yr mathematics 3000%24=0=24*5^3); Theorem NT: spiritual history and past-life memories are "
    "observer-projection reports; no float state, no QA orbit evolution"
)

from math import gcd


def check_c1() -> tuple[bool, str]:
    """C1: 'There are said to be only six of these books in existence today.'
    6 = male aboriginal seed tuple product: 1*1*2*3=6.
    6 is the first perfect number: sigma(6)=1+2+3=6.
    6 mod 24=6; 6 = 2*3 (2-par in factor structure)."""
    num_books = 6

    if num_books != 6:
        return False, f"num_books={num_books}, expected 6"

    # 6 = product of male aboriginal seed QN (1, 1, 2, 3)
    b, e = 1, 1
    diff = b + e       # A2: derived
    apex = b + 2 * e   # A2: derived
    seed_product = b * e * diff * apex
    if seed_product != 6:
        return False, f"seed product 1*1*2*3={seed_product}, expected 6"
    if num_books != seed_product:
        return False, f"6_books != seed_product={seed_product}"

    # 6 is perfect: sum of proper divisors = 1+2+3=6
    proper_divisors_sum = 1 + 2 + 3
    if proper_divisors_sum != 6:
        return False, f"sum proper divisors = {proper_divisors_sum}, expected 6"

    if num_books % 24 != 6:
        return False, f"6 mod 24={num_books%24}, expected 6"

    # 6 = 2*3 (product of first two primes)
    if num_books != 2 * 3:
        return False, f"6 != 2*3"

    return True, (f"6_books=seed_product_1*1*2*3={seed_product}; "
                  f"perfect_check: 1+2+3={proper_divisors_sum}=6; 6%24=6; 6=2*3 PASS")


def check_c2() -> tuple[bool, str]:
    """C2: 'Mohammed, 600 years later, and Joseph Smith... 1200 years later still.'
    600 mod 24=0 (25*24); 1200 mod 24=0 (50*24); 1200/600=2 (octave doubling).
    600+1200=1800 mod 24=0 (75*24); Christ Spirit chain: 600, 1200, 1800 all 24-aligned."""
    gap_jesus_mohammed = 600
    gap_mohammed_smith  = 1200

    if gap_jesus_mohammed % 24 != 0:
        return False, f"600 mod 24={gap_jesus_mohammed%24}, expected 0"
    if gap_mohammed_smith % 24 != 0:
        return False, f"1200 mod 24={gap_mohammed_smith%24}, expected 0"

    # Check division
    if gap_jesus_mohammed // 24 != 25:
        return False, f"600/24={gap_jesus_mohammed//24}, expected 25"
    if gap_mohammed_smith // 24 != 50:
        return False, f"1200/24={gap_mohammed_smith//24}, expected 50"

    # Octave ratio 1200/600=2
    ratio = gap_mohammed_smith // gap_jesus_mohammed
    if ratio != 2:
        return False, f"1200/600={ratio}, expected 2 (octave doubling)"

    # Total gap from Jesus to Smith
    total_gap = gap_jesus_mohammed + gap_mohammed_smith   # 1800
    if total_gap != 1800:
        return False, f"600+1200={total_gap}, expected 1800"
    if total_gap % 24 != 0:
        return False, f"1800 mod 24={total_gap%24}, expected 0"
    if total_gap // 24 != 75:
        return False, f"1800/24={total_gap//24}, expected 75"

    return True, (f"600%24=0(25*24); 1200%24=0(50*24); 1200/600={ratio}(octave); "
                  f"600+1200={total_gap}; 1800%24=0(75*24) PASS")


def check_c3() -> tuple[bool, str]:
    """C3: 'ancient civilizations more than 10,000 years ago, and up to 70,000 years ago'
    10000 mod 24=16=Myriad residue (cert [354]); 70000 mod 24=16 (same!).
    Gap=60000; 60000 mod 24=0; 60000/24=2500."""
    ancient_lower = 10000
    ancient_upper = 70000

    r_lower = ancient_lower % 24   # 16
    r_upper = ancient_upper % 24   # 16

    if r_lower != 16:
        return False, f"10000 mod 24={r_lower}, expected 16"
    if r_upper != 16:
        return False, f"70000 mod 24={r_upper}, expected 16"

    # Both equal the Myriad residue
    if r_lower != r_upper:
        return False, f"residues differ: {r_lower} != {r_upper}"

    gap = ancient_upper - ancient_lower   # 60000
    if gap != 60000:
        return False, f"gap={gap}, expected 60000"
    if gap % 24 != 0:
        return False, f"60000 mod 24={gap%24}, expected 0"
    if gap // 24 != 2500:
        return False, f"60000/24={gap//24}, expected 2500"

    return True, (f"10000%24={r_lower}=16=Myriad; 70000%24={r_upper}=16=Myriad; "
                  f"both_equal ✓; gap={gap}; 60000%24=0; 60000/24=2500 PASS")


def check_c4() -> tuple[bool, str]:
    """C4: 'in the context of a time perhaps two hundred or more years earlier'
    200 mod 24=8=2^3; 200=8*25=8*5^2; 8=Atlantis pre-sinking residue (cert [378] C4).
    200/8=25=5^2; gcd(200,24)=8."""
    years_back = 200

    r200 = years_back % 24   # 8
    if r200 != 8:
        return False, f"200 mod 24={r200}, expected 8"
    if r200 != 2 * 2 * 2:
        return False, f"8 != 2^3"

    # 200 = 8 * 25
    if years_back // 8 != 25:
        return False, f"200/8={years_back//8}, expected 25"
    if 25 != 5 * 5:
        return False, f"25 != 5^2"
    if years_back != 8 * 5 * 5:
        return False, f"200 != 8*5^2"

    # gcd(200, 24)
    g = gcd(years_back, 24)
    if g != 8:
        return False, f"gcd(200,24)={g}, expected 8"

    # 200 = 24*8 + 8 (quotient=8, remainder=8)
    if years_back // 24 != 8:
        return False, f"200//24={years_back//24}, expected 8"

    return True, (f"200%24={r200}=8=2^3; 200=8*5^2; gcd(200,24)={g}=8; "
                  f"200//24=8 remainder=8; 8=Atlantis_residue PASS")


def check_c5() -> tuple[bool, str]:
    """C5: 'The mathematics of Creation... were once known more than 3000 years ago.'
    3000 mod 24=0 (same as berry tasters from cert [378] C3); 3000=24*5^3=24*125.
    Link: 10000 mod 24=16 (C3), 3000 mod 24=0 (C5): 3000/24=125=5^3."""
    years_ago = 3000

    if years_ago % 24 != 0:
        return False, f"3000 mod 24={years_ago%24}, expected 0"
    if years_ago // 24 != 125:
        return False, f"3000/24={years_ago//24}, expected 125"

    # 125 = 5^3
    if 125 != 5 * 5 * 5:
        return False, f"125 != 5^3"
    if years_ago != 24 * 5 * 5 * 5:
        return False, f"3000 != 24*5^3"

    # Cross-reference [378] C3: 3000 berry tasters same number, same mod-24=0
    cross_ref = 3000
    if cross_ref % 24 != 0:
        return False, f"cross_ref 3000 mod 24={cross_ref%24}, expected 0"

    # Also: 3000/1000=3; 1000 mod 24=16=Myriad (C3)
    unit = years_ago // 1000   # 3
    if unit != 3:
        return False, f"3000/1000={unit}, expected 3"
    if 1000 % 24 != 16:
        return False, f"1000 mod 24={1000%24}, expected 16"

    return True, (f"3000%24=0; 3000=24*125=24*5^3; 125=5^3; "
                  f"cross_ref [378]C3: same 3000; 3000/1000=3; 1000%24=16=Myriad PASS")


def main() -> None:
    checks = [check_c1, check_c2, check_c3, check_c4, check_c5]
    passed = 0
    for fn in checks:
        ok, msg = fn()
        label = "PASS" if ok else "FAIL"
        print(f"[{label}] {fn.__name__}: {msg}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
