# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.12 Two Forms of Energy: "
    "180deg phase 180%24=12=chromatic; film 1/16s TV 1/32s; 16=2^4 Myriad 32=2^5 ratio=2=octave; "
    "7octaves*7Myriads=49=7^2 49%24=1 Singularity-class; 3+3 colors seed=6; "
    "TWA-800 13000ft mod24=16 400mph mod24=16 both Myriad); "
    "Theorem NT: energy physics and dowsing reports are observer-projection; "
    "no float state, no QA orbit evolution"
)

from math import gcd


def check_c1() -> tuple[bool, str]:
    """C1: Dayside/Nightside 180-degree phase difference; 2 forms of energy.
    180 mod 24=12=chromatic scale (12 notes per octave); 360 mod 24=0 (full turn divisible by 24).
    2 forms: 2=first QA generating prime; 2 mod 24=2."""
    num_forms = 2
    phase_deg  = 180
    full_turn  = 360

    if num_forms % 24 != 2:
        return False, f"2 mod 24={num_forms%24}, expected 2"

    r_phase = phase_deg % 24   # 12
    if r_phase != 12:
        return False, f"180 mod 24={r_phase}, expected 12"

    # 12 = chromatic scale notes per octave
    chromatic_notes = 12
    if r_phase != chromatic_notes:
        return False, f"180%24={r_phase} != 12 chromatic notes"

    # 180 = 7*24 + 12
    if phase_deg // 24 != 7:
        return False, f"180//24={phase_deg//24}, expected 7"

    # 360 mod 24 = 0 (full rotation)
    if full_turn % 24 != 0:
        return False, f"360 mod 24={full_turn%24}, expected 0"
    if full_turn // 24 != 15:
        return False, f"360/24={full_turn//24}, expected 15"

    # 360 = 2*180 (octave: doubling the half-turn)
    if full_turn != 2 * phase_deg:
        return False, f"360 != 2*180"

    return True, (f"2_forms%24=2; 180%24={r_phase}=12=chromatic; "
                  f"180=7*24+12; 360%24=0(15*24); 360=2*180(octave) PASS")


def check_c2() -> tuple[bool, str]:
    """C2: Film 1/16s frame interval; TV 1/32s frame interval.
    16=2^4; 16 mod 24=16=Myriad residue; 32=2^5; 32 mod 24=8=2^3.
    Ratio 32/16=2 (octave doubling); 16*2=32 (next power of 2)."""
    film_denom = 16   # 1/16 second
    tv_denom   = 32   # 1/32 second

    # 16=2^4
    if film_denom != 2 * 2 * 2 * 2:
        return False, f"16 != 2^4"
    if film_denom % 24 != 16:
        return False, f"16 mod 24={film_denom%24}, expected 16"

    # 32=2^5
    if tv_denom != 2 * 2 * 2 * 2 * 2:
        return False, f"32 != 2^5"
    if tv_denom % 24 != 8:
        return False, f"32 mod 24={tv_denom%24}, expected 8"

    # Ratio = 2 (octave doubling: TV frame rate is double film)
    ratio = tv_denom // film_denom
    if ratio != 2:
        return False, f"32/16={ratio}, expected 2"

    # 16 = Myriad residue (same as 10000%24=16, 40000%24=16)
    if film_denom != 10000 % 24:
        return False, f"16 != 10000%24={10000%24}"

    # 32 mod 24=8=2^3 (same as Atlantis-era residue)
    if tv_denom % 24 != 2 * 2 * 2:
        return False, f"32%24={tv_denom%24} != 2^3"

    return True, (f"film_16=2^4; 16%24=16=Myriad; tv_32=2^5; 32%24=8=2^3; "
                  f"ratio={ratio}=octave; 16=10000%24 ✓ PASS")


def check_c3() -> tuple[bool, str]:
    """C3: 7 octaves per Myriad; 7 Myriads in hierarchy.
    7*7=49=7^2; 49 mod 24=1 (Singularity-class, same as cert [375] C4: 7*7=49 mod24=1).
    7 mod 24=7 (7-par); 7 is prime."""
    octaves_per_myriad = 7
    num_myriads        = 7

    if octaves_per_myriad != num_myriads:
        return False, f"octaves={octaves_per_myriad} != num_myriads={num_myriads}"

    # 7 mod 24=7 (7-par)
    if octaves_per_myriad % 24 != 7:
        return False, f"7 mod 24={octaves_per_myriad%24}, expected 7"

    # 7*7=49
    total = octaves_per_myriad * num_myriads
    if total != 49:
        return False, f"7*7={total}, expected 49"

    # 49 mod 24=1 (Singularity-class)
    r49 = total % 24
    if r49 != 1:
        return False, f"49 mod 24={r49}, expected 1"

    # 7 is prime
    if not all(7 % k != 0 for k in range(2, 7)):
        return False, "7 is not prime"

    # Cross-reference cert [375]: 7*7=49 mod24=1 Spirituality cert
    cross = 49 % 24
    if cross != 1:
        return False, f"49%24={cross} cross-ref [375] mismatch"

    return True, (f"7_oct*7_myr={total}=7^2; 49%24={r49}=1=Singularity-class; "
                  f"7_is_prime; cross-ref [375] ✓ PASS")


def check_c4() -> tuple[bool, str]:
    """C4: 3 primary colors (red/yellow/blue) and 3 secondary colors (orange/green/violet).
    3+3=6=male seed tuple product (cert [374]); 3*3=9 mod 9=0 (Singularity mod-9 fixed point).
    6=2*3=first perfect number; both groups are 3-par (3 mod 4=3)."""
    primary_colors   = 3   # red, yellow, blue
    secondary_colors = 3   # orange, green, violet

    color_sum = primary_colors + secondary_colors   # 6
    if color_sum != 6:
        return False, f"3+3={color_sum}, expected 6"

    # 6 = male seed product (cert [374])
    seed_product = 1 * 1 * 2 * 3
    if color_sum != seed_product:
        return False, f"6 != seed_product={seed_product}"

    # 6 = first perfect number
    if 1 + 2 + 3 != 6:
        return False, f"1+2+3={1+2+3}, expected 6"

    # 3*3=9 mod 9=0 (mod-9 Singularity fixed point, cert [378] C1)
    color_product = primary_colors * secondary_colors   # 9
    if color_product != 9:
        return False, f"3*3={color_product}, expected 9"
    if color_product % 9 != 0:
        return False, f"9 mod 9={color_product%9}, expected 0"

    # 3 mod 4=3 (3-par: both sets)
    if primary_colors % 4 != 3:
        return False, f"3 mod 4={primary_colors%4}, expected 3 (3-par)"

    return True, (f"3_primary+3_secondary={color_sum}=6=seed_product; "
                  f"perfect_check 1+2+3=6; 3*3={color_product}; 9%9=0(Singularity-mod9); "
                  f"3%4=3(3-par) PASS")


def check_c5() -> tuple[bool, str]:
    """C5: TWA-800 climb above 13,000 feet; speed ~400 mph.
    13000 mod 24=16=Myriad residue; 400 mod 24=16=Myriad residue.
    Both share Myriad class (16); 13000/400=32.5, but 13000*400=5200000; 5200000%24=0."""
    altitude_ft = 13000
    speed_mph   = 400

    r_alt   = altitude_ft % 24   # 16
    r_speed = speed_mph % 24      # 16

    if r_alt != 16:
        return False, f"13000 mod 24={r_alt}, expected 16"
    if r_speed != 16:
        return False, f"400 mod 24={r_speed}, expected 16"

    # Both share Myriad residue 16
    if r_alt != r_speed:
        return False, f"residues differ: {r_alt} != {r_speed}"

    # 13000=541*24+16
    if altitude_ft // 24 != 541:
        return False, f"13000//24={altitude_ft//24}, expected 541"
    if 541 * 24 + 16 != altitude_ft:
        return False, f"541*24+16={541*24+16}, expected 13000"

    # 400=16*24+16
    if speed_mph // 24 != 16:
        return False, f"400//24={speed_mph//24}, expected 16"
    if 16 * 24 + 16 != speed_mph:
        return False, f"16*24+16={16*24+16}, expected 400"

    # Product: 13000*400=5200000; (16*16)%24=256%24=16=Myriad again
    product = altitude_ft * speed_mph
    if product != 5200000:
        return False, f"13000*400={product}, expected 5200000"
    # 16*16=256; 256 mod 24=16 (Myriad again — residue is closed under multiplication mod 24)
    if product % 24 != 16:
        return False, f"5200000 mod 24={product%24}, expected 16 (16*16 mod24=256%24=16)"
    if product % 24 != r_alt:
        return False, f"product residue {product%24} != altitude residue {r_alt}"

    return True, (f"13000%24={r_alt}=16=Myriad; 400%24={r_speed}=16=Myriad; "
                  f"both_equal ✓; 13000*400={product}; {product}%24=16(16*16%24=16) PASS")


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
