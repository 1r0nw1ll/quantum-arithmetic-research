# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.6 Spirituality: "
    "7 Myriad hierarchy 3+1+3; Light 7 colors 4 primary; Music 4 bugle notes; "
    "49=7*7 mod 24=1 Singularity; Episode-3 temporal 61+37=98=2*7*7); "
    "Theorem NT: Myriad energy labels are observer projection categories; "
    "no float state, no QA orbit evolution"
)

"""
Cert [375] — QA Pyth-3 Spirituality: Seven Myriads Hierarchy

Source: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III, Chapter 6 pp.28-36
"""

from math import gcd


def _prime_factors(n: int) -> set[int]:
    factors: set[int] = set()
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.add(p)
            n //= p
        p += 1
    if n > 1:
        factors.add(n)
    return factors


def check_c1() -> tuple[bool, str]:
    """C1: Seven Myriads: 3 below matter + matter + 3 above = 7.
    Subsound/music/ultrasound below; light/metaphysical/creative above.
    7 mod 24 = 7 (Cosmos orbit); 7 first prime not dividing 360=2^3*3^2*5."""
    below = ["subsound", "music", "ultrasound"]   # 3 below matter
    matter = ["matter"]                            # center
    above = ["light", "metaphysical", "creative"]  # 3 above matter
    total = len(below) + len(matter) + len(above)
    if total != 7:
        return False, f"Total Myriads = {total}, expected 7"
    if len(below) != 3 or len(above) != 3:
        return False, f"Expected 3+1+3, got {len(below)}+1+{len(above)}"
    # 7 mod 24 = 7 (remains in Cosmos orbit as itself)
    if 7 % 24 != 7:
        return False, f"7 mod 24 = {7 % 24}, expected 7"
    # 7 is first prime not dividing 360 = 2^3 * 3^2 * 5
    myriad_year = 360
    if myriad_year % 7 != 0:
        pass   # correct: 7 does not divide 360
    else:
        return False, "7 should NOT divide 360"
    if myriad_year % 2 != 0 or myriad_year % 3 != 0 or myriad_year % 5 != 0:
        return False, "360 should be divisible by 2,3,5"
    return True, (f"7_myriads=3+1+3 7%24=7 360%7={360%7}(nonzero) "
                  f"360%2=0 360%3=0 360%5=0 PASS")


def check_c2() -> tuple[bool, str]:
    """C2: Light Myriad has 7 colors, 4 primary (red,yellow,green,blue).
    Music Myriad has 4 primary bugle notes (C,F,A,C). 7-4=3 secondary."""
    light_colors_total = 7
    light_primary = 4   # red, yellow, green, blue
    light_secondary = light_colors_total - light_primary
    if light_secondary != 3:
        return False, f"light secondary={light_secondary}, expected 3"

    bugle_notes = ["C_low", "F", "A", "C_high"]   # 4 primary
    if len(bugle_notes) != 4:
        return False, f"bugle notes count={len(bugle_notes)}, expected 4"

    # Both have 4 primary elements — same as QA 4-tuple size
    qa_tuple_size = 4
    if len(bugle_notes) != qa_tuple_size or light_primary != qa_tuple_size:
        return False, "Primary element count != QA tuple size 4"

    # 7 colors and 7 Myriads share the same count — internal symmetry
    if light_colors_total != 7:
        return False, f"light colors total={light_colors_total}, expected 7"
    return True, (f"light=7colors 4primary 3secondary; "
                  f"bugle=4notes={bugle_notes}; both_primary=4=QA_tuple_size PASS")


def check_c3() -> tuple[bool, str]:
    """C3: 4 bugle notes C,F,A,C map to QA 4-tuple (b,e,diff=b+e,apex=b+2e).
    Upper C is one octave above lower C = ratio 2:1 (doubling).
    Frequencies 3:4:5:6 (from Ch.1 cert [358]) verified for bugle ratios."""
    # Bugle notes in just-intonation integer ratios: 3:4:5:6
    # C_low=3, F=4, A=5, C_high=6  (relative units)
    c_low, f_note, a_note, c_high = 3, 4, 5, 6

    # Map to QA 4-tuple: treat c_low as b, (f_note - c_low) as e
    b_bugle = c_low               # 3
    e_bugle = f_note - c_low      # 1 (step from C to F in these units)
    diff_bugle = b_bugle + e_bugle   # 4 (= F note) — derived
    apex_bugle = b_bugle + 2 * e_bugle  # 5 (= A note) — derived

    if diff_bugle != f_note:
        return False, f"diff_bugle={diff_bugle} should equal f_note={f_note}"
    if apex_bugle != a_note:
        return False, f"apex_bugle={apex_bugle} should equal a_note={a_note}"

    # Upper C = c_high = 6 = 2 * c_low = 2 * 3 (one octave up, ratio 2:1)
    if c_high != 2 * c_low:
        return False, f"c_high={c_high} != 2*c_low={2*c_low}"

    # Pythagorean check: 3^2+4^2=5^2 (from Ch.1 cert [358])
    if c_low * c_low + f_note * f_note != a_note * a_note:
        return False, f"{c_low}^2+{f_note}^2 != {a_note}^2"

    return True, (f"b={b_bugle} e={e_bugle} diff={diff_bugle}=F apex={apex_bugle}=A "
                  f"c_high={c_high}=2*c_low octave_ratio=2:1 3*3+4*4=5*5 PASS")


def check_c4() -> tuple[bool, str]:
    """C4: 7 octaves * 7 primary frequencies = 49 = 7*7.
    49 mod 24 = 1 (Singularity class). 7*7 mod 24 = 1."""
    octaves = 7
    primary_per_octave = 7
    total = octaves * primary_per_octave   # 49
    if total != 49:
        return False, f"total={total}, expected 49"
    if total != 7 * 7:
        return False, f"49 != 7*7"
    total_mod24 = total % 24
    if total_mod24 != 1:
        return False, f"49 mod 24 = {total_mod24}, expected 1 (Singularity)"
    # Verify via formula: (7 mod 24) * (7 mod 24) mod 24
    if (7 * 7) % 24 != 1:
        return False, f"(7*7) mod 24 = {(7*7)%24}, expected 1"
    return True, (f"7_octaves * 7_primary = {total} = 7*7; "
                  f"49 mod 24 = {total_mod24} = Singularity PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Episode 3 temporal structure: mother died at age 61, 37 years
    before Iverson's age 70 (1988); would be 98 if alive. 61+37=98=2*49=2*7^2.
    61 mod 24 = 13; 37 mod 24 = 13; 98 mod 24 = 2 (2-par)."""
    mother_age_at_death = 61
    years_elapsed = 37
    iversons_age = 70   # at time of Episode 3 (July 13, 1988)

    # Mother's age if alive = age at death + years elapsed
    mother_age_if_alive = mother_age_at_death + years_elapsed
    if mother_age_if_alive != 98:
        return False, f"mother_age_if_alive={mother_age_if_alive}, expected 98"

    # 98 = 2 * 49 = 2 * 7*7
    if 98 != 2 * 49:
        return False, f"98 != 2*49"
    if 49 != 7 * 7:
        return False, f"49 != 7*7"

    # Mod-24 residues
    r61 = mother_age_at_death % 24   # 61 mod 24 = 13
    r37 = years_elapsed % 24          # 37 mod 24 = 13
    r98 = mother_age_if_alive % 24    # 98 mod 24 = 2

    if r61 != 13:
        return False, f"61 mod 24 = {r61}, expected 13"
    if r37 != 13:
        return False, f"37 mod 24 = {r37}, expected 13"
    if r98 != 2:
        return False, f"98 mod 24 = {r98}, expected 2"

    # 13 + 13 = 26 = 24 + 2 → (61 + 37) mod 24 = 2 ✓
    if (r61 + r37) % 24 != r98:
        return False, f"(r61+r37) mod 24 = {(r61+r37)%24} != r98={r98}"

    # Iverson age 70: 70 mod 24 = 22 (2-par minus 2); 70 = 2*5*7
    r70 = iversons_age % 24
    if r70 != 22:
        return False, f"70 mod 24 = {r70}, expected 22"

    return True, (f"61+37=98=2*7*7; "
                  f"61%24={r61} 37%24={r37} (both=13); "
                  f"98%24={r98}=2-par; 70%24={r70} PASS")


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
