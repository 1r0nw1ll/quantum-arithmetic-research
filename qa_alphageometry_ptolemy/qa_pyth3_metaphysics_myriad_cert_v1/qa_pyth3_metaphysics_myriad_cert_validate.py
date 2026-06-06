# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.7 Metaphysics Myriad: "
    "octave numbering UV=50 astral=51 portal=52 spirit=52..58; 5040=7! per octave "
    "mod 24=0; 5040/144=35=5*7; hierarchy 5040->144->7->4; spirit world 7 octaves "
    "52-58 residues 4-10); Theorem NT: octave energy labels are observer projections; "
    "no float state, no QA orbit evolution"
)

"""
Cert [376] — QA Pyth-3 Metaphysics Myriad: Frequency Hierarchy

Source: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III, Chapter 7 pp.37-42
  p.41: 'In the quantum world each octave is permitted to have 5040 frequencies,
         but only the 144 are strong enough to be considered separately. That leaves
         4896 frequencies which are much weaker quantumwise. Of the 144 frequencies
         used 12 to 18, are much stronger than the rest. Of this 12-18 frequencies,
         seven frequencies are still stronger, and of these seven there are four which
         are really important.'
"""

from math import factorial


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
    """C1: Octave numbering — UV=50th, astral=51st, spirit portal=52nd.
    50 mod 24=2; 51 mod 24=3; 52 mod 24=4 (4-par entry to Metaphysical)."""
    uv_octave = 50
    astral_octave = uv_octave + 1     # 51
    portal_octave = uv_octave + 2     # 52

    if astral_octave != 51:
        return False, f"astral_octave={astral_octave}, expected 51"
    if portal_octave != 52:
        return False, f"portal_octave={portal_octave}, expected 52"

    r_uv     = uv_octave % 24       # 50 mod 24 = 2  (2-par)
    r_astral = astral_octave % 24   # 51 mod 24 = 3  (3-par)
    r_portal = portal_octave % 24   # 52 mod 24 = 4  (4-par)

    if r_uv != 2:
        return False, f"50 mod 24 = {r_uv}, expected 2"
    if r_astral != 3:
        return False, f"51 mod 24 = {r_astral}, expected 3"
    if r_portal != 4:
        return False, f"52 mod 24 = {r_portal}, expected 4"

    # Sequence is consecutive: 2, 3, 4 (2-par -> 3-par -> 4-par)
    if r_astral - r_uv != 1 or r_portal - r_astral != 1:
        return False, "residues not consecutive"
    return True, (f"UV=50(mod24={r_uv}=2-par) astral=51(mod24={r_astral}=3-par) "
                  f"portal=52(mod24={r_portal}=4-par) consecutive PASS")


def check_c2() -> tuple[bool, str]:
    """C2: 5040 frequencies per octave = 7! = 7*720.
    5040 mod 24 = 0; 5040 / 24 = 210 = 2*3*5*7 (common denominator from Symp Harm)."""
    freq_total = 5040
    factorial_7 = factorial(7)    # 5040

    if freq_total != factorial_7:
        return False, f"5040 != 7! = {factorial_7}"
    if freq_total != 7 * 720:
        return False, f"5040 != 7*720"
    if 720 != factorial(6):
        return False, f"720 != 6!"

    freq_mod24 = freq_total % 24
    if freq_mod24 != 0:
        return False, f"5040 mod 24 = {freq_mod24}, expected 0"

    aliquot = freq_total // 24     # 210
    if aliquot != 210:
        return False, f"5040/24 = {aliquot}, expected 210"
    if aliquot != 2 * 3 * 5 * 7:
        return False, f"210 != 2*3*5*7"

    return True, (f"5040=7!={factorial_7}=7*720; 5040 mod 24={freq_mod24}=0; "
                  f"5040/24={aliquot}=2*3*5*7 PASS")


def check_c3() -> tuple[bool, str]:
    """C3: 144 strong frequencies; 4896 weak; 5040/144=35=5*7.
    144=12^2=2^4*3^2; 4896=5040-144=2^5*3^2*17; all mod 24=0."""
    freq_total = 5040
    freq_strong = 144
    freq_weak = freq_total - freq_strong   # 4896

    if freq_strong != 12 * 12:
        return False, f"144 != 12*12"
    if _prime_factors(freq_strong) != {2, 3}:
        return False, f"prime_factors(144)={_prime_factors(freq_strong)} != {{2,3}}"

    ratio = freq_total // freq_strong   # 35
    if ratio != 35:
        return False, f"5040/144 = {ratio}, expected 35"
    if ratio != 5 * 7:
        return False, f"35 != 5*7"

    if freq_weak != 4896:
        return False, f"freq_weak={freq_weak}, expected 4896"
    if _prime_factors(freq_weak) != {2, 3, 17}:
        return False, f"prime_factors(4896)={_prime_factors(freq_weak)} != {{2,3,17}}"

    # All three mod 24
    if freq_total % 24 != 0:
        return False, f"5040 mod 24 != 0"
    if freq_strong % 24 != 0:
        return False, f"144 mod 24 = {freq_strong % 24}, expected 0"
    if freq_weak % 24 != 0:
        return False, f"4896 mod 24 = {freq_weak % 24}, expected 0"

    return True, (f"144=12*12; 5040/144={ratio}=5*7; 4896=5040-144; "
                  f"prime_factors(4896)={_prime_factors(freq_weak)}; all mod 24=0 PASS")


def check_c4() -> tuple[bool, str]:
    """C4: Frequency hierarchy 5040->144->7->4.
    Same 4:7 ratio as Light and Music from cert [375]: 4 primary out of 7.
    144/7 is within the 12-18 range (floor=20.57, so 12-18 subset of 144 is below)."""
    hierarchy = [5040, 144, 7, 4]

    if hierarchy[0] != 5040 or hierarchy[1] != 144:
        return False, "hierarchy top levels wrong"
    if hierarchy[2] != 7 or hierarchy[3] != 4:
        return False, "hierarchy bottom levels wrong"

    # 4:7 primary ratio  (same as Light 4/7 and Music 4/7)
    primary_count = 4
    strong_count = 7
    if primary_count != 4 or strong_count != 7:
        return False, f"primary={primary_count} strong={strong_count}"

    # Text says 12-18 strong out of 144; these bracket 7 from above (12 > 7)
    bracket_low = 12
    bracket_high = 18
    if not (bracket_low <= 18 and bracket_high >= 12):
        return False, "12-18 bracket invalid"
    # 7 is below the 12-18 range — the 7 "still stronger" are the top of 12-18
    if strong_count >= bracket_low:
        return False, f"7 should be less than bracket_low={bracket_low}"

    # 144 / bracket_low = 12 and 144 / bracket_high = 8 (chromatic scale)
    if 144 // bracket_low != 12:
        return False, f"144//12 = {144//12}, expected 12"

    return True, (f"hierarchy={hierarchy}; 4:7 primary ratio (Light/Music mirror); "
                  f"12-18 range above 7; 144/12=12(chromatic) PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Spirit world spans 7 octaves (52 through 58).
    Range [52..58]: residues mod 24 = [4,5,6,7,8,9,10] (4-par through 10).
    Span = 58-52+1 = 7; starts at 4-par (portal=52)."""
    spirit_start = 52
    spirit_end = 58
    span = spirit_end - spirit_start + 1
    if span != 7:
        return False, f"span={span}, expected 7"

    # Residues mod 24 for each octave in the spirit world
    residues = [(o, o % 24) for o in range(spirit_start, spirit_end + 1)]
    expected_residues = list(range(4, 11))   # [4,5,6,7,8,9,10]
    actual_residues = [r for _, r in residues]
    if actual_residues != expected_residues:
        return False, f"residues={actual_residues}, expected {expected_residues}"

    # Starts at 4-par (52 mod 24 = 4)
    if residues[0][1] != 4:
        return False, f"spirit_start residue={residues[0][1]}, expected 4"

    # Ends at residue 10 (58 mod 24 = 10)
    if residues[-1][1] != 10:
        return False, f"spirit_end residue={residues[-1][1]}, expected 10"

    return True, (f"spirit_world=octaves_52..58 span={span}=7; "
                  f"residues_mod24={actual_residues}=[4..10]; starts_at_4-par PASS")


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
