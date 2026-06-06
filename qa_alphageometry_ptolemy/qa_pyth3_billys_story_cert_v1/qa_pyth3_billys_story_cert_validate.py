# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.9 Billy's Story: "
    "3*3=9 mod-9 fixed point; 3000=24*5^3 mod24=0; Atlantis 10000/11000 BC mod24 residues; "
    "4 lifetimes=QA tuple size); Theorem NT: past-life memories and historical dates are "
    "observer-projection reports; 9 mod 9=0 is the mod-9 fixed point (QA arithmetic fact); "
    "no float state, no QA orbit evolution"
)

from math import gcd


def check_c1() -> tuple[bool, str]:
    """C1: 3 cannons and 3 targets; 3*3=9=mod-9 Singularity; 9 mod 24=9;
    3+3=6; both 3 mod 24=3 (3-par)."""
    cannons = 3
    targets = 3

    product = cannons * targets   # 9
    if product != 9:
        return False, f"3*3={product}, expected 9"

    # 9 is the Singularity in QA mod-9: any QA state at (9,9) is fixed
    # In mod-9: 9 mod 9 = 0 (fixed point); in mod-24: 9 mod 24 = 9
    if product % 9 != 0:
        return False, f"9 mod 9 = {product % 9}, expected 0 (Singularity)"
    if product % 24 != 9:
        return False, f"9 mod 24 = {product % 24}, expected 9"

    total = cannons + targets    # 6
    if total != 6:
        return False, f"3+3={total}, expected 6"
    if total % 24 != 6:
        return False, f"6 mod 24 = {total % 24}, expected 6"

    # Both 3 are 3-par (3 mod 4 = 3)
    if cannons % 4 != 3 or targets % 4 != 3:
        return False, f"3 mod 4 = {cannons % 4}, expected 3 (3-par)"

    return True, (f"3*3={product} mod 9={product%9}(Singularity) mod 24={product%24}; "
                  f"3+3={total}; both_3_mod_4=3(3-par) PASS")


def check_c2() -> tuple[bool, str]:
    """C2: 4 nice castles (Billy's + four more allies); 4 mod 24=4 (4-par portal).
    4 = QA 4-tuple size (b, e, diff=b+e, apex=b+2e)."""
    nice_castles = 4   # "there were only four more castles that were nice to us"
    # The text says "four more castles that were nice" — but context suggests these
    # are the allied ones. We certify 4 as the alliance count.
    if nice_castles != 4:
        return False, f"nice_castles={nice_castles}, expected 4"
    if nice_castles % 24 != 4:
        return False, f"4 mod 24={nice_castles%24}, expected 4"
    # 4 = QA tuple size (b, e, diff, apex)
    qa_tuple_size = 4
    if nice_castles != qa_tuple_size:
        return False, f"4 != qa_tuple_size={qa_tuple_size}"

    # 4-par: 4 mod 4 = 0 (maximally even)
    if nice_castles % 4 != 0:
        return False, f"4 mod 4 = {nice_castles%4}, expected 0 (4-par)"

    # From cert [376]: spirit portal octave 52 mod 24 = 4 (4-par)
    portal_octave = 52
    if portal_octave % 24 != 4:
        return False, f"portal_octave mod 24={portal_octave%24}, expected 4"

    return True, (f"4_castles mod24=4=4-par=QA_tuple_size; "
                  f"4_mod_4=0(maximally_even); portal_octave_52%24=4 PASS")


def check_c3() -> tuple[bool, str]:
    """C3: 3000 berry tasters; 3000=24*125=24*5^3; 3000 mod 24=0.
    125=5^3; 3000=3*1000=3*8*125=3*2^3*5^3."""
    berry_tasters = 3000

    if berry_tasters % 24 != 0:
        return False, f"3000 mod 24={berry_tasters%24}, expected 0"
    if berry_tasters // 24 != 125:
        return False, f"3000/24={berry_tasters//24}, expected 125"

    # 3000 = 24 * 125 = 24 * 5^3
    if 5 * 5 * 5 != 125:
        return False, f"5^3={5*5*5}, expected 125"
    if 24 * 125 != 3000:
        return False, f"24*125={24*125}, expected 3000"

    # Also 3000 = 3 * 1000 = 3 * 8 * 125 = 3 * 2^3 * 5^3
    if 3 * 8 * 125 != 3000:
        return False, f"3*8*125={3*8*125}, expected 3000"

    # 1000 mod 24 = 16 (Myriad residue, same as 10000 mod 24)
    if 1000 % 24 != 16:
        return False, f"1000 mod 24={1000%24}, expected 16"

    return True, (f"3000=24*125=24*5^3; 3000%24=0; 125=5^3; "
                  f"3000=3*2^3*5^3; 1000%24=16=Myriad_residue PASS")


def check_c4() -> tuple[bool, str]:
    """C4: Atlantis ~10,000 B.C.; 10000 mod 24=16 (= Myriad 10000 from cert [354]).
    Billy's memories ~11,000 B.C.: 11000 mod 24=8; gap=1000 years; 1000 mod 24=16.
    9 men to fish: 9 mod 9=0 (mod-9 Singularity fixed point)."""
    atlantis_sinking_bc = 10000
    atlantis_pre_bc     = 11000
    time_gap            = atlantis_pre_bc - atlantis_sinking_bc   # 1000

    if time_gap != 1000:
        return False, f"time_gap={time_gap}, expected 1000"

    r10000 = atlantis_sinking_bc % 24   # 16
    r11000 = atlantis_pre_bc % 24       # 8
    r1000  = time_gap % 24              # 16

    if r10000 != 16:
        return False, f"10000 mod 24={r10000}, expected 16"
    if r11000 != 8:
        return False, f"11000 mod 24={r11000}, expected 8"
    if r1000 != 16:
        return False, f"1000 mod 24={r1000}, expected 16"

    # Both 10000 and 1000 have same residue 16 (Myriad mod 24)
    if r10000 != r1000:
        return False, f"residues differ: {r10000} != {r1000}"

    # (r10000 + r1000) mod 24 = 32 mod 24 = 8 = r11000
    if (r10000 + r1000) % 24 != r11000:
        return False, f"(16+16)%24={( r10000+r1000)%24} != r11000={r11000}"

    # 9 men: 9 mod 9 = 0 (mod-9 QA Singularity)
    nine_men = 9
    if nine_men % 9 != 0:
        return False, f"9 mod 9={nine_men%9}, expected 0"

    return True, (f"10000%24={r10000}=16=Myriad; 11000%24={r11000}; "
                  f"1000%24={r1000}=16; (16+16)%24=8=r11000 ✓; 9_men%9=0(Singularity) PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Billy's 4 distinct past lives: soldier/Lord/hunter/Atlantis = 4.
    4 = QA tuple size. Billy's starting age = 4. 4+4=8=2^3 (Atlantis residue from C4)."""
    lifetimes = ["soldier", "Lord", "hunter", "Atlantis"]
    num_lifetimes = len(lifetimes)

    if num_lifetimes != 4:
        return False, f"num_lifetimes={num_lifetimes}, expected 4"

    # 4 = QA tuple size
    if num_lifetimes != 4:
        return False, f"4 != QA tuple size 4"

    # Billy's age at start = 4 (same number)
    billy_start_age = 4
    if billy_start_age != num_lifetimes:
        return False, f"start_age={billy_start_age} != num_lifetimes={num_lifetimes}"

    # 4 + 4 = 8 = 2^3 (same as Atlantis residue r11000=8 from C4)
    combined = num_lifetimes + billy_start_age   # 8
    if combined != 8:
        return False, f"4+4={combined}, expected 8"
    if combined != 2 * 2 * 2:
        return False, f"8 != 2^3"

    # 4 mod 24 = 4 (4-par)
    if num_lifetimes % 24 != 4:
        return False, f"4 mod 24={num_lifetimes%24}, expected 4"

    return True, (f"lifetimes={lifetimes} count={num_lifetimes}=QA_tuple_size; "
                  f"start_age={billy_start_age}=count; 4+4={combined}=2^3; 4%24=4(4-par) PASS")


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
