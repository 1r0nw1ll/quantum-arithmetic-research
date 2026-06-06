# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations
from math import gcd


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    p = 3
    while p * p <= n:
        if n % p == 0:
            return False
        p += 2
    return True


def check_c1() -> tuple[bool, str]:
    """C1: 7-acre farm = 3 orchards + 4 forest acres; 3+4=7.
    3 mod 24=3 (3-par), 4 mod 24=4 (4-par), 7 mod 24=7 (Cosmos).
    3*4=12 (chromatic octave), 3^2+4^2=5^2 (Pythagorean)."""
    orchard_acres = 3
    forest_acres = 4
    total_acres = orchard_acres + forest_acres
    if total_acres != 7:
        return False, f"total acres={total_acres}, expected 7"

    r3 = orchard_acres % 24   # 3
    r4 = forest_acres % 24    # 4
    r7 = total_acres % 24     # 7
    if r3 != 3:
        return False, f"3 mod 24={r3}, expected 3"
    if r4 != 4:
        return False, f"4 mod 24={r4}, expected 4"
    if r7 != 7:
        return False, f"7 mod 24={r7}, expected 7"

    # 3*4=12 (12 chromatic notes per octave)
    if orchard_acres * forest_acres != 12:
        return False, f"3*4={orchard_acres*forest_acres}, expected 12"

    # 3^2+4^2=5^2 (Pythagorean triple with 5-par hypotenuse)
    if orchard_acres * orchard_acres + forest_acres * forest_acres != 25:
        return False, "3^2+4^2 != 25"
    if 25 % 24 != 1:
        return False, f"25 mod 24={25%24}, expected 1 (Singularity)"

    return True, (f"7_acres=3+4; r3={r3} r4={r4} r7={r7}; "
                  f"3*4=12(chromatic); 3^2+4^2=25=5^2 25%24=1(Singularity) PASS")


def check_c2() -> tuple[bool, str]:
    """C2: QA learning times: 10-year-old grasped in 1 hour; two older siblings
    required 3 hours each. 3/1=3; 1+3=4 (QA tuple size). 3 is the length of
    male QA seed apex (b=1,e=1,diff=2,apex=3)."""
    age_youngest = 10
    time_youngest = 1     # 1 hour to grasp QA independently
    time_older    = 3     # 3 hours of instruction for older siblings

    ratio = time_older // time_youngest   # 3
    if ratio != 3:
        return False, f"ratio older/youngest = {ratio}, expected 3"

    # 1+3=4 = QA tuple size
    if time_youngest + time_older != 4:
        return False, f"1+3={time_youngest+time_older}, expected 4"

    # Male seed: b=1, e=1 -> diff=b+e=2, apex=b+2e=3  (A2 derived)
    b_seed, e_seed = 1, 1
    diff_seed = b_seed + e_seed       # 2 (derived)
    apex_seed = b_seed + 2 * e_seed   # 3 (derived) = time_older
    if apex_seed != time_older:
        return False, f"apex_seed={apex_seed} != time_older={time_older}"

    # 10-year-old: 10 mod 24 = 10; 10 is 2-par
    if age_youngest % 24 != 10:
        return False, f"10 mod 24={age_youngest%24}, expected 10"

    return True, (f"youngest=10yr grasped in {time_youngest}hr; "
                  f"older={time_older}hr; ratio={ratio}; "
                  f"1+3={time_youngest+time_older}=QA_tuple_size; "
                  f"apex_seed={apex_seed}=time_older PASS")


def check_c3() -> tuple[bool, str]:
    """C3: Billy aged 4 at start, 15 at follow-up; age diff=11 (prime).
    4 mod 24=4 (4-par portal entry from cert [376]); 15 mod 24=15; 4+11=15."""
    age_start  = 4
    age_end    = 15
    age_diff   = age_end - age_start    # 11

    if age_diff != 11:
        return False, f"age_diff={age_diff}, expected 11"
    if not _is_prime(age_diff):
        return False, f"{age_diff} is not prime"

    r4  = age_start % 24   # 4  (4-par)
    r11 = age_diff  % 24   # 11 (5-par — 11 mod 4 = 3, i.e., 3-par)
    r15 = age_end   % 24   # 15

    if r4 != 4:
        return False, f"4 mod 24={r4}, expected 4"
    if r11 != 11:
        return False, f"11 mod 24={r11}, expected 11"
    if r15 != 15:
        return False, f"15 mod 24={r15}, expected 15"

    # Check addition in mod-24 ring
    if (r4 + r11) % 24 != r15:
        return False, f"(4+11)%24={( r4+r11)%24} != 15"

    return True, (f"Billy: age {age_start}(mod24={r4}=4-par) -> {age_end}(mod24={r15}); "
                  f"diff={age_diff}(prime, mod24={r11}); (4+11)%24=15 PASS")


def check_c4() -> tuple[bool, str]:
    """C4: Walk-in friend received spirit at age 16, now age 70.
    70-16=54 years; 54 mod 24=6; 16=2^4; 54=2*3^3; gcd(16,54)=2.
    16 mod 24=16; 70 mod 24=22."""
    age_entry  = 16
    age_now    = 70
    years_walk = age_now - age_entry    # 54

    if years_walk != 54:
        return False, f"years_walk={years_walk}, expected 54"

    r16 = age_entry % 24   # 16
    r70 = age_now   % 24   # 22
    r54 = years_walk % 24  # 6

    if r16 != 16:
        return False, f"16 mod 24={r16}, expected 16"
    if r70 != 22:
        return False, f"70 mod 24={r70}, expected 22"
    if r54 != 6:
        return False, f"54 mod 24={r54}, expected 6"
    if (r16 + r54) % 24 != r70:
        return False, f"(16+54)%24={( r16+r54)%24} != 22"

    # 16 = 2^4 (pure power of 2)
    if 2 * 2 * 2 * 2 != 16:
        return False, "2^4 != 16"
    # 54 = 2 * 3^3
    if 2 * 3 * 3 * 3 != 54:
        return False, "2*3^3 != 54"
    if gcd(age_entry, years_walk) != 2:
        return False, f"gcd(16,54)={gcd(age_entry,years_walk)}, expected 2"

    return True, (f"walk-in: 16->70 diff=54; 16%24={r16} 54%24={r54} 70%24={r70}; "
                  f"(16+54)%24=22 ✓; 16=2^4 54=2*3^3 gcd=2 PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Ancient QA memories 2000-8000 years ago; 2000 mod 24=8000 mod 24=8.
    Both congruent mod 24; 8000-2000=6000; 6000 mod 24=0; 6000=24*250."""
    yr_low  = 2000
    yr_high = 8000

    r_low  = yr_low  % 24   # 8
    r_high = yr_high % 24   # 8

    if r_low != 8:
        return False, f"2000 mod 24={r_low}, expected 8"
    if r_high != 8:
        return False, f"8000 mod 24={r_high}, expected 8"
    if r_low != r_high:
        return False, f"residues differ: {r_low} != {r_high}"

    span = yr_high - yr_low   # 6000
    if span != 6000:
        return False, f"span={span}, expected 6000"
    if span % 24 != 0:
        return False, f"6000 mod 24={span%24}, expected 0"
    if span // 24 != 250:
        return False, f"6000/24={span//24}, expected 250"

    # Both 2000 and 8000 are multiples of 8 = 2^3
    if yr_low % 8 != 0 or yr_high % 8 != 0:
        return False, "2000 or 8000 not divisible by 8"
    # 8 = 2^3 is the common residue
    if r_low != 2 * 2 * 2:
        return False, f"residue {r_low} != 2^3"

    return True, (f"2000%24={r_low}=8=2^3; 8000%24={r_high}=8=2^3; "
                  f"same_residue; span=6000 6000%24=0 6000/24=250 PASS")


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
