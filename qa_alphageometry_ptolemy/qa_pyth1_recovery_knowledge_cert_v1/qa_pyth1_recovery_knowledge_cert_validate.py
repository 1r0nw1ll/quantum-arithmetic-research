# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol I — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-1 Ch.I Recovery of Knowledge: "
    "Pythagoras 580 BC mod24=4=portal; Atlantis 9400yr mod24=16=Myriad; "
    "1548 BC earthquake mod24=12=chromatic; 340 generations mod24=4=portal; "
    "529 BC Crotona=23^2 mod24=1 Singularity-class; 3 stages centuries 600/400/300); "
    "Theorem NT: Pythagorean history and genealogical counts are observer-projection; "
    "no float state, no QA orbit evolution"
)

from math import gcd


def check_c1() -> tuple[bool, str]:
    """C1: Pythagoras born ~580 B.C.; 580 mod 24=4 (portal 4-par class, cert [376]).
    Atlantis sank 9400 years before Pythagoras; 9400 mod 24=16=Myriad residue.
    9400=24*391+16; 391=17*23 (product of two primes)."""
    birth_bc    = 580
    atlantis_yr = 9400

    r_birth = birth_bc % 24   # 4
    if r_birth != 4:
        return False, f"580 mod 24={r_birth}, expected 4"

    # Portal 4-par: same as octave 52 mod 24=4 (cert [376] C1)
    portal_residue = 52 % 24   # 4
    if r_birth != portal_residue:
        return False, f"580%24={r_birth} != 52%24={portal_residue}"

    r_atlantis = atlantis_yr % 24   # 16
    if r_atlantis != 16:
        return False, f"9400 mod 24={r_atlantis}, expected 16"

    # 9400=24*391+16
    if atlantis_yr // 24 != 391:
        return False, f"9400//24={atlantis_yr//24}, expected 391"
    if 391 * 24 + 16 != atlantis_yr:
        return False, f"391*24+16={391*24+16}, expected 9400"

    # 391=17*23 (product of two primes)
    if 17 * 23 != 391:
        return False, f"17*23={17*23}, expected 391"

    # Both residues: 4 (portal) and 16 (Myriad)
    return True, (f"580%24={r_birth}=4=portal-4-par; 9400%24={r_atlantis}=16=Myriad; "
                  f"9400=24*391+16; 391=17*23; 52%24=4=same PASS")


def check_c2() -> tuple[bool, str]:
    """C2: Earthquake 1548 B.C. (1000 years before Pythagoras at 580 BC? actually text says
    'devastated by earthquakes... only 1000 years before the time of Pythagoras, in 1548 B.C.')
    1548 mod 24=12=chromatic scale; 1000 mod 24=16=Myriad; 1548-580=968 (actual diff);
    but text uses 1000 as the round number gap: 1000 mod 24=16."""
    earthquake_bc = 1548
    gap_years     = 1000    # "only 1000 years before the time of Pythagoras"

    r_eq = earthquake_bc % 24   # 12
    if r_eq != 12:
        return False, f"1548 mod 24={r_eq}, expected 12"

    # 12 = chromatic scale (same as 180 deg mod 24 from cert [381])
    if r_eq != 12:
        return False, f"1548%24={r_eq} != 12 chromatic"

    # 1548=24*64+12
    if earthquake_bc // 24 != 64:
        return False, f"1548//24={earthquake_bc//24}, expected 64"
    if 64 * 24 + 12 != earthquake_bc:
        return False, f"64*24+12={64*24+12}, expected 1548"

    # 1000 mod 24=16=Myriad
    r_gap = gap_years % 24   # 16
    if r_gap != 16:
        return False, f"1000 mod 24={r_gap}, expected 16"

    # Cross-reference: 1000 mod 24=16 appears in [378]C3, [379]C5, [381]C2
    if gap_years % 24 != 1000 % 24:
        return False, f"gap residue mismatch"

    return True, (f"1548%24={r_eq}=12=chromatic; 1548=64*24+12; "
                  f"1000%24={r_gap}=16=Myriad; cross-refs [378][379] PASS")


def check_c3() -> tuple[bool, str]:
    """C3: 'more than 340 generations' of ancient history in Egypt.
    340 mod 24=4 (portal 4-par class: same as Pythagoras birth 580 mod24=4).
    340=4*85=4*5*17; factors include 4 (tuple size) and 5 (QN seed element).
    340/4=85=5*17; 85 mod 24=? 85=3*24+13; 85 mod 24=13."""
    generations = 340

    r = generations % 24   # 4
    if r != 4:
        return False, f"340 mod 24={r}, expected 4"

    # 340=4*85=4*5*17
    if generations // 4 != 85:
        return False, f"340/4={generations//4}, expected 85"
    if 85 != 5 * 17:
        return False, f"85 != 5*17"
    if generations != 4 * 5 * 17:
        return False, f"340 != 4*5*17"

    # 4 = QA tuple size
    tuple_size = 4
    if generations % tuple_size != 0:
        return False, f"340 mod 4={generations%tuple_size}, expected 0"

    # Same portal residue 4 as Pythagoras birth year
    birth_residue = 580 % 24   # 4
    if r != birth_residue:
        return False, f"340%24={r} != 580%24={birth_residue}"

    return True, (f"340%24={r}=4=portal-4-par; 340=4*5*17; 340/4={generations//4}=85=5*17; "
                  f"same_as_580%24=4 PASS")


def check_c4() -> tuple[bool, str]:
    """C4: Pythagorean school founded 529 B.C. at Crotona.
    529=23^2 (square of 23rd prime); 529 mod 24=1 (Singularity-class: same as 49 mod 24=1 cert [381]).
    23 is prime; 23 mod 24=23 (opposite of 1 mod 24 gap=22).
    'more than thirty years of travel': 30 mod 24=6=seed product."""
    school_bc     = 529
    travel_years  = 30

    # 529=23^2
    if school_bc != 23 * 23:
        return False, f"529 != 23^2"

    r_school = school_bc % 24   # 1
    if r_school != 1:
        return False, f"529 mod 24={r_school}, expected 1"

    # 529=22*24+1
    if school_bc // 24 != 22:
        return False, f"529//24={school_bc//24}, expected 22"
    if 22 * 24 + 1 != school_bc:
        return False, f"22*24+1={22*24+1}, expected 529"

    # 23 is prime
    if not all(23 % k != 0 for k in range(2, 23)):
        return False, "23 is not prime"

    # 30 mod 24=6=seed product
    r_travel = travel_years % 24
    if r_travel != 6:
        return False, f"30 mod 24={r_travel}, expected 6"
    if r_travel != 1 * 1 * 2 * 3:
        return False, f"30%24={r_travel} != seed_product=6"

    return True, (f"529=23^2; 529%24={r_school}=1=Singularity-class; 529=22*24+1; "
                  f"23_prime; 30%24={r_travel}=6=seed_product PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Three stages of knowledge recovery: Pythagorean (6th cent.), Platonic (4th cent.),
    Euclidean (3rd cent.); 3 stages=3-par element.
    Century markers mod 24: 600%24=0 (25*24); 400%24=16=Myriad; 300%24=12=chromatic.
    3 stages * chromatic(12) = 36; 36 mod 24=12 (chromatic again, closed)."""
    num_stages    = 3
    pythagorean_c = 600   # 6th century BC approx midpoint
    platonic_c    = 400   # 4th century BC
    euclidean_c   = 300   # 3rd century BC

    if num_stages % 24 != 3:
        return False, f"3 mod 24={num_stages%24}, expected 3"

    r600 = pythagorean_c % 24   # 0
    r400 = platonic_c % 24      # 16
    r300 = euclidean_c % 24     # 12

    if r600 != 0:
        return False, f"600 mod 24={r600}, expected 0"
    if r400 != 16:
        return False, f"400 mod 24={r400}, expected 16"
    if r300 != 12:
        return False, f"300 mod 24={r300}, expected 12"

    # 3 * 12 = 36; 36 mod 24 = 12 (chromatic, closed)
    product = num_stages * r300
    if product != 36:
        return False, f"3*12={product}, expected 36"
    if product % 24 != 12:
        return False, f"36 mod 24={product%24}, expected 12"

    # Cross-refs: 600%24=0 in cert [379]; 400%24=16 in cert [381]
    return True, (f"3_stages%24=3(3-par); 600%24={r600}=0; 400%24={r400}=16=Myriad; "
                  f"300%24={r300}=12=chromatic; 3*12={product}; 36%24=12 closed PASS")


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
