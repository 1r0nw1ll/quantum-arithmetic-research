# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-2 Ch.XI Basics: "
    "Plato Atlantis 9600yr mod24=0=400*24; 9600-9400=200 mod24=8; Ishango 7000 BC mod24=16=Myriad; "
    "8 primes to 19=8=2^3; sea level 100m mod24=4=portal; Pythagorean school 529-505=24=QA modulus; "
    "505 mod24=1=529 mod24=1 Singularity-class; 600yr destruction mod24=0; 4 elements=QA tuple); "
    "Theorem NT: historical and archaeological data are observer-projection; "
    "no float state, no QA orbit evolution"
)

from math import gcd


def check_c1() -> tuple[bool, str]:
    """C1: Plato wrote of Atlantis '9,600 years' before his time; 9600 mod 24=0 (400*24).
    Contrast [382] Pyth-1: Atlantis was 9400 years before Pythagoras (9400 mod 24=16).
    Difference: 9600-9400=200; 200 mod 24=8=2^3 (cert [379] C4)."""
    plato_atlantis  = 9600
    pyth_atlantis   = 9400   # from cert [382]

    if plato_atlantis % 24 != 0:
        return False, f"9600 mod 24={plato_atlantis%24}, expected 0"
    if plato_atlantis // 24 != 400:
        return False, f"9600/24={plato_atlantis//24}, expected 400"

    # 9400 mod 24=16 (from cert [382])
    if pyth_atlantis % 24 != 16:
        return False, f"9400 mod 24={pyth_atlantis%24}, expected 16"

    # Difference 200 mod 24=8=2^3
    diff = plato_atlantis - pyth_atlantis
    if diff != 200:
        return False, f"9600-9400={diff}, expected 200"
    if diff % 24 != 8:
        return False, f"200 mod 24={diff%24}, expected 8"
    if diff % 24 != 2 * 2 * 2:
        return False, f"200%24={diff%24} != 2^3"

    return True, (f"9600%24=0(400*24); 9400%24=16; 9600-9400={diff}; "
                  f"200%24={diff%24}=8=2^3 PASS")


def check_c2() -> tuple[bool, str]:
    """C2: Ishango counting bone dated '7000 B.C.' shows 'all prime numbers up to 19'.
    7000 mod 24=16=Myriad residue; primes up to 19={2,3,5,7,11,13,17,19}=8 primes.
    8=2^3; 8=phi(30) (cert [367] C1)."""
    ishango_bc     = 7000
    primes_to_19   = [2, 3, 5, 7, 11, 13, 17, 19]

    r = ishango_bc % 24   # 16
    if r != 16:
        return False, f"7000 mod 24={r}, expected 16"

    # 7000=291*24+16
    if ishango_bc // 24 != 291:
        return False, f"7000//24={ishango_bc//24}, expected 291"
    if 291 * 24 + 16 != ishango_bc:
        return False, f"291*24+16={291*24+16}, expected 7000"

    num_primes = len(primes_to_19)
    if num_primes != 8:
        return False, f"primes_to_19 count={num_primes}, expected 8"

    # 8 = 2^3
    if num_primes != 2 * 2 * 2:
        return False, f"8 != 2^3"

    # 8 = phi(30) (cert [367])
    # phi(30): count integers 1..30 coprime to 30; 30=2*3*5; phi(30)=30*(1-1/2)*(1-1/3)*(1-1/5)=8
    coprime_to_30 = [n for n in range(1, 30) if gcd(n, 30) == 1]
    if len(coprime_to_30) != 8:
        return False, f"phi(30)={len(coprime_to_30)}, expected 8"

    return True, (f"7000%24={r}=16=Myriad; 7000=291*24+16; "
                  f"primes_to_19={num_primes}=8=2^3=phi(30) PASS")


def check_c3() -> tuple[bool, str]:
    """C3: Atlantis destruction caused sea level drop of 'up to 100 meters'.
    100 mod 24=4 (portal 4-par class, same as Pythagoras 580 BC mod24=4).
    100=4*25=4*5^2; gcd(100,24)=4; 100/4=25=5^2."""
    sea_level_m = 100

    r = sea_level_m % 24   # 4
    if r != 4:
        return False, f"100 mod 24={r}, expected 4"

    # 100=4*25=4*5^2
    if sea_level_m // 4 != 25:
        return False, f"100/4={sea_level_m//4}, expected 25"
    if 25 != 5 * 5:
        return False, f"25 != 5^2"
    if sea_level_m != 4 * 5 * 5:
        return False, f"100 != 4*5^2"

    # gcd(100,24)=4
    g = gcd(sea_level_m, 24)
    if g != 4:
        return False, f"gcd(100,24)={g}, expected 4"

    # Same portal residue 4 as 580 BC (cert [382])
    birth_residue = 580 % 24   # 4
    if r != birth_residue:
        return False, f"100%24={r} != 580%24={birth_residue}"

    return True, (f"100%24={r}=4=portal; 100=4*5^2; gcd(100,24)={g}=4; "
                  f"same_as_580%24=4 PASS")


def check_c4() -> tuple[bool, str]:
    """C4: Pythagorean school founded 529 B.C.; Pythagoreans expelled 505 B.C.
    529-505=24=QA modulus (one complete cycle); both 529 mod24=1 and 505 mod24=1 (Singularity-class).
    505=5*101 (5-par base); 101 is prime."""
    school_bc    = 529
    expelled_bc  = 505

    duration = school_bc - expelled_bc   # 24
    if duration != 24:
        return False, f"529-505={duration}, expected 24 (QA modulus)"

    r_school   = school_bc % 24    # 1
    r_expelled = expelled_bc % 24   # 1

    if r_school != 1:
        return False, f"529 mod 24={r_school}, expected 1"
    if r_expelled != 1:
        return False, f"505 mod 24={r_expelled}, expected 1"

    # Both Singularity-class
    if r_school != r_expelled:
        return False, f"residues differ: {r_school} != {r_expelled}"

    # 505=5*101; 101 is prime
    if expelled_bc != 5 * 101:
        return False, f"505 != 5*101"
    if not all(101 % k != 0 for k in range(2, 101)):
        return False, "101 is not prime"

    return True, (f"529-505={duration}=24=QA_modulus; 529%24={r_school}=1; "
                  f"505%24={r_expelled}=1=Singularity-class; both_equal; 505=5*101; 101_prime PASS")


def check_c5() -> tuple[bool, str]:
    """C5: '600 years' of systematic mathematical destruction after Pythagoreans.
    600 mod 24=0 (25*24; same as cert [379] C2 Christ Spirit gap).
    Four states of matter (earth/air/fire/water)=4=QA tuple size; 4 mod 24=4=portal.
    600/4=150=6*25=6*5^2; 150 mod 24=? 150=6*24+6; 150 mod 24=6=seed product."""
    destroy_years  = 600
    num_elements   = 4   # earth, air, fire, water

    if destroy_years % 24 != 0:
        return False, f"600 mod 24={destroy_years%24}, expected 0"
    if destroy_years // 24 != 25:
        return False, f"600/24={destroy_years//24}, expected 25"

    # 4 elements = QA tuple size
    if num_elements != 4:
        return False, f"num_elements={num_elements}, expected 4"
    if num_elements % 24 != 4:
        return False, f"4 mod 24={num_elements%24}, expected 4"

    # 600/4=150; 150 mod 24=6=seed product
    ratio = destroy_years // num_elements   # 150
    if ratio != 150:
        return False, f"600/4={ratio}, expected 150"
    if ratio % 24 != 6:
        return False, f"150 mod 24={ratio%24}, expected 6"
    if ratio % 24 != 1 * 1 * 2 * 3:
        return False, f"150%24={ratio%24} != seed_product=6"

    return True, (f"600%24=0(25*24); 4_elements=QA_tuple_size; 4%24=4=portal; "
                  f"600/4={ratio}; 150%24={ratio%24}=6=seed_product PASS")


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
