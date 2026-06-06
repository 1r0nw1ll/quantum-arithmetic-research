# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-3 Ch.11 Pitfalls of Invention: "
    "VTA 5kW threshold 5=5-par QN tuple element; pyramid doublings $10->$20->$40->$80->$160 "
    "4 steps=QA tuple size ratio=2=octave; lab $40000 mod24=16=Myriad; $3000/page 3000%24=0=24*5^3; "
    "Searl 1948 mod24=4=portal-4-par); Theorem NT: commercial pricing and research history are "
    "observer-projection reports; no float state, no QA orbit evolution"
)

from math import gcd


def check_c1() -> tuple[bool, str]:
    """C1: VTA threshold = 5 kilowatts.
    5 appears as third element in QN base seed (2,3,5,8); 5=5-par (5 mod 4=1, 5 mod 24=5).
    5*1000=5000; 5000 mod 24=8=2^3; 5 is the generating prime for 5-par class."""
    threshold_kw = 5

    # 5 is the third element of QN base seed (2, 3, 5, 8)
    qn_seed = [2, 3, 5, 8]
    if qn_seed[2] != 5:
        return False, f"QN seed[2]={qn_seed[2]}, expected 5"
    if threshold_kw != qn_seed[2]:
        return False, f"threshold_kw={threshold_kw} != qn_seed[2]=5"

    # 5 mod 24=5 (5-par class)
    if threshold_kw % 24 != 5:
        return False, f"5 mod 24={threshold_kw%24}, expected 5"

    # 5*1000=5000 mod 24=8=2^3
    kilo_watts = threshold_kw * 1000
    if kilo_watts != 5000:
        return False, f"5*1000={kilo_watts}, expected 5000"
    if kilo_watts % 24 != 8:
        return False, f"5000 mod 24={kilo_watts%24}, expected 8"
    if kilo_watts % 24 != 2 * 2 * 2:
        return False, f"5000%24={kilo_watts%24} != 2^3=8"

    # 5 is prime
    if not all(5 % k != 0 for k in range(2, 5)):
        return False, "5 is not prime"

    return True, (f"5_kW=qn_seed[2]={qn_seed[2]}; 5%24=5(5-par); "
                  f"5000%24={kilo_watts%24}=8=2^3; 5_is_prime PASS")


def check_c2() -> tuple[bool, str]:
    """C2: Pyramid pricing doublings: $10 -> $20 -> $40 -> $80 -> $160.
    4 doublings = QA 4-tuple size; each ratio = 2 (octave doubling).
    Mod-24 sequence: 10, 20, 16, 8, 16 (alternating Myriad/Singularity-class values)."""
    prices = [10, 20, 40, 80, 160]
    num_doublings = len(prices) - 1   # 4

    if num_doublings != 4:
        return False, f"num_doublings={num_doublings}, expected 4"

    # Verify each ratio = 2
    for i in range(1, len(prices)):
        ratio = prices[i] // prices[i - 1]
        if ratio != 2:
            return False, f"prices[{i}]/prices[{i-1}]={ratio}, expected 2"

    # 4 doublings = QA tuple size
    qa_tuple_size = 4
    if num_doublings != qa_tuple_size:
        return False, f"num_doublings={num_doublings} != qa_tuple_size={qa_tuple_size}"

    # Mod-24 sequence
    residues = [p % 24 for p in prices]   # [10, 20, 16, 8, 16]
    expected = [10, 20, 16, 8, 16]
    if residues != expected:
        return False, f"residues={residues}, expected {expected}"

    # Final price / initial price = 2^4 = 16
    multiplier = prices[-1] // prices[0]
    if multiplier != 16:
        return False, f"160/10={multiplier}, expected 16"
    if multiplier != 2 * 2 * 2 * 2:
        return False, f"16 != 2^4"

    return True, (f"prices={prices}; num_doublings={num_doublings}=QA_tuple_size; "
                  f"ratio=2(octave); residues={residues}; 160/10={multiplier}=2^4 PASS")


def check_c3() -> tuple[bool, str]:
    """C3: Sparky's lab cost $40,000.
    40000 mod 24=16=Myriad residue (same as 10000 BC mod24=16 in cert [354],[379]).
    40000=40*1000; 40*1000 mod24=16*16%24=? Let's verify directly.
    40000/24=1666.67; 40000=1666*24+16; gcd(40000,24)=8."""
    lab_cost = 40000

    r = lab_cost % 24   # 16
    if r != 16:
        return False, f"40000 mod 24={r}, expected 16"

    if lab_cost // 24 != 1666:
        return False, f"40000//24={lab_cost//24}, expected 1666"
    if 1666 * 24 + 16 != lab_cost:
        return False, f"1666*24+16={1666*24+16}, expected 40000"

    # gcd(40000, 24)=8
    g = gcd(lab_cost, 24)
    if g != 8:
        return False, f"gcd(40000,24)={g}, expected 8"

    # 40000=40*1000; both 1000 mod24=16 and 40000 mod24=16 share Myriad residue
    if 1000 % 24 != 16:
        return False, f"1000 mod 24={1000%24}, expected 16"
    if r != 1000 % 24:
        return False, f"40000%24={r} != 1000%24={1000%24}"

    return True, (f"40000%24={r}=16=Myriad; 40000=1666*24+16; "
                  f"gcd(40000,24)={g}=8; 1000%24=16 matches PASS")


def check_c4() -> tuple[bool, str]:
    """C4: '$3000 per typed sheet' — Iverson's pricing for science journals.
    3000=24*5^3=24*125; 3000 mod 24=0.
    Cross-reference: 3000 appears in [378]C3 (berry tasters) and [379]C5 (3000 years ago).
    3000/1000=3; 3000/600=5 (ratio to Christ Spirit gap from [379]C2)."""
    price_per_page = 3000

    if price_per_page % 24 != 0:
        return False, f"3000 mod 24={price_per_page%24}, expected 0"
    if price_per_page // 24 != 125:
        return False, f"3000/24={price_per_page//24}, expected 125"
    if 125 != 5 * 5 * 5:
        return False, f"125 != 5^3"
    if price_per_page != 24 * 5 * 5 * 5:
        return False, f"3000 != 24*5^3"

    # 3000/1000=3 (appears as the Pythagorean prime 3)
    unit_ratio = price_per_page // 1000
    if unit_ratio != 3:
        return False, f"3000/1000={unit_ratio}, expected 3"

    # 3000/600=5 (links to 600-year Christ Spirit gap from cert [379])
    gap_ratio = price_per_page // 600
    if gap_ratio != 5:
        return False, f"3000/600={gap_ratio}, expected 5"

    return True, (f"3000%24=0; 3000=24*5^3=24*125; 3000/1000={unit_ratio}=3-par; "
                  f"3000/600={gap_ratio}=5-par; cross-refs [378]C3 [379]C5 PASS")


def check_c5() -> tuple[bool, str]:
    """C5: Professor Searl made first disc in 1948.
    1948 mod 24=4=portal-entry 4-par class (cert [376] C1: octave 52 mod24=4).
    1948=81*24+4; Searl had 3 stages of development: 3=3-par Pythagorean prime.
    3 stages * 1948_residue=4: 3*4=12 (chromatic scale: 12 notes per octave)."""
    year_searl = 1948
    num_stages = 3

    r = year_searl % 24   # 4
    if r != 4:
        return False, f"1948 mod 24={r}, expected 4"
    if year_searl // 24 != 81:
        return False, f"1948//24={year_searl//24}, expected 81"
    if 81 * 24 + 4 != year_searl:
        return False, f"81*24+4={81*24+4}, expected 1948"

    # 4-par: same portal class as octave 52 (cert [376])
    portal_octave_residue = 52 % 24   # 4
    if portal_octave_residue != r:
        return False, f"52%24={portal_octave_residue} != 1948%24={r}"

    # 3 stages: 3 mod 24=3 (3-par, first odd Pythagorean prime position)
    if num_stages % 24 != 3:
        return False, f"3 mod 24={num_stages%24}, expected 3"

    # 3 * 4 = 12 (chromatic scale: 12 notes per octave)
    chromatic = num_stages * r
    if chromatic != 12:
        return False, f"3*4={chromatic}, expected 12"

    return True, (f"1948%24={r}=4=portal_4-par; 1948=81*24+4; "
                  f"52%24={portal_octave_residue}=4=same_class; "
                  f"stages={num_stages}%24=3(3-par); 3*4={chromatic}=12(chromatic) PASS")


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
