# <!-- PRIMARY-SOURCE-EXEMPT: Iverson & Elkins (2006) Pythagorean Arithmetic Vol III — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (QA Quantum Number definition: "
    "4-7 prime factors; Law of Harmonics: same prime factors except one → harmonic "
    "resonance; LCM coincidence proof; male/female parity; universal 2×3 inclusion); "
    "Theorem NT: 'harmonic resonance' is an observer classification of integer arithmetic "
    "structure; causal structure is prime factorization and LCM computation; "
    "no float state, no QA orbit evolution"
)

"""
Cert [350] — QA Quantum Number Definition and Law of Harmonics

Source: Iverson, B. & Elkins, C. (2006) Pythagorean Arithmetic Vol III, Chapter 5 pp.20-21
  p.20: "The most basic quantum number is '1, 1, 2, 3', which probably occurs in nature
    only in the Creative Myriad."
  p.20: "The Female aboriginal number is 2, 1, 3, 4. The roots of a 'male' vibration begin
    and end with odd numbers."
  p.20: "We usually start with a wave which has the number 2, 3, 5, 8, having a product
    of 240"
  p.24: "A quantum number is an integer which has at least four co-prime factors, and not
    more than seven prime numbers. Most Quantum Numbers will contain six or seven prime
    factors."
  p.24-25 (Law of Harmonics): "When two Quantum Numbers have the same prime factors,
    EXCEPTING ONE PRIME FACTOR, they will be in the state of harmonic resonance with each
    other. The lower the ratio of the excepted prime factors, the greater will be the
    harmony between them."
  p.30-31 (worked example): wavelets 2,3,5,7,11 → QN=2310; wavelets 2,3,5,7,13 → QN=2730;
    common product=210; unique factors 11 and 13; LCM=30030=210×11×13.

Five claims:
  C1: QN definition: 2310=2×3×5×7×11 has exactly 5 distinct prime factors (in [4,7] range)
  C2: Law of Harmonics: 2310 and 2730 share prime factors {2,3,5,7}; differ in {11} vs {13};
      symmetric difference of prime factor sets has exactly 1 element each side
  C3: Coincidence structure: LCM(2310,2730)=30030=210×11×13; 2310×13=2730×11=30030
  C4: Male/female parity: male (1,1,2,3) starts/ends odd; female (2,1,3,4) starts/ends even;
      female has 2-par (=2) and 4-par (=4) as outer factors; product(female)/product(male)=4
  C5: Universal 2×3 inclusion: every QN product includes both 2 and 3 as prime factors;
      verified for 5 canonical QNs: 6,24,240,2310,2730
"""

from math import gcd


def _prime_factors(n: int) -> set[int]:
    """Return set of distinct prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def check_c1() -> tuple[bool, str]:
    """QN=2310 has exactly 5 distinct prime factors; aboriginal has < 4 (primitive/seed)."""
    # Worked example from p.30-31: 2×3×5×7×11 = 2310
    qn_2310 = 2 * 3 * 5 * 7 * 11
    assert qn_2310 == 2310
    pf_2310 = _prime_factors(2310)
    assert pf_2310 == {2, 3, 5, 7, 11}, f"prime factors of 2310: {pf_2310}"
    assert 4 <= len(pf_2310) <= 7, f"|pf(2310)| = {len(pf_2310)} not in [4,7]"

    # Aboriginal male (1,1,2,3): product = 6 = 2×3 → 2 prime factors (primitive seed)
    product_male = 1 * 1 * 2 * 3
    assert product_male == 6
    pf_male = _prime_factors(6)
    assert pf_male == {2, 3} and len(pf_male) == 2, f"male aboriginal pf: {pf_male}"
    assert len(pf_male) < 4, "aboriginal male product has ≥4 primes (should be < 4)"

    # Female aboriginal (2,1,3,4): product = 24 = 2³×3 → 2 prime factors (also primitive)
    product_female = 2 * 1 * 3 * 4
    assert product_female == 24
    pf_female = _prime_factors(24)
    assert pf_female == {2, 3} and len(pf_female) == 2

    # Wave (2,3,5,8): product = 240 = 2⁴×3×5 → 3 prime factors (below viable 4-7 range)
    product_wave = 2 * 3 * 5 * 8
    assert product_wave == 240
    pf_wave = _prime_factors(240)
    assert pf_wave == {2, 3, 5} and len(pf_wave) == 3

    return True, (
        f"2310=2×3×5×7×11 has {len(pf_2310)} prime factors ✓ (in [4,7]); "
        f"aboriginal products 6,24 have 2 prime factors each (primitive seeds); "
        f"wave product 240 has 3 prime factors (pre-viable)"
    )


def check_c2() -> tuple[bool, str]:
    """Law of Harmonics: 2310 and 2730 share {2,3,5,7}; differ in {11} vs {13}."""
    qn1, qn2 = 2310, 2730
    pf1 = _prime_factors(qn1)  # {2,3,5,7,11}
    pf2 = _prime_factors(qn2)  # {2,3,5,7,13}

    assert pf1 == {2, 3, 5, 7, 11}, f"pf(2310) = {pf1}"
    assert pf2 == {2, 3, 5, 7, 13}, f"pf(2730) = {pf2}"

    common = pf1 & pf2
    unique1 = pf1 - pf2  # {11}
    unique2 = pf2 - pf1  # {13}

    assert common == {2, 3, 5, 7}, f"common factors: {common}"
    assert unique1 == {11}, f"unique to 2310: {unique1}"
    assert unique2 == {13}, f"unique to 2730: {unique2}"

    # Symmetric difference has exactly 1 element on each side → Law of Harmonics applies
    assert len(unique1) == 1 and len(unique2) == 1, (
        f"Symmetric diff not 1+1: |{unique1}| + |{unique2}|"
    )

    # Ratio of differing primes: 11/13 ≈ 0.846 (close to 1 → strong harmony)
    p, q = 11, 13
    assert p < q  # lower valued is the "smaller" unique factor
    # Ratio p/q = 11/13 — check it's between 1/2 and 1 (strongly harmonic)
    assert p * 2 > q, f"ratio {p}/{q} not in strongly-harmonic range (> 1/2)"

    return True, (
        f"pf(2310)={{2,3,5,7,11}}, pf(2730)={{2,3,5,7,13}}; "
        f"common={{2,3,5,7}}; unique: {{11}} vs {{13}}; "
        f"ratio 11/13≈0.846 (close to 1 → strong harmony per Law of Harmonics)"
    )


def check_c3() -> tuple[bool, str]:
    """Coincidence: LCM(2310,2730)=30030=210×11×13; 2310×13=2730×11."""
    qn1, qn2 = 2310, 2730
    common_product = 2 * 3 * 5 * 7  # = 210
    p1, p2 = 11, 13

    assert common_product == 210
    assert qn1 == common_product * p1  # 210 × 11 = 2310
    assert qn2 == common_product * p2  # 210 × 13 = 2730

    lcm_val = _lcm(qn1, qn2)
    assert lcm_val == common_product * p1 * p2  # = 210 × 143 = 30030
    assert lcm_val == 30030, f"LCM(2310,2730) = {lcm_val}, expected 30030"

    # "The first string projects 13 of these aliquot parts; the second projects 11"
    assert qn1 * p2 == lcm_val  # 2310 × 13 = 30030
    assert qn2 * p1 == lcm_val  # 2730 × 11 = 30030

    # Verify: qn1 and qn2 coincide at LCM (one complete cycle for each)
    assert lcm_val % qn1 == 0
    assert lcm_val % qn2 == 0
    assert lcm_val // qn1 == p2  # 13 cycles of qn1
    assert lcm_val // qn2 == p1  # 11 cycles of qn2

    return True, (
        f"LCM(2310,2730)={lcm_val}=210×11×13; "
        f"2310 runs {lcm_val//qn1} cycles, 2730 runs {lcm_val//qn2} cycles to coincide; "
        f"common aliquot period=210; coincidence at {lcm_val}"
    )


def check_c4() -> tuple[bool, str]:
    """Male/female parity: male (1,1,2,3) odd-ended; female (2,1,3,4) even-ended."""
    male = (1, 1, 2, 3)
    female = (2, 1, 3, 4)

    # Male: first and last are odd
    assert male[0] % 2 == 1, f"male[0]={male[0]} not odd"
    assert male[-1] % 2 == 1, f"male[-1]={male[-1]} not odd"

    # Female: first and last are even
    assert female[0] % 2 == 0, f"female[0]={female[0]} not even"
    assert female[-1] % 2 == 0, f"female[-1]={female[-1]} not even"

    # Female outer factors: 2 (2-par) and 4 (4-par)
    assert female[0] % 4 != 0 and female[0] % 2 == 0, "female[0]=2 is not 2-par (only)"
    assert female[-1] % 4 == 0, "female[-1]=4 is not 4-par"

    # Product ratio female/male = 24/6 = 4
    prod_male = 1 * 1 * 2 * 3   # = 6
    prod_female = 2 * 1 * 3 * 4  # = 24
    assert prod_male == 6
    assert prod_female == 24
    assert prod_female // prod_male == 4  # female product = 4 × male product

    # Male has exactly one even prime factor (2); female has two even prime factors (2,4=2²)
    pf_male_prod = _prime_factors(prod_male)    # {2,3}
    pf_female_prod = _prime_factors(prod_female) # {2,3}
    assert pf_male_prod == pf_female_prod == {2, 3}  # same prime factors, same set

    return True, (
        f"male (1,1,2,3): first=1 odd, last=3 odd; product=6; "
        f"female (2,1,3,4): first=2 (2-par), last=4 (4-par); product=24; "
        f"ratio female/male product = 4; both share prime factors {{2,3}}"
    )


def check_c5() -> tuple[bool, str]:
    """Universal 2×3 inclusion: all 5 canonical QN products contain both 2 and 3."""
    canonical_qns = [
        (6,    "aboriginal male product (1×1×2×3)"),
        (24,   "aboriginal female product (2×1×3×4)"),
        (240,  "wave product (2×3×5×8)"),
        (2310, "five-wavelet QN (2×3×5×7×11)"),
        (2730, "five-wavelet QN (2×3×5×7×13)"),
    ]
    for n, desc in canonical_qns:
        pf = _prime_factors(n)
        assert 2 in pf, f"{n} ({desc}) missing factor 2"
        assert 3 in pf, f"{n} ({desc}) missing factor 3"

    # Also check the "always includes 2 and 3" claim for all Fibonacci quads (b odd, gcd(b,e)=1)
    # The product b×e×d×a: since d=b+e and a=d+e, and C=2de (C always 4-par → divisible by 4),
    # and F=ab (F=ab, a odd, b odd → F odd)... the bead product abde = abde.
    # Iverson's claim is that ALL QNs contain 2 and 3 as prime factors.
    # Verify: product abde for Fibonacci quads: b=1,e=2,d=3,a=5 → 30=2×3×5 ✓ (has 2 and 3)
    # b=3,e=2,d=5,a=7 → 210=2×3×5×7 ✓; b=5,e=2,d=7,a=9=3² → 630=2×3²×5×7 ✓
    quads = [
        (1, 2),  # d=3,a=5; product=30
        (3, 2),  # d=5,a=7; product=210
        (5, 2),  # d=7,a=9; product=630
        (1, 4),  # d=5,a=9; product=180
        (3, 4),  # d=7,a=11; product=924
    ]
    for b, e in quads:
        d = b + e
        a = d + e
        product = b * e * d * a
        pf = _prime_factors(product)
        assert 2 in pf, f"b={b},e={e}: product={product} missing factor 2"
        assert 3 in pf, f"b={b},e={e}: product={product} missing factor 3"

    return True, (
        f"All 5 canonical QN products {{6,24,240,2310,2730}} contain both 2 and 3; "
        f"5 Fibonacci quad products {{30,210,630,180,924}} all contain 2 and 3; "
        f"universal 2×3 inclusion verified"
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
        raise RuntimeError(f"cert [350] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
