#!/usr/bin/env python3
"""
qa_great_pyramid_analysis.py — Pre-registered QA analysis of the Great Pyramid.

PRE-REGISTRATION (defined BEFORE looking at results):

STEP 1: From the EXTERIOR slope ratio (base/height = 440/280 cubits),
        derive the QA direction (d,e) and full QN (b,e,d,a).

STEP 2: From that single QN, compute ALL derived quantities:
        C, F, G triple; spread of face angle; spread of passage angle;
        quantum ellipse axis ratio; conic type; par classification.

STEP 3: Test whether these PREDICTED quantities match INDEPENDENT
        internal measurements (passage angles, chamber proportions)
        that were NOT used to fit the QN.

SUCCESS CRITERION: If ≥3 independent measurements match predictions
from the single QN (within measurement uncertainty), the pyramid
encodes QA structure. If <3 match, it doesn't.

MEASUREMENTS (Petrie 1883, confirmed by Cole 1925, JMR Alison 2019):
  - Base: 440 royal cubits (230.36 m mean of 4 sides)
  - Height: 280 royal cubits (146.7 m original)
  - Seked: 5½ palms per cubit rise = 22/28 = 11/14
  - Ascending passage angle: 26°33'54" ≈ 26.565°
  - Descending passage angle: 26°31'23" ≈ 26.524°
  - Grand Gallery slope: 26° (same as passages)
  - King's Chamber: 20 × 10 × ~11.18 cubits (2:1:√5 proportion)
  - Royal cubit: 20.62 ± 0.005 inches (Petrie)

Author: Will Dale (question), Claude (analysis)
"""

QA_COMPLIANCE = "observer=archaeogeometric_analysis, state_alphabet=pyramid_measurements"

import math
from math import gcd
from fractions import Fraction


def find_qn_exact(half_base, height):
    """Find QA direction (d,e) from the pyramid's cross-section triangle.

    The pyramid cross-section is an isosceles triangle:
    half-base = 220 cubits, height = 280 cubits.
    The SLOPE direction vector is (height, half_base) = (280, 220).
    Reduce to coprime: gcd(280,220) = 20, so direction = (14, 11).

    In QA: d = 14, e = 11 → b = d-e = 3, a = b+2e = 25.
    QN = (3, 11, 14, 25).
    """
    g = gcd(half_base, height)
    d = height // g
    e = half_base // g
    b = d - e
    a = b + 2 * e
    return b, e, d, a


def main():
    print("=" * 80)
    print("GREAT PYRAMID QA ANALYSIS — PRE-REGISTERED DESIGN")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Derive QN from exterior slope
    # ═══════════════════════════════════════════════════════════════════
    print("\n── STEP 1: QN FROM EXTERIOR SLOPE ──\n")

    base_cubits = 440  # full base
    height_cubits = 280
    half_base = base_cubits // 2  # = 220

    print(f"  Great Pyramid: base = {base_cubits} cubits, height = {height_cubits} cubits")
    print(f"  Half-base = {half_base} cubits")
    print(f"  Slope ratio (rise:run) = {height_cubits}:{half_base}")

    g = gcd(height_cubits, half_base)
    print(f"  gcd({height_cubits}, {half_base}) = {g}")
    print(f"  Reduced direction: ({height_cubits//g}, {half_base//g}) = (14, 11)")

    b, e, d, a = find_qn_exact(half_base, height_cubits)
    print(f"\n  QA direction (d, e) = ({d}, {e})")
    print(f"  QN (b, e, d, a) = ({b}, {e}, {d}, {a})")
    print(f"  Check: b+e = {b}+{e} = {b+e} = d={d} ✓")
    print(f"  Check: b+2e = {b}+{2*e} = {b+2*e} = a={a} ✓")
    print(f"  Check: gcd(b,e) = gcd({b},{e}) = {gcd(b,e)} {'✓ primitive' if gcd(b,e)==1 else '✗ NOT primitive'}")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Compute ALL derived quantities from this single QN
    # ═══════════════════════════════════════════════════════════════════
    print("\n── STEP 2: DERIVED QUANTITIES FROM QN ({},{},{},{}) ──\n".format(b, e, d, a))

    # Triple
    C = 2 * d * e       # Green quadrance
    F = a * b            # Red quadrance = ab  (S1: no **)
    G = d * d + e * e    # Blue quadrance (S1: d*d not d**2)

    print(f"  TRIPLE (C, F, G):")
    print(f"    C = 2de = 2×{d}×{e} = {C}")
    print(f"    F = ab = {a}×{b} = {F}")
    print(f"    G = d²+e² = {d*d}+{e*e} = {G}")
    print(f"    Check C²+F²=G²: {C*C}+{F*F} = {C*C+F*F} = {G*G} {'✓' if C*C+F*F==G*G else '✗'}")

    # Spread of the face angle (slope angle)
    # The face angle θ has tan(θ) = height/half_base = 280/220 = 14/11
    # spread = sin²(θ) = (height²)/(height²+half_base²) = d²/(d²+e²) = d²/G
    # WAIT — spread of the slope line with horizontal:
    # Direction vector of slope = (half_base, height) = (e, d) in (horizontal, vertical)
    # Spread with horizontal = (vertical component)² / (total²) = d²/G
    s_face = Fraction(d * d, G)
    print(f"\n  FACE ANGLE SPREAD:")
    print(f"    s_face = d²/G = {d*d}/{G} = {s_face} = {float(s_face):.6f}")
    print(f"    θ_face = arcsin(√s) = {math.degrees(math.asin(math.sqrt(float(s_face)))):.4f}°")
    print(f"    Classical: arctan(280/220) = {math.degrees(math.atan2(280,220)):.4f}°")

    # Seked: horizontal displacement per cubit of rise = e/d (in palms/cubit)
    # 1 cubit = 7 palms, so seked = 7 × (half_base/height) = 7 × 220/280 = 7 × 11/14 = 5.5 palms
    seked = Fraction(7 * e, d)
    print(f"\n  SEKED (Egyptian slope measure):")
    print(f"    seked = 7 × e/d = 7 × {e}/{d} = {seked} = {float(seked):.4f} palms/cubit")
    print(f"    Historical seked of Great Pyramid: 5½ palms = 5.5 ✓")

    # Passage angle prediction
    # The descending/ascending passages have angle ≈ 26.565° = arctan(1/2)
    # In QA terms: this is the direction (2,1), giving spread = 4/5 = 0.8
    # But can we derive it from our QN (3,11,14,25)?
    # The passage angle spread = sin²(26.565°) = 0.2
    # That's spread = 1/5 exactly — direction (1,2) or equivalently (2,1)
    s_passage_measured = math.sin(math.radians(26.565)) * math.sin(math.radians(26.565))
    print(f"\n  PASSAGE ANGLE ANALYSIS:")
    print(f"    Measured passage angle: 26°33'54\" ≈ 26.565°")
    print(f"    Measured spread: sin²(26.565°) = {s_passage_measured:.6f}")
    print(f"    Nearest rational: 1/5 = {1/5} (error: {abs(s_passage_measured - 0.2):.6f})")
    print(f"    Direction for spread 1/5: (1,2) → tan(θ) = 1/2, θ = 26.565° ✓")

    # Connection to pyramid QN?
    # (3,11,14,25): is (1,2) or (2,1) related?
    # 14 = 7 × 2, 25 = 5 × 5. No direct element sharing.
    # BUT: the passage direction (1,2,3,5) IS the fundamental QN!
    # And b=3 of the pyramid QN is the d of the passage QN (1,2,3,5)
    b_pass, e_pass, d_pass, a_pass = 1, 2, 3, 5
    print(f"    Passage QN: ({b_pass},{e_pass},{d_pass},{a_pass})")
    print(f"    LINK: pyramid b={b} = passage d={d_pass} = 3")
    print(f"    LINK: passage a={a_pass} = 5, and pyramid a={a} = 5²")

    # King's Chamber proportions
    # 20 × 10 × ~11.18 cubits → 2:1:√5
    # In QA: the 2:1 ratio IS the passage direction
    # √5 ≈ 2.236 → the chamber height/width = √5/1
    # Spread of diagonal: 5/(1+5) = 5/6? Let's check
    print(f"\n  KING'S CHAMBER ANALYSIS:")
    print(f"    Dimensions: 20 × 10 × 11.18 cubits (Petrie)")
    print(f"    Length:Width = 20:10 = 2:1 = passage direction ✓")
    kc_height = Fraction(20 * 20 + 10 * 10).limit_denominator(1000)
    print(f"    If height² = length² + width² = 400+100 = 500")
    print(f"    Then height = √500 = 10√5 ≈ 22.36... but measured ~11.18")
    print(f"    Actually: 11.18 ≈ 5√5 = 5×2.236 = 11.18 ✓")
    print(f"    So chamber is 20 × 10 × 5√5, i.e., 4:2:√5 ratio")
    print(f"    Height/Width = √5/2 = {math.sqrt(5)/2:.4f}")
    print(f"    This is φ - 1/2 = {(1+math.sqrt(5))/2 - 0.5:.4f} (NOT exactly φ)")
    print(f"    BUT: (height/width)² = 5/4 → spread of chamber diagonal with floor")
    s_chamber = Fraction(5, 4 + 5)
    print(f"    Spread = 5/9 = {float(s_chamber):.6f}")
    print(f"    5/9 in QA: 5 = F_5, 9 = mod-9 modulus!")

    # Conic type
    I = C - F
    print(f"\n  CONIC DISCRIMINANT:")
    print(f"    I = C - F = {C} - {F} = {I}")
    if I > 0:
        print(f"    I > 0 → HYPERBOLA")
    elif I == 0:
        print(f"    I = 0 → PARABOLA")
    else:
        print(f"    I < 0 → ELLIPSE")

    # Par classification
    par_b = b % 4
    par_e = e % 4
    par_a = a % 4
    par_names = {0: "4-par", 1: "5-par", 2: "2-par", 3: "3-par"}
    print(f"\n  PAR CLASSIFICATION:")
    print(f"    b={b}: {b} mod 4 = {par_b} → {par_names[par_b]}")
    print(f"    e={e}: {e} mod 4 = {par_e} → {par_names[par_e]}")
    print(f"    d={d}: {d} mod 4 = {d%4} → {par_names[d%4]}")
    print(f"    a={a}: {a} mod 4 = {par_a} → {par_names[par_a]}")
    print(f"    Gender: b={b} is {'odd → MALE' if b % 2 == 1 else 'even → FEMALE'}")

    # Quantum ellipse
    axis_ratio = math.sqrt(a * b) / d
    ecc = e / d
    print(f"\n  QUANTUM ELLIPSE:")
    print(f"    Eccentricity = e/d = {e}/{d} = {Fraction(e,d)} = {ecc:.6f}")
    print(f"    Axis ratio = √(ab)/d = √({a*b})/{d} = {axis_ratio:.6f}")
    print(f"    Semi-major = d = {d}, semi-minor = √F = √{F} = {math.sqrt(F):.4f}")

    # Prime factors
    product = b * e * d * a
    print(f"\n  PRIME STRUCTURE:")
    print(f"    Product beda = {b}×{e}×{d}×{a} = {product}")
    # Factor
    def prime_factors(n):
        factors = {}
        d_f = 2
        while d_f * d_f <= n:
            while n % d_f == 0:
                factors[d_f] = factors.get(d_f, 0) + 1
                n //= d_f
            d_f += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    pf = prime_factors(product)
    print(f"    Factorization: {' × '.join(f'{p}^{k}' if k > 1 else str(p) for p, k in sorted(pf.items()))}")
    print(f"    Primes: {sorted(pf.keys())}")
    print(f"    b={b}=3, e={e}=11, d={d}=2×7, a={a}=5²")

    # Pi connection
    # Petrie cubit = 20.62 inches. 1000 cubits ≈ 20,620 inches.
    # Pi_QA = 20612/6561. Close but not exact.
    print(f"\n  PI_QA CONNECTION:")
    print(f"    Royal cubit = 20.62 inches (Petrie)")
    print(f"    1000 cubits = 20,620 inches")
    print(f"    Pi_QA = 20612/6561 = {20612/6561:.10f}")
    print(f"    2 × base / height = 2 × {base_cubits} / {height_cubits} = {2*base_cubits/height_cubits:.10f}")
    print(f"    π = {math.pi:.10f}")
    print(f"    Pyramid 2b/h vs π: error = {abs(2*base_cubits/height_cubits - math.pi):.6f} ({abs(2*base_cubits/height_cubits - math.pi)/math.pi*100:.3f}%)")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: SCORECARD — Pre-registered predictions vs measurements
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STEP 3: PRE-REGISTERED SCORECARD")
    print("=" * 80)

    predictions = [
        ("Seked = 5½ palms", "seked = 7e/d = 7×11/14 = 5.5", True,
         "The seked follows DIRECTLY from the QN. Not independent — it's the same ratio."),

        ("Passage angle ≈ 26.565°", "spread 1/5 → direction (1,2,3,5)",
         abs(s_passage_measured - 0.2) < 0.001,
         "The passage direction (1,2,3,5) is the FUNDAMENTAL QN. "
         "It shares b=3 with the pyramid QN (3,11,14,25). "
         "But 26.565° = arctan(1/2) is the simplest possible slope after 45°. "
         "Could be coincidence — any 2:1 construction rectangle gives this angle."),

        ("King's Chamber 2:1 ratio", "matches passage direction (1,2)",
         True,
         "The 2:1 length:width ratio = passage direction. Consistent. "
         "But 2:1 is the simplest nontrivial ratio — weak evidence alone."),

        ("C²+F²=G² for (3,11,14,25)", f"C={C}, F={F}, G={G}: {C*C}+{F*F}={C*C+F*F}={G*G}",
         C*C + F*F == G*G,
         "Always true for any QA tuple. Not a prediction — it's an identity."),

        ("2×base/height ≈ π", f"880/280 = {880/280:.6f} vs π = {math.pi:.6f}",
         abs(880/280 - math.pi) < 0.005,
         "0.05% error. Well-known. But this is a DERIVED property of the 14:11 ratio, "
         "not a separate QA prediction. 2×440/280 = 880/280 = 22/7 × (40/40) — "
         "the pyramid approximates π via 22/7, the simplest rational approx."),

        ("QN shares element with fundamental (1,1,2,3)", "b=3 = a of (1,1,2,3)",
         b == 3,  # b of pyramid = a of fundamental
         "The pyramid's b=3 equals the fundamental QN's a=3. "
         "Ben's Law of Harmonics: shared prime factor 3. "
         "But 3 is trivially common — weak evidence."),

        ("Direction (14,11) is primitive", f"gcd(14,11) = {gcd(14,11)}",
         gcd(d, e) == 1,
         "gcd(14,11)=1. The direction is primitive (coprime). Required for QA."),
    ]

    pass_count = 0
    total = len(predictions)

    for label, detail, passed, note in predictions:
        status = "PASS" if passed else "FAIL"
        if passed:
            pass_count += 1
        print(f"\n  [{status}] {label}")
        print(f"         {detail}")
        print(f"         NOTE: {note}")

    print(f"\n{'='*80}")
    print(f"VERDICT: {pass_count}/{total} predictions passed")
    print(f"{'='*80}\n")

    print(f"  HONEST ASSESSMENT:")
    print(f"  The Great Pyramid's exterior slope gives QN (3, 11, 14, 25).")
    print(f"  This is a VALID QA tuple (primitive, coprime, all identities hold).")
    print(f"  The seked, π approximation, and C²+F²=G² identity all follow")
    print(f"  automatically from the 14:11 ratio — they're not independent tests.")
    print()
    print(f"  The ONLY potentially independent prediction is:")
    print(f"  • Passage angle = arctan(1/2) → direction (1,2,3,5)")
    print(f"  • This shares b=3 with the pyramid QN")
    print(f"  • But arctan(1/2) is the simplest slope after 45° — it could be")
    print(f"    chosen for ease of construction, not QA reasons.")
    print()
    print(f"  TIER: This analysis is TIER 1 at best (exact reformulation of")
    print(f"  known measurements in QA language). It does NOT reach Tier 2")
    print(f"  (structural correspondence) because we cannot distinguish")
    print(f"  'the builders knew QA' from 'the builders used simple ratios")
    print(f"  that happen to be QA-compatible.'")
    print()
    print(f"  The 14:11 slope ratio IS a QA direction. The 2:1 passage ratio")
    print(f"  IS a QA direction. But saying the pyramid 'encodes QA' is a")
    print(f"  stronger claim than the data supports. What we CAN say:")
    print(f"  'The Great Pyramid's proportions are exactly expressible as")
    print(f"  QA directions and the RT framework replaces all trigonometric")
    print(f"  analysis with exact rational arithmetic.'")
    print()
    print(f"  TO REACH TIER 2: We would need to show that the pyramid's")
    print(f"  proportions are SPECIFICALLY QA (not just any simple ratios).")
    print(f"  Candidate: does QN (3,11,14,25) predict a measurement that")
    print(f"  has NOT been noticed before? E.g., the quantum ellipse of this")
    print(f"  QN has specific axis ratio and eccentricity — do these appear")
    print(f"  anywhere in the pyramid's geometry?")


if __name__ == "__main__":
    main()
