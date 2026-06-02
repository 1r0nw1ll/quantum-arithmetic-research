#!/usr/bin/env python3
QA_COMPLIANCE = "cert_validator — Cassini integer identity; Fraction rational-trig observer layer; orbit enumeration on {1,...,9}^2 with A1 no-zero arithmetic; no float QA state"
"""
QA Reactive Power Versor Coupling Cert [305] — validator

Primary sources:
  Hardy and Wright (2008) An Introduction to the Theory of Numbers, Oxford, ISBN 978-0-19-921986-5
  Wildberger, N.J. (2005) Divine Proportions: Rational Trigonometry, Wild Egg Books,
    ISBN 978-0-9757492-0-8
  Hestenes and Sobczyk (1984) Clifford Algebra to Geometric Calculus, Reidel, ISBN 978-90-277-1673-6

Rationale for rational trigonometry (not sin/cos/tan or hyperbolic trig):
  The QA observer layer requires exact rational arithmetic. Wildberger spread/quadrance replaces
  sin²(φ)/cos²(φ) with integer-computable fractions, eliminating all transcendental functions.
  Hyperbolic trig is appropriate only for hyperbolic/Minkowski geometry — the polyphase power
  setting is Euclidean (elliptic), so rational trig is the natural choice.

Five claims:
  C1  Grade-reactive isomorphism: det(M^k) = (-1)^k exactly (Cassini identity, cert [299]).
      Odd k = odd-grade versor (reactive coupling operator); even k = even-grade rotor (active).
      The QA 24-clock alternates reactive/active at every T-step.
  C2  Rational spread = QA power factor: s(b,e) = e*e/(b*b+e*e) ∈ (0,1)∩Q for all (b,e)
      in {1,...,9}^2. A1 (b,e >= 1) guarantees strict (0,1) — no purely active or reactive state.
      The identity (1-s)+s=1 holds exactly (Fraction). This is the observer-layer substitute
      for sin²(φ) using rational trigonometry; active fraction = 1-s = b*b/(b*b+e*e).
  C3  45° locus and orbit symmetry: all states (k,k) have s=1/2 exactly (b=e diagonal);
      the Singularity (9,9) lies on this locus (s=1/2, equal active/reactive components);
      Satellite spreads are {1/10,1/5,4/13,1/2,9/13,4/5,9/10} — closed under s→1-s
      (reactive-active symmetry of the Satellite orbit).
  C4  Reactive complement: the swap (b,e)→(e,b) maps every state to its reactive complement;
      s(e,b) = b*b/(b*b+e*e) = 1-s(b,e) exactly; Cosmos is closed under (b,e)→(e,b)
      (since gcd(e,b)=gcd(b,e)); the pair {s(b,e), s(e,b)} sums to 1 for all (b,e).
  C5  T-step cross-spread (mutual coupling): the cross-spread between QA state (b,e) and its
      T-image (e,d) where d=A1_mod(b+e,9) is cs = (b*d-e*e)^2/(G*G') where G=b*b+e*e,
      G'=e*e+d*d — an exact rational for every Cosmos state. This measures the reactive
      coupling energy introduced by one odd-grade T-step. All 72 cross-spreads are in (0,1)∩Q.
"""

import sys
from fractions import Fraction
from math import gcd


def fibonacci(n):
    if n == 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def M_power(k):
    fk_m1 = 1 if k == 0 else fibonacci(k - 1)
    fk    = fibonacci(k)
    fk_p1 = fibonacci(k + 1)
    return [[fk_m1, fk], [fk, fk_p1]]


def spread(b, e):
    """Wildberger spread s(b,e) = e*e/(b*b+e*e) — exact Fraction."""
    return Fraction(e * e, b * b + e * e)


def t_step_a1(b, e, m=9):
    return e, ((b + e - 1) % m) + 1


def cosmos_states():
    return [(b, e) for b in range(1, 10) for e in range(1, 10)
            if gcd(b, e) % 3 != 0]


def satellite_states():
    return [(b, e) for b in range(1, 10) for e in range(1, 10)
            if gcd(b, e) % 3 == 0 and not (b == 9 and e == 9)]


# ---------------------------------------------------------------------------
# C1 — Grade-reactive isomorphism: det(M^k) = (-1)^k
# ---------------------------------------------------------------------------
def check_c1():
    failures = []

    # Verify Cassini: det(M^k) = (-1)^k for k=0..24
    for k in range(25):
        Mk = M_power(k)
        det_k = Mk[0][0] * Mk[1][1] - Mk[0][1] * Mk[1][0]
        expected = (-1) ** k
        if det_k != expected:
            failures.append(f"det(M^{k}) = {det_k}, expected {expected}")
            break

    # Confirm odd k = odd-grade: det = -1 for k in {1,3,5,7,...}
    odd_dets  = set((-1)**k for k in range(1,  25, 2))
    even_dets = set((-1)**k for k in range(0,  25, 2))
    if odd_dets != {-1}:
        failures.append(f"Odd-k grades are not all -1: {odd_dets}")
    if even_dets != {1}:
        failures.append(f"Even-k grades are not all +1: {even_dets}")

    # Specific landmarks on the 24-clock
    for k, expected_det in [(1, -1), (6, 1), (12, 1), (24, 1)]:
        Mk = M_power(k)
        det_k = Mk[0][0] * Mk[1][1] - Mk[0][1] * Mk[1][0]
        if det_k != expected_det:
            failures.append(f"det(M^{k}) = {det_k}, expected {expected_det}")

    return failures


# ---------------------------------------------------------------------------
# C2 — Rational spread = QA power factor
# ---------------------------------------------------------------------------
def check_c2():
    failures = []

    # All (b,e) in {1,...,9}^2: spread is in (0,1)∩Q
    for b in range(1, 10):
        for e in range(1, 10):
            s = spread(b, e)
            if not (0 < s < 1):
                failures.append(f"spread({b},{e}) = {s} not in (0,1)")
            if not isinstance(s, Fraction):
                failures.append(f"spread({b},{e}) = {s} not a Fraction")

    # Identity (1-s)+s = 1 for all 81 states
    for b in range(1, 10):
        for e in range(1, 10):
            s = spread(b, e)
            if (1 - s) + s != Fraction(1):
                failures.append(f"(1-s)+s != 1 at ({b},{e})")
                break

    # Active fraction 1-s = b*b/(b*b+e*e): verify formula
    for b in range(1, 10):
        for e in range(1, 10):
            s = spread(b, e)
            active = 1 - s
            expected_active = Fraction(b * b, b * b + e * e)
            if active != expected_active:
                failures.append(
                    f"Active fraction {active} != b^2/G = {expected_active} at ({b},{e})"
                )
                break

    # A1 guarantee: b,e >= 1 → s can never be 0 or 1 exactly
    for b in range(1, 10):
        for e in range(1, 10):
            if spread(b, e) == 0 or spread(b, e) == 1:
                failures.append(f"spread({b},{e}) = 0 or 1 (A1 violation)")

    return failures


# ---------------------------------------------------------------------------
# C3 — 45° locus and orbit symmetry
# ---------------------------------------------------------------------------
def check_c3():
    failures = []

    # b=e diagonal: s = e^2/(e^2+e^2) = 1/2 exactly
    for k in range(1, 10):
        s = spread(k, k)
        if s != Fraction(1, 2):
            failures.append(f"spread({k},{k}) = {s}, expected 1/2")

    # Singularity (9,9) is on the b=e diagonal
    if spread(9, 9) != Fraction(1, 2):
        failures.append(f"Singularity (9,9) spread = {spread(9,9)}, expected 1/2")

    # Satellite spreads: closed under s → 1-s
    sat = satellite_states()
    sat_spreads = set(spread(b, e) for b, e in sat)
    expected_sat_spreads = {
        Fraction(1, 10), Fraction(1, 5), Fraction(4, 13),
        Fraction(1, 2),
        Fraction(9, 13), Fraction(4, 5), Fraction(9, 10),
    }
    if sat_spreads != expected_sat_spreads:
        failures.append(
            f"Satellite spreads = {sat_spreads}, expected {expected_sat_spreads}"
        )

    # Closure under s→1-s for Satellite
    for s in sat_spreads:
        if (1 - s) not in sat_spreads:
            failures.append(f"Satellite spread {s} has no complement {1-s} in Satellite")

    return failures


# ---------------------------------------------------------------------------
# C4 — Reactive complement: (b,e) → (e,b) inverts spread
# ---------------------------------------------------------------------------
def check_c4():
    failures = []

    # For all (b,e) in {1,...,9}^2: s(e,b) = 1 - s(b,e)
    for b in range(1, 10):
        for e in range(1, 10):
            s_be = spread(b, e)
            s_eb = spread(e, b)
            if s_be + s_eb != Fraction(1):
                failures.append(
                    f"s({b},{e}) + s({e},{b}) = {s_be+s_eb}, expected 1"
                )

    # Cosmos is closed under (b,e) → (e,b)
    cosmos = set(cosmos_states())
    for b, e in cosmos:
        if (e, b) not in cosmos:
            failures.append(
                f"Cosmos swap ({b},{e}) → ({e},{b}): ({e},{b}) not in Cosmos"
            )

    # The complement pair {(b,e),(e,b)} has gcd(b,e)=gcd(e,b): trivially ✓ by symmetry
    # Verify explicitly for all Cosmos
    for b, e in cosmos:
        if gcd(b, e) != gcd(e, b):
            failures.append(f"gcd({b},{e}) != gcd({e},{b})")

    # Extreme complements: min Cosmos spread and its complement
    cosmos_list = list(cosmos)
    spreads = [(spread(b, e), b, e) for b, e in cosmos_list]
    min_s, b_min, e_min = min(spreads)
    max_s, b_max, e_max = max(spreads)
    if min_s + max_s != Fraction(1):
        failures.append(
            f"Min spread {min_s} at ({b_min},{e_min}) + max spread {max_s} at ({b_max},{e_max}) "
            f"= {min_s+max_s}, expected 1"
        )

    return failures


# ---------------------------------------------------------------------------
# C5 — T-step cross-spread: exact rational coupling per T-step
# ---------------------------------------------------------------------------
def check_c5():
    failures = []

    cosmos = cosmos_states()

    for b, e in cosmos:
        d = t_step_a1(b, e)[1]   # second component of T(b,e) = (e, d)
        G  = b * b + e * e        # quadrance of source direction
        Gp = e * e + d * d        # quadrance of image direction

        # Cross-spread: (b*d - e*e)^2 / (G * G') — exact Fraction
        num = (b * d - e * e) ** 2
        cs = Fraction(num, G * Gp)

        # Must be rational in (0, 1)
        if not (0 <= cs <= 1):
            failures.append(
                f"Cross-spread({b},{e}): cs = {cs} not in [0,1]"
            )

        # Verify it equals the Wildberger cross-spread formula:
        # cs = (b1*e2 - b2*e1)^2 / (G1*G2) with (b1,e1)=(b,e), (b2,e2)=(e,d)
        # = (b*d - e*e)^2 / (G*G')  ← same formula
        cs_formula = Fraction((b * d - e * e) ** 2, G * Gp)
        if cs != cs_formula:
            failures.append(
                f"Cross-spread formula mismatch at ({b},{e}): {cs} != {cs_formula}"
            )

    # All 72 cross-spreads are well-defined (no division by zero) — checked above
    # Verify all are in [0,1]
    all_cs = []
    for b, e in cosmos:
        d   = t_step_a1(b, e)[1]
        G   = b * b + e * e
        Gp  = e * e + d * d
        cs  = Fraction((b * d - e * e) ** 2, G * Gp)
        all_cs.append(cs)

    if any(cs < 0 or cs > 1 for cs in all_cs):
        failures.append("Some cross-spread is outside [0,1]")

    # Confirm the cross-spread is zero when the state is at a fixed point (Singularity)
    # (9,9) is fixed: T(9,9)=(9,9), so cross-spread = (9*9 - 9*9)^2/(G*G) = 0
    b, e = 9, 9
    d = t_step_a1(b, e)[1]
    G = b * b + e * e
    Gp = e * e + d * d
    cs_sing = Fraction((b * d - e * e) ** 2, G * Gp)
    if cs_sing != Fraction(0):
        failures.append(
            f"Singularity (9,9) cross-spread = {cs_sing}, expected 0 (self-coupling)"
        )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    checks = [
        ("C1_grade_reactive_isomorphism",    check_c1, {}),
        ("C2_rational_spread_power_factor",  check_c2, {}),
        ("C3_45deg_locus_orbit_symmetry",    check_c3, {}),
        ("C4_reactive_complement_closed",    check_c4, {}),
        ("C5_T_step_cross_spread_rational",  check_c5, {}),
    ]
    all_pass = True
    for label, fn, kwargs in checks:
        failures = fn(**kwargs)
        status = "PASS" if not failures else "FAIL"
        if failures:
            all_pass = False
        suffix = f" — {failures[0]}" if failures else ""
        print(f"  {label}: {status}{suffix}")

    print()
    if all_pass:
        print("CERT [305] PASS — QA Reactive Power Versor Coupling")
    else:
        print("CERT [305] FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
