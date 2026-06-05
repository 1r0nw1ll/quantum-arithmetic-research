# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Pyth-2 Ch.XVI Cattle Problem: "
    "1/m+1/(m+1)=(2m+1)/(m(m+1)); bull solution W1=2226,X1=1602,Y1=891,Z1=1580; "
    "divisibility conditions; fractional remainder 1/6*X1+11/20*Z1+29/42*W1=3*Y1; "
    "minimum integer solution Y1=891); "
    "Theorem NT: 'white bulls', 'black cows', 'Archimedes', 'ellipse diagram' are "
    "observer-projection labels; QA layer = integer equations + divisibility; no float state"
)

"""
Cert [372] — QA Pyth-2 Cattle Problem Integer Structure (Ch.XVI)

Source: Iverson (1993) Pythagorean Arithmetic Vol II, Chapter XVI pp.112-122+
  p.113-114: Seven cattle equations with fraction coefficients from {1/m+1/(m+1)};
             three bull equations: W1=Y1+5/6*X1; X1=Y1+9/20*Z1; Z1=Y1+13/42*W1
  p.115-116: Solution W1=2226, X1=1602, Y1=891, Z1=1580 (from Wurm 19th century);
             'Only two values must be given directly' — underdetermined system
  p.117: 'X1 must be a mod 6(0) number'; 'Z1 is a mod 20(0) number'; 'W1 is a mod 42(0) number';
         '(W1-Y1) is a mod 5(0) number'; '(X1-Y1) is a mod 9(0) number';
         '(Z1-Y1) is a mod 13(0) number'
  p.117: '1/6*X1 + 11/20*Z1 + 29/42*W1 = 3*Y1'
         values: '1/6*X1=267; 11/20*Z1=869; and 29/42*W1=1537'

Five claims (QA integer structure of the Cattle Problem bull subsystem):
  C1: Fraction coefficients: 1/m+1/(m+1)=(2m+1)/(m*(m+1)); cattle problem uses m=2,3,4,5,6
      giving 5/6; 7/12; 9/20; 11/30; 13/42
  C2: Bull solution satisfies all 3 equations: W1=2226,X1=1602,Y1=891,Z1=1580 (exact integer)
  C3: Divisibility conditions: 6|X1, 20|Z1, 42|W1; 5|(W1-Y1), 9|(X1-Y1), 13|(Z1-Y1)
  C4: Fractional remainder identity: 1/6*X1 + 11/20*Z1 + 29/42*W1 = 3*Y1 = 2673
      values: 267 + 869 + 1537 = 2673 = 3*891
  C5: Minimum integer solution: smallest Y1 with all-integer W1,X1,Z1 is Y1=891=3*297;
      integer solutions occur at Y1=891j for j=1,2,3,...
"""

from fractions import Fraction


def check_c1() -> tuple[bool, str]:
    """1/m + 1/(m+1) = (2m+1)/(m*(m+1)) for m=2,3,4,5,6."""
    results = {}
    for m in range(2, 7):
        f = Fraction(1, m) + Fraction(1, m + 1)
        expected = Fraction(2 * m + 1, m * (m + 1))
        assert f == expected, f"1/{m}+1/{m+1}={f} != {expected}"
        results[m] = f
    # Verify specific cattle fraction coefficients used in Ch.XVI:
    # Bulls: 5/6 (m=2, eq1); 9/20 (m=4, eq2); 13/42 (m=6, eq3)
    # Cows: 7/12 (m=3, eq4); 9/20 (m=4, eq5); 11/30 (m=5, eq6); 13/42 (m=6, eq7)
    assert results[2] == Fraction(5, 6)
    assert results[3] == Fraction(7, 12)
    assert results[4] == Fraction(9, 20)
    assert results[5] == Fraction(11, 30)
    assert results[6] == Fraction(13, 42)
    # Also verify the general formula: (2m+1)/(m*(m+1)) = 1/(m*(m+1)/(2m+1))... just verify
    for m in range(2, 20):
        f = Fraction(1, m) + Fraction(1, m + 1)
        assert f == Fraction(2 * m + 1, m * (m + 1))
    return True, (
        f"1/m+1/(m+1)=(2m+1)/(m(m+1)) verified for m=2..19; "
        f"cattle-specific fractions: m=2→5/6; m=3→7/12; m=4→9/20; m=5→11/30; m=6→13/42; "
        f"Iverson: 'W1=Y1+(1/2+1/3)*X1; X1=Y1+(1/4+1/5)*Z1; Z1=Y1+(1/6+1/7)*W1' ✓"
    )


def check_c2() -> tuple[bool, str]:
    """Bull solution W1=2226, X1=1602, Y1=891, Z1=1580 satisfies all 3 bull equations."""
    W1, X1, Y1, Z1 = 2226, 1602, 891, 1580
    eq1 = Y1 + Fraction(5, 6) * X1
    eq2 = Y1 + Fraction(9, 20) * Z1
    eq3 = Y1 + Fraction(13, 42) * W1
    assert eq1 == W1, f"Eq1 failed: Y1+5/6*X1={eq1} != W1={W1}"
    assert eq2 == X1, f"Eq2 failed: Y1+9/20*Z1={eq2} != X1={X1}"
    assert eq3 == Z1, f"Eq3 failed: Y1+13/42*W1={eq3} != Z1={Z1}"
    # All values are positive integers
    assert all(isinstance(v, int) and v > 0 for v in [W1, X1, Y1, Z1])
    return True, (
        f"Bull solution verified: W1=2226, X1=1602, Y1=891, Z1=1580 satisfies "
        f"W1=Y1+5/6*X1 ({Y1}+5/6*{X1}={eq1}=2226); "
        f"X1=Y1+9/20*Z1 ({Y1}+9/20*{Z1}={eq2}=1602); "
        f"Z1=Y1+13/42*W1 ({Y1}+13/42*{W1}={eq3}=1580); "
        f"Iverson/Wurm (19th century) solution; all positive integers ✓"
    )


def check_c3() -> tuple[bool, str]:
    """Divisibility conditions: 6|X1, 20|Z1, 42|W1; 5|(W1-Y1), 9|(X1-Y1), 13|(Z1-Y1)."""
    W1, X1, Y1, Z1 = 2226, 1602, 891, 1580
    # Modular conditions on the values themselves
    assert X1 % 6 == 0, f"6 does not divide X1={X1}"
    assert Z1 % 20 == 0, f"20 does not divide Z1={Z1}"
    assert W1 % 42 == 0, f"42 does not divide W1={W1}"
    # Modular conditions on differences
    assert (W1 - Y1) % 5 == 0, f"5 does not divide W1-Y1={W1-Y1}"
    assert (X1 - Y1) % 9 == 0, f"9 does not divide X1-Y1={X1-Y1}"
    assert (Z1 - Y1) % 13 == 0, f"13 does not divide Z1-Y1={Z1-Y1}"
    # Derive why: W1-Y1 = 5/6*X1 → 5-part of X1/6; X1-Y1=9/20*Z1; Z1-Y1=13/42*W1
    assert W1 - Y1 == Fraction(5, 6) * X1   # = 5/6 * 1602 = 1335
    assert X1 - Y1 == Fraction(9, 20) * Z1  # = 9/20 * 1580 = 711
    assert Z1 - Y1 == Fraction(13, 42) * W1 # = 13/42 * 2226 = 689
    return True, (
        f"Divisibility conditions: "
        f"6|X1={X1}: X1/6={X1//6}; 20|Z1={Z1}: Z1/20={Z1//20}; 42|W1={W1}: W1/42={W1//42}; "
        f"5|(W1-Y1={(W1-Y1)}): (W1-Y1)/5={(W1-Y1)//5}; "
        f"9|(X1-Y1={(X1-Y1)}): (X1-Y1)/9={(X1-Y1)//9}; "
        f"13|(Z1-Y1={(Z1-Y1)}): (Z1-Y1)/13={(Z1-Y1)//13}; "
        f"Iverson: 'X1 must be mod 6(0); Z1 mod 20(0); W1 mod 42(0)' ✓"
    )


def check_c4() -> tuple[bool, str]:
    """Fractional remainder identity: 1/6*X1 + 11/20*Z1 + 29/42*W1 = 3*Y1 = 2673."""
    W1, X1, Y1, Z1 = 2226, 1602, 891, 1580
    rem1 = Fraction(1, 6) * X1    # = 267
    rem2 = Fraction(11, 20) * Z1  # = 869
    rem3 = Fraction(29, 42) * W1  # = 1537
    total = rem1 + rem2 + rem3
    assert rem1 == 267, f"1/6*X1={rem1} != 267"
    assert rem2 == 869, f"11/20*Z1={rem2} != 869"
    assert rem3 == 1537, f"29/42*W1={rem3} != 1537"
    assert total == 3 * Y1, f"Sum={total} != 3*Y1={3*Y1}"
    # Algebraic derivation: 1/6 = 1 - 5/6; 11/20 = 1 - 9/20; 29/42 = 1 - 13/42
    # So rem1 + rem2 + rem3 = X1+Z1+W1 - (5/6*X1 + 9/20*Z1 + 13/42*W1)
    #   = X1+Z1+W1 - ((W1-Y1)+(X1-Y1)+(Z1-Y1)) = X1+Z1+W1-(W1+X1+Z1-3*Y1) = 3*Y1 ✓
    bracket = Fraction(5, 6) * X1 + Fraction(9, 20) * Z1 + Fraction(13, 42) * W1
    check = (W1 + X1 + Z1) - bracket
    assert check == 3 * Y1
    return True, (
        f"Fractional remainder identity: "
        f"1/6*X1={rem1}; 11/20*Z1={rem2}; 29/42*W1={rem3}; "
        f"sum={total} = 3*Y1=3*{Y1}={3*Y1}; "
        f"algebraic proof: X1+Z1+W1-(5/6*X1+9/20*Z1+13/42*W1) = X1+Z1+W1-(W1-Y1+X1-Y1+Z1-Y1)=3*Y1; "
        f"Iverson: '1/6*X1=267; 11/20*Z1=869; and 29/42*W1=1537' ✓"
    )


def check_c5() -> tuple[bool, str]:
    """Minimum integer solution: Y1=891; integer solutions at Y1=891*j for j=1,2,3,..."""
    # The system has one degree of freedom (underdetermined: 3 eq, 4 unknowns)
    # Parameterize by Y1: solve for W1,X1,Z1 as rational functions of Y1
    # From the equations (algebraically):
    # W1 = Y1 + 5/6*X1
    # X1 = Y1 + 9/20*Z1
    # Z1 = Y1 + 13/42*W1
    # Substituting: W1 = Y1 + 5/6*(Y1 + 9/20*(Y1 + 13/42*W1))
    # W1(1 - 5/6*9/20*13/42) = Y1*(1 + 5/6 + 5/6*9/20)
    coeff_W = 1 - Fraction(5, 6) * Fraction(9, 20) * Fraction(13, 42)
    coeff_Y = 1 + Fraction(5, 6) + Fraction(5, 6) * Fraction(9, 20)
    # W1 = Y1 * coeff_Y / coeff_W
    W1_per_Y1 = coeff_Y / coeff_W
    X1_per_Y1 = (1 + Fraction(9, 20) * Fraction(13, 42) * W1_per_Y1 + Fraction(9, 20))
    # Actually let's just parameterize directly:
    # W1 = alpha*Y1; X1 = beta*Y1; Z1 = gamma*Y1 for rational alpha, beta, gamma
    # From eq3: Z1 = Y1 + 13/42*W1 = Y1(1 + 13/42*alpha) = gamma*Y1
    # From eq2: X1 = Y1 + 9/20*Z1 = Y1(1 + 9/20*gamma) = beta*Y1
    # From eq1: W1 = Y1 + 5/6*X1 = Y1(1 + 5/6*beta) = alpha*Y1
    # So: alpha = 1 + 5/6*beta; beta = 1 + 9/20*gamma; gamma = 1 + 13/42*alpha
    # Substitute: alpha = 1 + 5/6*(1 + 9/20*(1 + 13/42*alpha))
    # alpha - 5/6*9/20*13/42*alpha = 1 + 5/6 + 5/6*9/20
    a = Fraction(5, 6) * Fraction(9, 20) * Fraction(13, 42)
    lhs_coeff = 1 - a
    rhs = 1 + Fraction(5, 6) + Fraction(5, 6) * Fraction(9, 20)
    alpha = rhs / lhs_coeff
    gamma = 1 + Fraction(13, 42) * alpha
    beta = 1 + Fraction(9, 20) * gamma
    # Verify consistency
    assert alpha == 1 + Fraction(5, 6) * beta, f"alpha inconsistent"
    assert beta == 1 + Fraction(9, 20) * gamma, f"beta inconsistent"
    assert gamma == 1 + Fraction(13, 42) * alpha, f"gamma inconsistent"
    # Integer solutions require alpha*Y1, beta*Y1, gamma*Y1 all integers
    # So Y1 must be divisible by lcm(alpha.denominator, beta.denominator, gamma.denominator)
    from math import lcm
    min_Y1 = lcm(alpha.denominator, beta.denominator, gamma.denominator)
    # Verify minimum integer solution
    W1_min = alpha * min_Y1
    X1_min = beta * min_Y1
    Z1_min = gamma * min_Y1
    assert W1_min.denominator == 1 and X1_min.denominator == 1 and Z1_min.denominator == 1
    assert min_Y1 == 891, f"Minimum Y1={min_Y1} != 891"
    assert int(W1_min) == 2226 and int(X1_min) == 1602 and int(Z1_min) == 1580
    # Verify that Y1=297 gives non-integer Z1
    Z1_297 = gamma * 297
    assert Z1_297.denominator != 1, f"Z1 should be non-integer for Y1=297"
    return True, (
        f"Minimum integer solution: Y1=891; "
        f"W1=alpha*Y1, X1=beta*Y1, Z1=gamma*Y1 where alpha={alpha}, beta={beta}, gamma={gamma}; "
        f"lcm(denoms)={min_Y1}; (W1,X1,Y1,Z1)=({int(W1_min)},{int(X1_min)},{min_Y1},{int(Z1_min)}); "
        f"Y1=297 gives Z1={Z1_297} (non-integer); integer solutions at Y1=891j for j=1,2,3,...; "
        f"Iverson: underdetermined 3-equation 4-unknown system solved via Fibonacci bead structure ✓"
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
        raise RuntimeError(f"cert [372] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
