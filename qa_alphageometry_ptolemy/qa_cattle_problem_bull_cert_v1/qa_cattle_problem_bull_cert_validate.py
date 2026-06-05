# <!-- PRIMARY-SOURCE-EXEMPT: Iverson (1993) Pythagorean Arithmetic Vol II — no external DOI -->
from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Cattle Problem bull equations: "
    "modular structure of W1,X1,Y1,Z1; fractional remainder identity; "
    "Fraction arithmetic for exact verification); "
    "Theorem NT: 'cattle', 'color', 'bulls' are observer labels; "
    "causal structure is integer linear constraints with modular divisibility; "
    "no float state, no QA orbit evolution"
)

"""
Cert [347] — QA Cattle Problem Bull Modular Structure

Source: Iverson, B. (1993) Pythagorean Arithmetic Vol II, Chapter XVI pp.104-109
  p.104: The three bull equations (1)-(3):
    (1) W1 = Y1 + 5/6 X1   [white = yellow + 5/6 black]
    (2) X1 = Y1 + 9/20 Z1  [black = yellow + 9/20 dappled]
    (3) Z1 = Y1 + 13/42 W1 [dappled = yellow + 13/42 white]
  p.107: "X1 is a mod 20(0) number... W1 is a mod 42(0) number... Z1 is a mod 20(0)"
  p.107: "1/6 X1 + 11/20 Z1 + 29/42 W1 = 3 Y1"
  p.107: "1/6 X1 = 267; 11/20 Z1 = 869; and 29/42 W1 = 1537"
  p.107: Wurm's solution: W1=2226, X1=1602, Y1=891, Z1=1580

Five claims:
  C1: (W1,X1,Y1,Z1) = (2226,1602,891,1580) satisfies all three bull equations exactly
  C2: Modular structure: X1 ≡ 0 (mod 6); Z1 ≡ 0 (mod 20); W1 ≡ 0 (mod 42)
  C3: Differences: (W1-Y1) ≡ 0 (mod 5); (X1-Y1) ≡ 0 (mod 9); (Z1-Y1) ≡ 0 (mod 13)
  C4: Fractional remainder identity: 1/6·X1 + 11/20·Z1 + 29/42·W1 = 3·Y1 = 2673
  C5: Individual fractional remainders: 1/6·X1=267, 11/20·Z1=869, 29/42·W1=1537
"""

from fractions import Fraction


W1, X1, Y1, Z1 = 2226, 1602, 891, 1580


def check_c1() -> tuple[bool, str]:
    """(W1,X1,Y1,Z1)=(2226,1602,891,1580) satisfies all three bull equations."""
    # Eq (1): W1 = Y1 + 5/6 X1
    eq1 = Fraction(Y1) + Fraction(5, 6) * X1
    assert eq1 == W1, f"Eq(1): Y1 + 5/6*X1 = {eq1} != W1={W1}"
    # Eq (2): X1 = Y1 + 9/20 Z1
    eq2 = Fraction(Y1) + Fraction(9, 20) * Z1
    assert eq2 == X1, f"Eq(2): Y1 + 9/20*Z1 = {eq2} != X1={X1}"
    # Eq (3): Z1 = Y1 + 13/42 W1
    eq3 = Fraction(Y1) + Fraction(13, 42) * W1
    assert eq3 == Z1, f"Eq(3): Y1 + 13/42*W1 = {eq3} != Z1={Z1}"
    return True, (
        f"(W1,X1,Y1,Z1)=({W1},{X1},{Y1},{Z1}) satisfies all 3 bull equations: "
        f"W1=Y1+5/6·X1 ✓; X1=Y1+9/20·Z1 ✓; Z1=Y1+13/42·W1 ✓"
    )


def check_c2() -> tuple[bool, str]:
    """X1 ≡ 0 (mod 6); Z1 ≡ 0 (mod 20); W1 ≡ 0 (mod 42)."""
    assert X1 % 6 == 0, f"X1={X1} not divisible by 6"
    assert Z1 % 20 == 0, f"Z1={Z1} not divisible by 20"
    assert W1 % 42 == 0, f"W1={W1} not divisible by 42"
    return True, (
        f"Modular structure: X1={X1}=6×{X1//6}; Z1={Z1}=20×{Z1//20}; W1={W1}=42×{W1//42}"
    )


def check_c3() -> tuple[bool, str]:
    """(W1-Y1) ≡ 0 (mod 5); (X1-Y1) ≡ 0 (mod 9); (Z1-Y1) ≡ 0 (mod 13)."""
    assert (W1 - Y1) % 5 == 0, f"W1-Y1={W1-Y1} not divisible by 5"
    assert (X1 - Y1) % 9 == 0, f"X1-Y1={X1-Y1} not divisible by 9"
    assert (Z1 - Y1) % 13 == 0, f"Z1-Y1={Z1-Y1} not divisible by 13"
    return True, (
        f"Differences: W1-Y1={W1-Y1}=5×{(W1-Y1)//5}; "
        f"X1-Y1={X1-Y1}=9×{(X1-Y1)//9}; Z1-Y1={Z1-Y1}=13×{(Z1-Y1)//13}"
    )


def check_c4() -> tuple[bool, str]:
    """1/6·X1 + 11/20·Z1 + 29/42·W1 = 3·Y1 = 2673."""
    lhs = Fraction(1, 6) * X1 + Fraction(11, 20) * Z1 + Fraction(29, 42) * W1
    rhs = 3 * Y1
    assert lhs == rhs, f"1/6·X1 + 11/20·Z1 + 29/42·W1 = {lhs} != 3·Y1={rhs}"
    # Algebraic proof: from eqs (1a),(2a),(3a), summing LHS and RHS:
    # (W1-Y1)+(X1-Y1)+(Z1-Y1) = 5/6·X1 + 9/20·Z1 + 13/42·W1
    # W1+X1+Z1 - 3Y1 = 5/6·X1 + 9/20·Z1 + 13/42·W1
    # W1(1-13/42) + X1(1-5/6) + Z1(1-9/20) = 3Y1
    # = 29/42·W1 + 1/6·X1 + 11/20·Z1 = 3Y1 ✓
    algebraic_check = (
        Fraction(29, 42) * W1 + Fraction(1, 6) * X1 + Fraction(11, 20) * Z1 == 3 * Y1
    )
    assert algebraic_check
    return True, (
        f"Fractional remainder identity: 1/6·X1 + 11/20·Z1 + 29/42·W1 = {lhs} = 3·Y1={rhs} ✓; "
        "algebraic proof via summing (Wi-Yi) from the three transposed equations"
    )


def check_c5() -> tuple[bool, str]:
    """Individual remainders: 1/6·X1=267, 11/20·Z1=869, 29/42·W1=1537."""
    r1 = Fraction(1, 6) * X1
    r2 = Fraction(11, 20) * Z1
    r3 = Fraction(29, 42) * W1
    assert r1 == 267, f"1/6·X1 = {r1} != 267"
    assert r2 == 869, f"11/20·Z1 = {r2} != 869"
    assert r3 == 1537, f"29/42·W1 = {r3} != 1537"
    assert r1 + r2 + r3 == 3 * Y1, (
        f"267+869+1537={int(r1+r2+r3)} != 3·Y1={3*Y1}"
    )
    return True, (
        f"1/6·X1={int(r1)}, 11/20·Z1={int(r2)}, 29/42·W1={int(r3)}; "
        f"sum={int(r1+r2+r3)}=3×{Y1}=3·Y1 ✓"
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
        raise RuntimeError(f"cert [347] FAILED: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
