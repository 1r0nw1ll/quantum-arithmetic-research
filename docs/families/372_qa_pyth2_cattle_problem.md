# [372] QA Pyth-2 Cattle Problem Integer Structure

**Family**: `qa_pyth2_cattle_problem_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XVI pp.112-122+

> *(p.113-114)*: Three bull equations: `W1=Y1+(1/2+1/3)X1`; `X1=Y1+(1/4+1/5)Z1`; `Z1=Y1+(1/6+1/7)W1`

> *(p.117)*: "X1 must be a mod 6(0) number... Z1 is a mod 20(0) number... W1 is a mod 42(0) number"

> *(p.117)*: "1/6 X1 + 11/20 Z1 + 29/42 W1 = 3 Y1... values: 267; 869; 1537"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 1/m+1/(m+1)=(2m+1)/(m(m+1)) for m=2..19; cattle values m=2,3,4,5,6 give 5/6,7/12,9/20,11/30,13/42 | PASS |
| C2 | W1=2226, X1=1602, Y1=891, Z1=1580 satisfies all 3 bull equations exactly (integer arithmetic) | PASS |
| C3 | 6\|X1, 20\|Z1, 42\|W1; 5\|(W1−Y1), 9\|(X1−Y1), 13\|(Z1−Y1) | PASS |
| C4 | 1/6×X1 + 11/20×Z1 + 29/42×W1 = 3×Y1 = 2673 (algebraic identity from the system) | PASS |
| C5 | Minimum integer solution is Y1=891; Y1=297 and Y1=594 give non-integer Z1; lcm(denoms)=891 | PASS |

## Mathematical Details

### C1: Cattle Fraction Coefficients

The seven cattle equations all use fractions of the form 1/m + 1/(m+1):

| m | 1/m+1/(m+1) | Equation |
|---|------------|---------|
| 2 | 5/6 | W1=Y1+5/6·X1 (bulls eq 1) |
| 3 | 7/12 | W2=7/12·(X1+X2) (cows eq 4) |
| 4 | 9/20 | X1=Y1+9/20·Z1 (bulls eq 2); X2=9/20·(Z1+Z2) (cows eq 5) |
| 5 | 11/30 | Z2=11/30·(Y1+Y2) (cows eq 6) |
| 6 | 13/42 | Z1=Y1+13/42·W1 (bulls eq 3); Y2=13/42·(W1+W2) (cows eq 7) |

**Formula**: 1/m + 1/(m+1) = (2m+1)/(m(m+1)) — verified for m=2..19.

### C2: Bull Solution (Exact Integer Verification)

The three-bull subsystem:
- W1 = Y1 + 5/6 × X1
- X1 = Y1 + 9/20 × Z1
- Z1 = Y1 + 13/42 × W1

**Solution**: W1=2226, X1=1602, Y1=891, Z1=1580

| Equation | LHS | RHS | Match |
|----------|-----|-----|-------|
| W1=Y1+5/6×X1 | 2226 | 891+5/6×1602=891+1335=2226 | ✓ |
| X1=Y1+9/20×Z1 | 1602 | 891+9/20×1580=891+711=1602 | ✓ |
| Z1=Y1+13/42×W1 | 1580 | 891+13/42×2226=891+689=1580 | ✓ |

Verified using `Fraction` arithmetic (exact, no floating point).

### C3: Divisibility Conditions

Requirement for integer results from fractional equations:

| Condition | Value | Result | Iverson |
|-----------|-------|--------|---------|
| 6\|X1 | 1602÷6=267 | ✓ | "X1 mod 6(0)" |
| 20\|Z1 | 1580÷20=79 | ✓ | "Z1 mod 20(0)" |
| 42\|W1 | 2226÷42=53 | ✓ | "W1 mod 42(0)" |
| 5\|(W1−Y1) | 1335÷5=267 | ✓ | "(W1−Y1) mod 5(0)" |
| 9\|(X1−Y1) | 711÷9=79 | ✓ | "(X1−Y1) mod 9(0)" |
| 13\|(Z1−Y1) | 689÷13=53 | ✓ | "(Z1−Y1) mod 13(0)" |

Note: W1/42 = (W1−Y1)/5 = 53 (same value!); X1/6 = (X1−Y1)/9·(6/9) ... actually 267, 79, 53 appear as the three quotients.

### C4: Fractional Remainder Identity

The remainders after the fractional parts are extracted:

1/6 × X1 + 11/20 × Z1 + 29/42 × W1 = **3 × Y1 = 2673**

Component values: 267 + 869 + 1537 = 2673

**Algebraic proof**: Note 1-5/6=1/6; 1-9/20=11/20; 1-13/42=29/42. So:
```
(1/6)X1 + (11/20)Z1 + (29/42)W1
= X1+Z1+W1 - [(5/6)X1 + (9/20)Z1 + (13/42)W1]
= X1+Z1+W1 - [(W1-Y1) + (X1-Y1) + (Z1-Y1)]
= X1+Z1+W1 - W1 - X1 - Z1 + 3Y1
= 3Y1 ✓
```

### C5: Minimum Integer Solution

The system has one free parameter (3 equations, 4 unknowns). Setting W1=α·Y1, X1=β·Y1, Z1=γ·Y1:

- α = 742/297
- β = 178/99  
- γ = 1580/891

For all three to be integers simultaneously, Y1 must be divisible by lcm(297, 99, 891) = **891**.

| Y1 | W1 | X1 | Z1 | All integer? |
|----|----|----|----|-------------|
| 297 | 742 | 534 | 1580/3 ≈ 526.67 | ✗ |
| 594 | 1484 | 1068 | 3160/3 ≈ 1053.33 | ✗ |
| **891** | **2226** | **1602** | **1580** | **✓** |
| 1782 | 4452 | 3204 | 3160 | ✓ |

## Theorem NT Note

"White bulls," "black cows," "Archimedes," "Cattle of Thrinacia," "Papyrus Rhind" are observer-projection labels. The QA discrete layer contains only:
- The fraction coefficients 1/m+1/(m+1) for m=2,3,4,5,6
- The integer constraint system with exact Fraction arithmetic
- The divisibility conditions on W1, X1, Z1
- The algebraic identity 1/6×X1+11/20×Z1+29/42×W1=3×Y1
- The minimum solution structure via lcm of rational denominators

**Depends on**: [371] Fibonacci Coprime Structure (coprimeness/divisibility context); [370] BABTHE (unit fractions from Rhind Papyrus — same ancient source as the cattle problem)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified the bull solution
in a fresh Fraction snippet (Y1+5/6·X1=W1, Y1+9/20·Z1=X1,
Y1+13/42·W1=Z1 all hold exactly for W1=2226,X1=1602,Y1=891,Z1=1580),
the divisibility conditions, and the C5 minimum-solution table
(lcm(297,99,891)=891; Y1=297→Z1=1580/3 and Y1=594→Z1=3160/3 both
non-integer, Y1=891 and Y1=1782 both integer) — all match exactly. The
validator (`qa_pyth2_cattle_problem_cert_validate.py`) genuinely
derives alpha/beta/gamma algebraically from the 3-equation
underdetermined system rather than hardcoding them, then asserts
lcm(denominators)=891 — no fixture-trusting gap.
