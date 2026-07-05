# [347] QA Cattle Problem Bull Modular Structure

**Family**: `qa_cattle_problem_bull_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XVI pp.104-109

> "W1 is a mod 42(0) number and (Z1-Y1) is a mod 13(0) number."  
> "1/6 X1 + 11/20 Z1 + 29/42 W1 = 3 Y1"  
> "W1=2226, X1=1602, Y1=891, Z1=1580"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | (W1,X1,Y1,Z1)=(2226,1602,891,1580) satisfies all three bull equations | PASS |
| C2 | Modular divisibility: X1≡0(mod 6); Z1≡0(mod 20); W1≡0(mod 42) | PASS |
| C3 | Difference coprimality: (W1-Y1)≡0(mod 5); (X1-Y1)≡0(mod 9); (Z1-Y1)≡0(mod 13) | PASS |
| C4 | Fractional remainder identity: ⅙X1 + 11/20·Z1 + 29/42·W1 = 3Y1 = 2673 | PASS |
| C5 | Individual remainders: ⅙X1=267, 11/20·Z1=869, 29/42·W1=1537 | PASS |

## Structure

### The Three Bull Equations

Archimedes' Cattle Problem (attributed, ~250 BC) yields three simultaneous constraints on bull counts:

$$W_1 = Y_1 + \frac{5}{6} X_1 \quad (1)$$
$$X_1 = Y_1 + \frac{9}{20} Z_1 \quad (2)$$
$$Z_1 = Y_1 + \frac{13}{42} W_1 \quad (3)$$

**Wurm's solution** (late 19th century): $W_1=2226,\; X_1=1602,\; Y_1=891,\; Z_1=1580$

### Modular Structure (C2)

From equation (1): $X_1$ must be exactly divisible by 6 (for $\frac{5}{6}X_1$ to be integer), and $W_1-Y_1$ must be divisible by 5.

| Variable | Required modulus | Value | Quotient |
|----------|-----------------|-------|---------|
| $X_1$ | mod 6 | $1602 = 6 \times 267$ | 267 |
| $Z_1$ | mod 20 | $1580 = 20 \times 79$ | 79 |
| $W_1$ | mod 42 | $2226 = 42 \times 53$ | 53 |

### Difference Modular Structure (C3)

| Difference | Required modulus | Value | Quotient |
|-----------|-----------------|-------|---------|
| $W_1 - Y_1 = 1335$ | mod 5 | $5 \times 267$ | 267 |
| $X_1 - Y_1 = 711$ | mod 9 | $9 \times 79$ | 79 |
| $Z_1 - Y_1 = 689$ | mod 13 | $13 \times 53$ | 53 |

The quotient factors {267, 79, 53} appear in both the modular structure and the difference structure — a deep arithmetic coincidence.

### Fractional Remainder Identity (C4, C5)

Transposing each equation: $W_i - Y_i = \text{fraction} \times \text{other variable}$. The remaining portions not used:

$$\frac{1}{6}X_1 + \frac{11}{20}Z_1 + \frac{29}{42}W_1 = 3Y_1$$

**Algebraic proof**: Summing the three transposed equations $(W_1-Y_1)+(X_1-Y_1)+(Z_1-Y_1)=\frac{5}{6}X_1+\frac{9}{20}Z_1+\frac{13}{42}W_1$, rearranging gives $W_1(1-\frac{13}{42})+X_1(1-\frac{5}{6})+Z_1(1-\frac{9}{20})=3Y_1$, i.e., $\frac{29}{42}W_1+\frac{1}{6}X_1+\frac{11}{20}Z_1=3Y_1$.

Individual remainders: $267 + 869 + 1537 = 2673 = 3 \times 891 = 3Y_1$.

## Observer Projection Note (Theorem NT)

"White bulls," "dappled cows," "cattle" are observer labels on integer linear constraints. The causal structure is divisibility (X1 mod 6 = 0), integer linear equations (W1=Y1+5/6·X1), and Fraction arithmetic. No continuous geometry enters the QA causal layer.

**Depends on**: [344] QA Prime Residue Symmetry; [342] QA Pythagorean Divisibility Laws

## Verification Note (2026-07-04)

Independently solved the 3-equation bull-only system from scratch with exact
`Fraction` arithmetic (Cramer's rule over ℚ), without assuming the cert's
claimed answer. **Result matches exactly**: (W1,X1,Y1,Z1)=(2226,1602,891,1580),
confirmed as the true minimal positive-integer solution of this subsystem —
no bug. Cross-checked against the well-known full classical problem (Wikipedia,
"Archimedes's cattle problem"): the real minimal bulls-only-plus-cows solution
is exactly this vector times 4657 — i.e. (10366482, 7460514, 4149387, 7358060)
= 4657×(2226,1602,891,1580) exactly — confirming this cert's smaller numbers
are the genuine minimal solution of the bull-only fragment, consistent with
(and a clean divisor of) the full published answer.

**One unresolved historical-attribution question**: the cert (following its
cited primary source, Iverson 1993 p.107) attributes this specific bulls-only
minimal solution to "Wurm's solution (late 19th century)." Independent web
sources describe J.F. Wurm's well-documented 1830 contribution as addressing
a different, larger sub-problem (the full 7-equation bulls+cows system under
the relaxed "rectangular number" condition, not this simpler 3-equation
bulls-only fragment) — no external source found crediting Wurm specifically
with this smaller computation. Could not access Iverson's original text
locally to check the primary source directly (the QA book corpus lives on a
different machine per project memory). Not asserting the attribution is
wrong — Iverson's own book may have good reason for it — but flagging this
as unverified rather than confirmed. The mathematics itself is independently
confirmed correct regardless of the attribution's precision.
