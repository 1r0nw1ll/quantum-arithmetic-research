# [345] QA BABTHE Dual Bead Chain Identity

**Family**: `qa_babthe_dual_bead_chain_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XIV pp.74-85

> "The first set is assigned the identities N, O, P, Q and the second set is  
>  assigned the identities Q, R, S, T."  
> "The third number, S, of the higher bead numbers is also the product of the  
>  two intermediate numbers of the lower bead numbers."  
> "In every case in the BABTHE program, O+P was equal to S-R."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | (N, O, P=N+O, Q=O+P) is Fibonacci-type: N+O=P, O+P=Q | PASS |
| C2 | (Q, R=S-Q, S=O·P, T=R+S) is Fibonacci-type: Q+R=S, R+S=T | PASS |
| C3 | Q is the shared junction: Q=O+P (4th of first) = S-R (1st of second) | PASS |
| C4 | S=O·P; equivalently R=(O-1)(P-1)-1 | PASS |
| C5 | 2/T = 1/S + 1/(OT) + 1/(PT) — Babylonian unit fraction decomposition | PASS |

## Structure

For coprime integers N (odd, ≥1) and O (≥2), let:

$$P = N + O, \quad Q = O + P = N + 2O, \quad S = O \cdot P, \quad R = S - Q, \quad T = R + S$$

**First bead quadruple**: $(N, O, P, Q)$ — Fibonacci-type since $N+O=P$ and $O+P=Q$

**Second bead quadruple**: $(Q, R, S, T)$ — Fibonacci-type since $Q+R=S$ and $R+S=T$

**Junction**: $Q$ is simultaneously the 4th element of the first set and the 1st element of the second set.

**Product**: $S = O \cdot P$ (the product of the inner pair of the first quadruple becomes the sum-element of the second quadruple).

**Canonical example** (Iverson's 2/97):
- $N=1, O=7$: First set $(1, 7, 8, 15)$, second set $(15, 41, 56, 97)$
- $2/97 = 1/56 + 1/679 + 1/776$ ✓

**First example** (Iverson's 2/7):
- $N=1, O=2$: First set $(1, 2, 3, 5)$, second set $(5, 1, 6, 7)$
- $2/7 = 1/6 + 1/14 + 1/21$ ✓

## Algebraic Proof of C5

$$\frac{1}{S} + \frac{1}{OT} + \frac{1}{PT} = \frac{1}{OP} + \frac{O+P}{OPT} = \frac{T + Q}{ST} = \frac{R+S+Q}{ST}$$

Since $R = S - Q$ we have $R + Q = S$, so $R+S+Q = 2S$, giving:

$$\frac{2S}{ST} = \frac{2}{T}$$

The unit fraction identity is a direct algebraic consequence of $R = S - Q$.

## Observer Projection Note (Theorem NT)

"Babylonian fractions," "unit fractions," "ancient mathematics" are historical observer labels. The causal arithmetic: $S=O\cdot P$ (integer product), $R=S-Q$ (integer difference), $T=R+S$ (integer sum), fraction equality via Fraction arithmetic. No continuous time or geometry enters the causal layer.

**Depends on**: [343] Fibonacci Bead Number Quadruple; [316] QA Double Quantum Number for Diadic Fractions; [317] QA Diadic Inner Structure
