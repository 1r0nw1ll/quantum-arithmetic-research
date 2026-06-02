# [297] QA Pell Rotor Eigenspread

**Family**: `qa_pell_rotor_eigenspread_cert_v1`  
**Depends on**: [292] Koenig Spread Optimality, [296] SL(2,Z) Versor Isomorphism

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | A=[[1,2],[1,1]], det(A)=−1 (odd-grade versor); maps (b,e)→(b+2e,b+e) exactly | PASS |
| C2 | Pell chain from (1,1): bₙ²−2eₙ²=(−1)^(n+1); \|I\|=1 always; sign alternates starting I₀=−1 | PASS |
| C3 | Spread sₙ=eₙ²/(bₙ²+eₙ²) satisfies \|sₙ−1/3\|=1/(3G̃ₙ); reproduces [292] formula I=3G̃\|s−1/3\|=1 exactly; spreads alternate above/below 1/3 | PASS |
| C4 | No positive integer solution to b²=2e² (√2 irrational); verified exhaustively for b,e≤10⁴ | PASS |
| C5 | A²=[[3,4],[2,3]] (the RLLR matrix from [296]); Möbius fixed point solves z²=2 (√2, irrational) | PASS |

## Key result

The Pell step matrix A=[[1,2],[1,1]] has det=−1 — it is an **odd-grade versor** (reflection) in GL(2,Z), the same grade as the T-operator M from [296]. The Möbius map f(z)=(z+2)/(z+1) has its unique positive fixed point at √2 (irrational). The spread of Pell directions converges to **exactly 1/3** from alternating sides, with `|sₙ−1/3| = 1/(3G̃ₙ)` — this is precisely the cert [292] formula I=3G̃|s−1/3| evaluated at I=1. The double step A² recovers the SB-word RLLR matrix from [296].

The eigenspread s=1/3 is a **limit**, not a value attained by any rational direction. The irrationality of √2 (C4) guarantees no rational fraction achieves s=1/3 exactly — the Pell chain approaches it asymptotically with rate 1/G̃ₙ → 0.

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.IV (irrationality of √2), Ch.XIII (Pell equation)
- Wildberger, N.J. (2005) *Divine Proportions*, Wild Egg Books, ISBN 978-0-9757492-0-8 (rational spread s=e²/(b²+e²))
