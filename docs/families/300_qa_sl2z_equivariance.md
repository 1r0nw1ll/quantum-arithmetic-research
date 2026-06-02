# [300] QA SL(2,Z) Equivariance

**Family**: `qa_sl2z_equivariance_cert_v1`  
**Depends on**: [294] SL(2,Z) Spine, [298] Orbit Grade Decomposition, [299] Cayley-Hamilton Fibonacci-Lucas

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | L-equivariance: gcd(b, b+e)=gcd(b,e) for all b,e≥1 — algebraic identity + exhaustive on {1,...,9}² | PASS |
| C2 | R-equivariance: gcd(b+e, e)=gcd(b,e) for all b,e≥1 — algebraic identity + exhaustive on {1,...,9}² | PASS |
| C3 | Full SL(2,Z) equivariance: all W∈{L,R}* of length 1..8 preserve gcd(W·[b,e]ᵀ)=gcd(b,e) for all 81 states | PASS |
| C4 | Versor sandwich explicit: L·M·L⁻¹=[[−1,1],[−1,2]] (≡[[8,1],[8,2]] mod 9); R·M·R⁻¹=[[1,1],[1,0]]; both have trace=1, det=−1, char poly x²−x−1, and satisfy N²=N+I | PASS |
| C5 | Sandwich orbit partition: L·M·L⁻¹ and R·M·R⁻¹ each produce orbit partition 1+8+72 on {1,...,9}² | PASS |

## Key result

**Cert [298]** showed the T-operator M preserves v₃(gcd). **Cert [300]** completes the picture: *all* of SL(2,Z) preserves v₃(gcd), because both generators L and R preserve gcd (C1, C2), and by [294] they generate the full group (C3).

The **versor sandwich** W·M·W⁻¹ for W∈{L,R}: the conjugated operator inherits the same characteristic polynomial x²−x−1 (from [299]), same trace and det, and therefore the same orbit partition 1+8+72. The three orbit strata are not just T-invariant — they are **SL(2,Z)-stable** under the left-action and **conjugation-stable** under the sandwich action.

Explicit sandwiches:
- `L·M·L⁻¹ = [[−1,1],[−1,2]]` — "reflected" T-operator
- `R·M·R⁻¹ = [[1,1],[1,0]]` — "reversed Fibonacci" operator (note: [[1,1],[1,0]] is M with rows/cols swapped)

Both satisfy N²=N+I (Cayley-Hamilton) and produce the identical 1+8+72 orbit partition.

## Versor arc summary [294]→[300]

| Cert | Algebraic fact |
|------|----------------|
| [294] | L,R generate SL(2,Z); M²=L·R |
| [295] | (RLLR)^∞ periodic; Pell e-values = Sturmian CF denominators |
| [296] | T=M (det=−1); Stern-Brocot bijection; orbit 1+8+72 |
| [297] | Pell step A=[[1,2],[1,1]]; \|s−1/3\|=1/(3G̃); eigenspread=1/3 |
| [298] | v₃(gcd) T-invariant; M^12≡−I (grade inversion) |
| [299] | M²=M+I; M^k=Fibonacci matrix; Tr(M^k)=L(k) |
| [300] | SL(2,Z) preserves gcd; sandwich W·M·W⁻¹ preserves orbit 1+8+72 |

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.V (gcd), Ch.X (SL(2,Z))
- Hestenes, D. and Sobczyk, G. (1984) *Clifford Algebra to Geometric Calculus*, Reidel, ISBN 978-90-277-1673-6 (versor sandwich, equivariance)
