# [296] QA SL(2,Z) Versor Isomorphism

**Family**: `qa_sl2z_versor_isomorphism_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [294] SL(2,Z) Spine

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Stern-Brocot bijection: every primitive (b,e) maps to unique W∈{L,R}* with W·[1,1]ᵀ=[b,e]ᵀ; all W∈SL(2,Z) | PASS |
| C2 | QA T-step (b,e)→(e,b+e) equals M·[b,e]ᵀ; M=[[0,1],[1,1]], det(M)=−1 (odd-grade operator) | PASS |
| C3 | M²=L·R as integer matrices; two T-steps = one SL(2,Z) rotor | PASS |
| C4 | (9,9) is the unique fixed point of T in {1,...,9}² under QA A1 arithmetic | PASS |
| C5 | Orbit partition under T on {1,...,9}² is 1+8+72 (Singularity/Satellite/Cosmos) | PASS |

## Key result

The QA T-operator is the Fibonacci matrix M=[[0,1],[1,1]] with det=−1 — an **odd-grade versor** (reflection) in GL(2,Z). Two T-steps equal L·R ∈ SL(2,Z), an **even-grade versor** (rotor). The Stern-Brocot tree gives a bijection between primitive QA states and the positive cone of SL(2,Z), with every word having det=+1. The three orbit classes (Singularity/Satellite/Cosmos) correspond to fixed-point, short-period, and generic versor orbits of M in GL(2,Z/9Z).

**Note on grade**: The grade (even/odd) is a property of the **operator** M^k (det(M^k)=(−1)^k), not the resulting **states** — SB words for states always have det=+1 regardless of how many T-steps were taken to reach them.

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.X
- Hestenes, D. and Sobczyk, G. (1984) *Clifford Algebra to Geometric Calculus*, Reidel, ISBN 978-90-277-1673-6
- Brocot, A. (1861) Calcul des rouages par approximation, *Revue Chronometrique* 3:186-194
