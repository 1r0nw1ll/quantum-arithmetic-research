# [298] QA Orbit Grade Decomposition

**Family**: `qa_orbit_grade_decomposition_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [296] SL(2,Z) Versor Isomorphism, [297] Pell Rotor Eigenspread

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | v₃(gcd(b,e)) is invariant under T; algebraic proof: gcd(e, b+e)=gcd(e,b)=gcd(b,e); exhaustive on all 81 states | PASS |
| C2 | Arithmetic stratification: v₃=2 → Singularity (1 state, 9\|b and 9\|e); v₃=1 → Satellite (8 states, 3\|gcd not 9\|both); v₃=0 → Cosmos (72 states) | PASS |
| C3 | Period by stratum: Singularity period 1; Satellite period 8; Cosmos period 24 | PASS |
| C4 | Even/odd T-step split: Cosmos orbits 12+12=24; Satellite orbit 4+4=8 | PASS |
| C5 | M^12 ≡ −I (mod 9) exactly: diagonal entries 8≡−1, off-diagonal 0; M^24 ≡ I; half-period is the grade-inversion antipodal map | PASS |

## Key result

The three QA orbits are not an arbitrary partition — they are the **v₃ stratification** of the state space {1,...,9}² by the 3-adic valuation of gcd(b,e). The T-operator preserves this stratification because `gcd(e, b+e) = gcd(b,e)` algebraically — not just empirically.

The **grade** in the versor sense is v₃(gcd):
- v₃=0 (Cosmos): primitive directions — the "bivector/spinor" layer, generic versor orbit
- v₃=1 (Satellite): one factor of 3 — intermediate symmetry, 8-cycle
- v₃=2 (Singularity): fixed by everything — the "scalar", universally stabilized

The **grade inversion** M^12≡−I (mod 9) is the half-period map. Applying T twelve times sends every Cosmos state to its "antipodal partner" — a different Cosmos state 12 steps away — and returns after 24 steps. This is the QA analog of a 180° rotation in GA rotor algebra.

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.IV (p-adic valuation, gcd)
- Wall, D.D. (1960) Fibonacci Primitive Roots, *Amer. Math. Monthly* 67(6):525–532, DOI 10.1080/00029890.1960.11989541 (Pisano period π(9)=24, orbit periods)
