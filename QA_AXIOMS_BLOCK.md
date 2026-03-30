# QA Axioms Block (Canonical v1.0)

Source of truth: `Formalizing tuple drift in quantum-native learning/files/files(1)/qa_canonical.md`

Session header (paste into new sessions):
```
You must follow qa_canonical.md (QA Canonical Reference v1.0).
Do not redefine symbols, simplify formulas, or infer missing constraints.
If a needed definition is absent from qa_canonical.md, stop and ask.
All results must be consistent with the canonical checksums in section 12.
```

## Axioms

State space:
- Primitive coordinates: (b, e) in Z_{>0}^2
- Derived coordinates: d = b + e, a = b + 2e
- Derived coordinates are never independent.

Invariant packet (21 elements, derived from b, e):
- Squares: B = b^2, E = e^2, D = d^2, A = a^2
- Products: X = e*d, C = 2*e*d, F = b*a
- Combined: G = D + E, L = (C*F)/12 (exact rational), H = C + F, I = |C - F|
- More: J = d*b, K = d*a, W = X + K, Y = A - D, Z = E + K, h2 = d^2*a*b

Phase functions:
- phi_9(a) = digital_root(a) = ((a - 1) mod 9) + 1 for a > 0
- phi_24(a) = a mod 24

Generator algebra (partial functions on Caps(N, N)):
- sigma(b, e) = (b, e + 1) if e + 1 <= N
- mu(b, e) = (e, b)
- lambda2(b, e) = (2b, 2e) if 2b <= N and 2e <= N
- nu(b, e) = (b/2, e/2) if b, e both even

Caps lattice:
- Caps(N, N) = {(b, e) in Z_{>0}^2 | 1 <= b <= N, 1 <= e <= N}

Failure taxonomy:
- OUT_OF_BOUNDS, PARITY, PHASE_VIOLATION, INVARIANT, REDUCTION

Non-negotiables:
- d, a are derived; (b, e, d, a) are not independent.
- L is exact rational; I = |C - F| is positive; C != F is a theorem.
- Failures are deterministic; no stochastic or continuous relaxation.
