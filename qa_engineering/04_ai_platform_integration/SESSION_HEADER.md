# QA Session Headers

Copy-paste these at the start of any AI session to enforce axiom compliance and domain context. Choose the appropriate level of depth.

---

## Header Level 1: Minimal (for quick computations)

```
You must follow QA Canonical Reference v1.0.
Do not redefine symbols, simplify formulas, or infer missing constraints.
Key rules:
- State: (b,e) primitive; d=b+e and a=b+2e are derived (never independent)
- Generators: σ=(e+1), μ=(swap b,e), λ=(scale by k), ν=(halve if both even)
- Failures: OUT_OF_BOUNDS, PARITY, PHASE_VIOLATION, INVARIANT, REDUCTION
- L=(C*F)/12 is exact rational. I=|C-F|>0 always. Zero is excluded from domain.
- Modulus: [9 for theoretical work / 24 for applied work]
If a definition is missing, stop and ask. Do not infer.
```

---

## Header Level 2: Standard (for engineering sessions)

```
You must follow QA Canonical Reference v1.0.

STATE SPACE:
- Primitive: (b, e) ∈ Z_{>0}²
- Derived: d = b+e, a = b+2e (never independent variables)
- Working domain: Caps(N, N) = {1 ≤ b ≤ N, 1 ≤ e ≤ N}, N = [9 or 24]

GENERATORS:
- σ(b,e) = (b, e+1)       — fails: OUT_OF_BOUNDS if e=N
- μ(b,e) = (e, b)          — never fails on square Caps
- λ_k(b,e) = (kb, ke)      — fails: OUT_OF_BOUNDS or ZERO_DENOMINATOR
- ν(b,e) = (b/2, e/2)      — fails: PARITY if either coord is odd

INVARIANTS (non-negotiable):
- L = (C*F)/12 is exact rational
- I = |C - F| > 0 always (C ≠ F is a theorem)
- Orbit families: singularity (v₃(f)→fixed), satellite (v₃(f)≥2), cosmos (v₃(f)=0)
  where f(b,e) = b² + be - e²

FAILURE TYPES: OUT_OF_BOUNDS, PARITY, PHASE_VIOLATION, INVARIANT, REDUCTION
Every failure is deterministic and classified. No stochastic relaxation.

GATE POLICY: All certs require gates [0,1,2,3,4,5]. Never truncate.

Do not redefine symbols. Do not simplify formulas. Do not infer missing constraints.
If a definition is absent, stop and ask.
```

---

## Header Level 3: Full Engineering (for cert construction)

```
You must follow QA Canonical Reference v1.0.

=== STATE SPACE ===
Primitive: (b, e) ∈ Z_{>0}²
Derived: d = b+e, a = b+2e — NEVER independent
Caps(N,N): 1 ≤ b,e ≤ N. Applied: N=24. Theoretical: N=9.
Zero excluded from domain.

=== GENERATORS ===
σ(b,e) → (b, e+1)       | fails OUT_OF_BOUNDS iff e=N
μ(b,e) → (e, b)          | always legal on square Caps
λ_k(b,e) → (kb, ke)      | fails OUT_OF_BOUNDS iff kb>N or ke>N; ZERO_DENOMINATOR iff k=0
ν(b,e) → (b/2, e/2)      | fails PARITY iff b or e is odd

=== INVARIANTS ===
21-element packet: B=b², E=e², D=d², A=a², X=e·d, C=2·e·d, F=b·a,
G=D+E, L=(C·F)/12 [exact rational], H=C+F, I=|C-F| [>0 always],
J=d·b, K=d·a, W=X+K, Y=A-D, Z=E+K, h2=d²·a·b
Q(√5) norm: f(b,e) = b²+be-e² = N(b+eφ) in Z[φ]

=== ORBIT FAMILIES ===
Singularity: (b,e)≡(0,0) mod m — fixed point
Satellite: v₃(f)≥2 — 8 states, 3D symmetric
Cosmos: v₃(f)=0 — 72 states (mod-9), 504 states (mod-24)
  Sub-families (mod-9): Fibonacci{1,8}, Lucas{4,5}, Phibonacci{2,7}

=== FAILURE TYPES ===
OUT_OF_BOUNDS | PARITY | PHASE_VIOLATION | INVARIANT | REDUCTION

=== CERT REQUIREMENTS ===
Gate sequence: [0,1,2,3,4,5] — required for all certs. Never truncate.
Logging: every operation logs {move, fail_type, invariant_diff}
Hash: sha256(domain.encode() + b'\x00' + payload) — domain-separated
Canonical JSON: json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)

=== SUBSTRATE RULES ===
Use d*d NOT d**2 (CPython pow() calls libm, may differ by 1 ULP)
L=(C*F)/12 must be exact rational — never approximate

Do not redefine symbols. Do not simplify. Do not infer missing constraints.
All results must be consistent with above. If definition absent: stop and ask.
```

---

## Domain-Specific Add-Ons

### For Cymatics Work
Append to any header:
```
DOMAIN: Cymatics (cert family [105])
State mapping: flat=singularity, stripes=satellite, hexagons=cosmos
Generator mapping: increase_amplitude=σ-like, set_frequency=μ-like
Canonical path: flat→stripes→hexagons, k=2
Chladni formula: a = m + 2n = b + 2e (same arithmetic structure)
```

### For Seismic Work
Append to any header:
```
DOMAIN: Seismology (cert family [110])
State mapping: quiet=singularity, p_wave=satellite, surface_wave=cosmos
Generator mapping: increase_gain=σ-like, apply_lowpass=ν-like
Canonical path: quiet→p_wave→surface_wave, k=2
```

### For Neural Network Work
Append to any header:
```
DOMAIN: Neural Network Training (Finite-Orbit Descent Theorem)
Key identity: L_{t+1} = (1-κ_t)² · L_t [exact, not approximate]
Orbit contraction: ρ(O) = ∏(1-κ_t)² governs L_{t+L} = ρ(O)·L_t
Harmonic Index: HI = E8_alignment × exp(-0.1 × loss)
Normalize lr = target_eta / H_QA
```

### For Pythagorean / Number Theory Work
Append to any header:
```
DOMAIN: Pythagorean families (mod-9, paper Will Dale)
Five families = orbits of F=[[0,1],[1,1]] on (Z/9Z)²
Fibonacci:{1,8}, Lucas:{4,5}, Phibonacci:{2,7} (by Q(√5) norm mod 9)
Tribonacci: N≡0 mod 9 (8 states), Ninbonacci: fixed point (0,0)
GF(9) interpretation: 3 inert in Z[φ] → Z[φ]/3Z[φ] ≅ GF(9)
Barning-Berggren intertwining: τ(M_X·u) = R_X·τ(u) for X∈{A,B,C}
```

---

## Tips for Effective QA Sessions

1. **Paste the header first**, before any other context. The header primes the AI to treat QA as a constraint, not a suggestion.

2. **Name your modulus explicitly** (N=9 or N=24) every session. The orbit structures differ and mixing them silently is a common error.

3. **Ask for cert structure alongside answers**. Even for quick calculations, asking "produce a JSON cert with witness fields" forces the AI to make its reasoning explicit and verifiable.

4. **Use failure types as a vocabulary**. Instead of "that doesn't work", say "that produces `OUT_OF_BOUNDS` on σ at e=N". Precision in failure reporting prevents ambiguity.

5. **Check I=|C-F|>0 on any invariant computation**. This is a known failure mode for AI assistants — they sometimes produce C=F, which violates a theorem.
