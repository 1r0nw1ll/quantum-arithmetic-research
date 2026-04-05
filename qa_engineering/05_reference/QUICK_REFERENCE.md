# QA Quick Reference Card

Print this. Keep it next to your keyboard during QA sessions.

---

## State Space

```
(b, e)           ← primitive (the two degrees of freedom)
d = b + e        ← derived (NEVER independent)
a = b + 2e       ← derived (NEVER independent)

Domain: Caps(N,N) = {1 ≤ b,e ≤ N}
Applied work: N = 24
Theoretical:  N = 9
Zero excluded: domain is {1, ..., N}, not {0, ..., N-1}
```

---

## Generators

```
σ(b,e) = (b, e+1)           fails: OUT_OF_BOUNDS if e=N
μ(b,e) = (e, b)              never fails
λ_k(b,e) = (kb, ke)          fails: OUT_OF_BOUNDS or ZERO_DENOMINATOR
ν(b,e) = (b/2, e/2)          fails: PARITY if b or e is odd
```

---

## Failure Types

```
OUT_OF_BOUNDS    σ, λ    result coordinate > N
PARITY           ν       coordinate not even
PHASE_VIOLATION  any     phase function inconsistent
INVARIANT        any     invariant packet self-consistency broken
REDUCTION        ν       division not integer
```

---

## Orbit Families

```
f(b,e) = b² + be - e²   (Q(√5) norm)

Singularity: (b,e) ≡ (0,0) mod m  [1 state]
Satellite:   v₃(f) ≥ 2             [8 states]
Cosmos:      v₃(f) = 0             [72 states mod-9 / 504 mod-24]

Cosmos sub-families (mod-9, by norm f mod 9):
  Fibonacci:   f ≡ {1, 8}
  Lucas:       f ≡ {4, 5}
  Phibonacci:  f ≡ {2, 7}
  Tribonacci:  f ≡ 0 (but ≠ singularity, 8 states, not cosmos)
```

---

## Invariant Packet (Key Values)

```
C = 2·e·d        F = b·a
L = (C·F)/12     ← exact rational, NEVER approximate
I = |C - F|      ← always > 0 (theorem, not assumption)
H = C + F
```

---

## Orbit Edge Counts (N arbitrary)

```
σ edges: N(N-1)         σ failures: N
μ edges: N²             μ failures: 0
λ₂ edges: ⌊N/2⌋²       λ₂ failures: N² - ⌊N/2⌋²
ν edges: ⌊N/2⌋²         ν failures: N² - ⌊N/2⌋²

#SCC = (N²+N)/2   when generator set ⊇ {μ}
```

---

## The Canonical Control Path

```
singularity → satellite → cosmos
path_length_k = 2
(proved domain-generic: same for cymatics and seismology)
```

---

## The Obstruction Rule

```
Target r with v_p(r) = 1 for inert prime p → UNREACHABLE
p=3 is inert in Z[φ] (mod-9 and mod-24)
p=7 is inert in Z[φ] (mod-24 only)

Check BEFORE searching. nodes_expanded = 0 if obstructed.
```

---

## Gate Sequence (Required for All Certs)

```
[0] Mapping protocol intake
[1] Schema validation
[2] Generator uniqueness + invariant resolution
[3] Failure algebra completeness
[4] Invariant diff check
[5] Canonical hash / Merkle integrity

Never truncate to [0,1,2,3]. Gates 4+5 are tamper-evidence.
```

---

## Hashing Rules

```
sha256(domain.encode() + b'\x00' + payload)   ← domain-separated
json.dumps(obj, sort_keys=True,
           separators=(',',':'), ensure_ascii=False)  ← canonical JSON
HEX64_ZERO = '0' * 64                          ← manifest placeholder
```

---

## Cert Family Cheat Sheet

```
[107] QA_CORE_SPEC.v1              ← kernel (everything else inherits from this)
[106] QA_PLAN_CONTROL_COMPILER     ← plan→control compilation relation
[105] QA_CYMATIC_CONTROL           ← cymatics (4-tier: mode/faraday/control/planner)
[110] QA_SEISMIC_CONTROL           ← seismology (domain instance of [106])
[117] QA_CONTROL_STACK             ← domain-genericity proof (cymatics + seismology)
[118] QA_CONTROL_STACK_REPORT      ← reader-ready report for [117]
[119] QA_DUAL_SPINE_UNIFICATION    ← top-level map (obstruction + control spines)
[120] QA_PUBLIC_OVERVIEW_DOC       ← presentation-grade (hand to reviewer first)
[111→116] Obstruction spine        ← v_p(r)=1 → unreachable → pruning_ratio=1.0
[108] QA_AREA_QUANTIZATION         ← forbidden quadreas {3,6} in mod-9
[109] QA_INHERITANCE_COMPAT        ← certifies inheritance edges as first-class objects
```

---

## Meta-Validator Quick Check

```bash
# From repo root:
cd qa_alphageometry_ptolemy
python qa_meta_validator.py
# Expected: 126/126 PASS

# Core axiom self-test:
python qa_core_spec/qa_core_spec_validate.py --self-test

# Cymatics full stack:
python qa_cymatics/qa_cymatics_validate.py --self-test

# Compiler:
python qa_plan_control_compiler/qa_plan_control_compiler_validate.py --demo
```

---

## Finite-Orbit Descent (Neural Network Convergence)

```
L_{t+1} = (1 - κ_t)² · L_t          [exact identity]
ρ(O) = ∏(1-κ_t)²                     [orbit contraction factor]
L_{t+L} = ρ(O) · L_t                 [orbit-level convergence]

ρ(O) < 1 iff κ_min > 0
Empirical: ρ ≈ 0.001582 (mod-9, lr=0.5, gain=1) = 632× per orbit
r(mean_κ, final_loss) = -0.843 (Exp 1) / -0.845 (Exp 3, gain-robust)

Normalize lr: target_eta / H_QA
```

---

## Q(√5) / Fibonacci Connection

```
φ = (1+√5)/2    (golden ratio)
T = F²          (QA Fibonacci matrix squared = ×φ² in Z[φ])
f(b,e) = N(b+eφ) in Z[φ]      (algebraic norm)
N(φ²) = 1       (f is T-invariant)

3 inert in Z[φ] → Z[φ]/3Z[φ] ≅ GF(9)
Five families = φ-orbits in GF(9)²
```
