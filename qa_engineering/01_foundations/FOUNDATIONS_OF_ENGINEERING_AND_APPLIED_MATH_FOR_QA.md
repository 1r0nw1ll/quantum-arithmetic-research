# Foundations of Engineering and Applied Mathematics for QA

**Who this is for**: Tier 4 patrons with a classical engineering or applied mathematics
background who want to understand how those foundations connect to QA.

**What this does**: maps the five core concepts of engineering and applied math — state, dynamics,
control, invariants, computation — into QA language, and points you to the exact files and certs
in this on-boarding pack where the formal version lives.

This is the entry-point synthesis doc. It assumes you have already read `QA_PRIMER.md`.

---

## The mental model (read this first)

Engineering asks:
> "What state am I in, and how do I move to another?"

QA answers:
- **State** = `(b, e)` — a pair of integers in `{1,...,N}`
- **Moves** = generators (σ, μ, λ, ν) — lawful arithmetic steps
- **Constraints** = arithmetic invariants — what is structurally preserved
- **Control** = find a valid generator path to the target
- **Failure** = provable obstruction — deterministic, typed, not silent

Everything else in this document is a formal version of those five lines.

---

## 1. State

### Classical meaning

In classical engineering a **state** is everything you need to know about a system right now to
predict its future behaviour. For a spring-mass system it might be position and velocity. For a
circuit it might be charge and current. For a Markov chain it is the current node.

State is typically represented as a vector in ℝⁿ. It lives in a **state space** — the set of all
possible configurations.

### QA translation

In QA, a state is a pair **(b, e)** drawn from the domain `{1, ..., N}²` — the set `Caps(N, N)`.
No zero, no negative values. The pair `(b, e)` completely determines the full 4-tuple
`(b, e, d, a)` where `d = b + e` and `a = b + 2e`. Those derived values are consequences of the
state, not independent degrees of freedom.

The state space has three structurally distinct regions (orbit families):

| Orbit family | Condition | Physical analogy |
|---|---|---|
| Singularity | `(b, e) ≡ (0, 0) mod m` | equilibrium / fixed point |
| Satellite | `v₃(f(b,e)) ≥ 2` | near-resonance / transient regime |
| Cosmos | `v₃(f(b,e)) = 0` | stable limit cycle |

where `f(b,e) = b·b + b·e - e·e` is the Q(√5) norm.

**One important constraint**: zero is not a valid state. A common mistake when translating
classical 0-indexed state-space models is to encode states as `(0, e)` or `(b, 0)`. QA uses
`{1,...,N}`, not `{0,...,N-1}`. The validator enforces this (EC1 in cert [121]).

### Where to look

- `QA_AXIOMS.md` — axioms A1–A3 define the state space and domain formally
- `QA_STATE_SPACE.md` — full orbit table, mod-24 structure, SCC layout
- `qa_engineering_core_cert/fixtures/engineering_core_fail_invalid_encoding.json` — live example
  of a state encoding error caught by the validator

---

## 2. Dynamics

### Classical meaning

**Dynamics** describes how a system evolves. In classical engineering this is usually expressed as
a differential equation (continuous time) or a difference equation (discrete time):

```
ẋ = f(x, u)       (continuous)
x_{t+1} = f(x_t, u_t)   (discrete)
```

The choice of `f` encodes the physics: springs, capacitors, fluids, heat, and so on. Dynamics
determine the trajectories — the paths the system can actually follow.

### QA translation

In QA, dynamics are **generator moves** — lawful discrete steps that transform one state into
another. The four primitive generators are:

| Generator | Action | Failure condition |
|---|---|---|
| σ | e → e + 1 | `OUT_OF_BOUNDS` if e + 1 > N |
| μ | swap b ↔ e | always defined |
| λ | scale (b, e) by k | `OUT_OF_BOUNDS` if result exceeds N |
| ν | halve (b, e) if both even | `PARITY` if not both even |

Every trajectory in QA is a finite sequence of generator moves. There are no differential
equations — the dynamics are fully discrete and arithmetic. This is not an approximation of
continuous dynamics; it is a different kind of system.

Domain-specific generators (like `excite` and `tune` in the spring-mass example, or `flat→stripes`
in cymatics) are **named generator sequences** in the domain's vocabulary, which then compile down
to the primitive generators.

### Where to look

- `QA_AXIOMS.md` — axioms A4–A5 define the generator algebra and failure taxonomy
- `02_control_theory/STEERING_GUIDE.md` — how to plan a trajectory (steps 1–5)
- `qa_engineering_core_cert/fixtures/engineering_core_pass_spring_mass.json` — a complete
  two-step trajectory with named generators
- Cert [107] `qa_core_spec/` — the formal generator alphabet definition

---

## 3. Control

### Classical meaning

**Control theory** asks: given a system with dynamics, can you drive it from an initial state to
a desired target state, and how? The foundational results are:

- **Controllability** (Kalman, 1960): a linear system is controllable if the controllability
  matrix has full rank. If it does, any state is reachable from any other in finite time.
- **Lyapunov stability**: a system is stable if there exists a Lyapunov function V(x) that
  decreases along trajectories.
- **Optimal control**: find the input sequence that reaches the target with minimum cost.

### QA translation

QA has direct counterparts to all three:

**Reachability**: whether a target state is reachable is determined first by arithmetic, then by
BFS. The key result is that **classical full-rank controllability is necessary but not sufficient
for QA reachability**:

> **QA reachability = (classical controllability) ∧ (arithmetic admissibility)**

Specifically:

> If `v_p(b·e) = 1` for any inert prime `p` of the modulus, the target is arithmetically
> unreachable — regardless of what the Kalman rank condition says.

This is EC11 in cert [121]. It is not a refinement of classical control theory; it is an
arithmetic constraint that classical linear algebra cannot see. QA detects arithmetic obstructions
that classical controllability tests do not.

**Stability**: the Lyapunov function maps directly to the Q(√5) norm `f(b,e)`. The orbit
contraction factor `ρ = ∏(1 - κ_t)²` plays the role of the Lyapunov decrease rate. Proved in the
**Finite-Orbit Descent Theorem**: `L_{t+L} = ρ(O) · L_t`, with `ρ < 1` iff `κ_min > 0`.

**Optimal control**: the minimality witness in a QA plan certifies that no shorter generator
sequence exists (by BFS frontier exhaustion). This is the discrete counterpart of time-optimal
control.

### Where to look

- `02_control_theory/CONTROL_THEOREMS.md` — six proved theorems including Finite-Orbit Descent
- `02_control_theory/PLAN_CONTROL_COMPILER.md` — the four-tier cert stack
- `06_classical_engineering_map/CLASSICAL_TO_QA_MAP.md` — full classical ↔ QA equivalence table
- `06_classical_engineering_map/QA_ENGINEERING_CORE_CERT_SPEC.md` — the [121] design spec
- Cert [121] `qa_engineering_core_cert/` — formal validator with EC9 (reachability), EC11 (obstruction)
- Cert [106] `qa_plan_control_compiler/` — compiled control cert
- Cert [117] `qa_control_stack/` — cross-domain domain-genericity theorem

---

## 4. Invariants

### Classical meaning

An **invariant** is a quantity that is preserved under the system's dynamics. In classical
engineering:

- **Energy** is conserved in an isolated mechanical system (first law of thermodynamics)
- **Angular momentum** is conserved under rotational symmetry (Noether's theorem)
- **Lyapunov invariants** certify stability by proving V(x) does not increase
- **Structural invariants** like rank, signature, and index classify systems up to equivalence

Invariants are what make a system provable rather than just simulable.

### QA translation

The central QA invariant is the **Q(√5) norm**:

```
f(b, e) = b·b + b·e - e·e
```

This is the norm `N(b + eφ)` in the ring of integers of Q(√5), where φ = (1+√5)/2. It classifies
every state into an orbit family (singularity / satellite / cosmos) and is preserved modulo `m`
under the generator dynamics.

The key structural fact: `det([[b, d], [e, a]]) = f(b, e)`. The determinant of the state matrix
IS the orbit classifier. This is why orbit family membership is structurally stable under
generator moves.

The **21-element invariant packet** in `QA_AXIOMS.md` lists all quantities that the validator
checks are preserved across a QA session: orbit family, modulus, generator alphabet, gate
sequence, canonical hash chain, failure taxonomy completeness, and more.

Invariants are also what determine **forbidden states**: the quadrea spectrum
`{0,1,2,4,5,7,8}` for mod-9 excludes `{3,6}` because those values have `v₃ = 1` — they cannot
appear as `f(b,e)` for any valid state. This is the arithmetic obstruction at the invariant level.

### Where to look

- `QA_AXIOMS.md` — the 21-element invariant packet, A6–A7
- `02_control_theory/CONTROL_THEOREMS.md` — SCC Monotonicity theorem (orbit family is invariant
  under generator composition)
- Cert [108] `qa_area_quantization/` — mod-9 forbidden quadrea theorem
- Cert [111] `qa_area_quantization_pk/` — generalised: inert primes → forbidden spectrum for any
  mod p^k

---

## 5. Computation

### Classical meaning

**Computation** in engineering means: how do you actually solve it? Engineering models are often
too complex for exact closed-form solutions. The practical toolkit is:

- **Numerical integration** — simulate ODE/PDE evolution step by step
- **Finite element methods** — discretise continuous domains
- **Iterative solvers** — converge on solutions to large linear systems
- **Search algorithms** — BFS/DFS/A* for planning and scheduling
- **Verification tools** — model checkers, type systems, proof assistants

Computation turns mathematical models into executable artefacts.

### QA translation

QA computation is **deterministic, arithmetic, and hash-bound**. Every computation produces
either a verified result with a witness, or a structured failure with a typed fail code. Nothing
is silent.

The computational stack has three levels:

**Level 1 — State arithmetic**: the generator moves themselves. Direct arithmetic on `(b, e)` mod
`m`. No floating point, no approximation. Substrate rule: always `b*b` not `b**2` to avoid CPython
`libm` divergence.

**Level 2 — Search and planning**: BFS over the state graph to find generator sequences reaching
a target. Bounded by a depth parameter. Produces a path (reachability witness) or a proof of
unreachability (obstruction check).

**Level 3 — Certificate validation**: the validators re-derive every claimed quantity from first
principles and compare against the cert. Hash-chained to prevent silent tampering. Self-test mode
emits structured JSON — designed to be called from CI.

**Canonical computation contract**:
```python
canonical_json = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
hash = sha256(domain.encode() + b'\x00' + canonical_json.encode())
```

Every cert is reproducible from its JSON alone. No external state, no network calls, no random
seeds.

### Where to look

- `QA_AXIOMS.md` — canonical session header (Level 1 substrate contract)
- `05_reference/QUICK_REFERENCE.md` — hash formula, gate sequence, canonical path
- `04_ai_platform_integration/AI_INTEGRATION_GUIDE.md` — how to use QA computation as a
  verification layer inside an AI platform session
- `qa_engineering_core_cert/qa_engineering_core_cert_validate.py` — a complete Level 3 validator
  (read this to understand the computational model concretely)
- Cert [107] `qa_core_spec/` — gate policy `[0,1,2,3,4,5]`, logging contract, hash chain spec

---

## Summary table

| Concept | Classical engineering | QA |
|---|---|---|
| **State** | vector in ℝⁿ, 0-indexed | pair (b,e) in {1..N}², orbit-classified |
| **Dynamics** | differential equations / difference equations | generator moves (σ, μ, λ, ν) |
| **Control** | Kalman rank + Lyapunov | BFS reachability + arithmetic obstruction check (EC11) |
| **Invariants** | energy, momentum, structural rank | Q(√5) norm f(b,e), orbit family, forbidden quadreas |
| **Computation** | numerical solvers, simulation | deterministic arithmetic + hash-bound cert validation |

---

## Reading order from here

This doc is the bridge entry point. From here:

1. **Formal axioms and generators** → `QA_AXIOMS.md`
2. **Proved theorems** → `02_control_theory/CONTROL_THEOREMS.md`
3. **Applied domain examples** → `03_applied_domains/CYMATICS_EXAMPLE.md`,
   `SEISMIC_EXAMPLE.md`
4. **Full classical engineering map** → `06_classical_engineering_map/CLASSICAL_TO_QA_MAP.md`
5. **Formal cert spec** → `06_classical_engineering_map/QA_ENGINEERING_CORE_CERT_SPEC.md`
   and cert [121] in `qa_alphageometry_ptolemy/qa_engineering_core_cert/`
