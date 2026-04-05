# Worked Example: Spring-Mass-Damper in QA

This document walks a single classical engineering system through the complete QA engineering
ladder — from physical description to formal certificate. It is the same example used in cert
[121], so you can verify every step against a machine-checkable artefact.

**System**: spring-mass-damper `mx'' + cx' + kx = F(t)`
**Modulus**: 9
**Cert**: `qa_alphageometry_ptolemy/qa_engineering_core_cert/fixtures/engineering_core_pass_spring_mass.json`

---

## Step 1 — Classical description

A spring-mass-damper has three recognisably distinct operating regimes:

| Regime | Physical description |
|--------|----------------------|
| **Still** | No oscillation. Mass at rest. All energy dissipated or not yet injected. |
| **Transient** | Damped startup. Energy injected; oscillation building but not yet settled. |
| **Steady oscillation** | Limit cycle. Oscillation stable; input energy balances damping loss. |

Classical control engineering treats this as a two-dimensional continuous state space (position x,
velocity x'). The system is **stable** (Lyapunov: V = ½mx'² + ½kx² decreasing toward
equilibrium) and **controllable** (Kalman rank condition is satisfied for a suitable input matrix).

This is the system you already know. Now we map it into QA.

---

## Step 2 — QA state encoding

The three regimes map to three states in `Caps(9, 9) = {1,...,9}²`:

| Classical regime | QA label | (b, e) | f(b,e) mod 9 | Orbit family |
|-----------------|----------|--------|--------------|--------------|
| Still | `still` | (9, 9) | 0 | **singularity** |
| Transient | `transient` | (3, 6) | 0 | **satellite** |
| Steady oscillation | `steady_oscillation` | (1, 2) | 8 | **cosmos** |

**How to read the orbit family**:

`f(b, e) = b·b + b·e - e·e` is the Q(√5) norm. For each state:

```
f(9, 9) = 81 + 81 - 81 = 81 ≡ 0 mod 9   → (9,9) ≡ (0,0) mod 9 → singularity (fixed point)
f(3, 6) = 9  + 18 - 36 = -9 ≡ 0 mod 9   → v₃(f) ≥ 2           → satellite
f(1, 2) = 1  + 2  - 4  = -1 ≡ 8 mod 9   → v₃(f) = 0            → cosmos
```

The orbit family assignment is not arbitrary — it is derived deterministically from the (b, e)
encoding. If you pick different (b, e) values, you will get different orbit families. The
validator (EC5) recomputes this independently and rejects any cert where the claimed orbit family
disagrees with the arithmetic.

**What zero means**: the state `(9, 9)` looks like it uses 9, but `9 ≡ 0 mod 9` — so modularly
it *is* the zero element, which is what makes it the singularity (fixed point). The QA domain is
`{1,...,9}`, meaning the coordinate values are 1–9 inclusive; `9` is the valid representation of
the additive identity in this modulus. Do not encode states as `(0, e)` — zero as a raw
coordinate is outside the domain.

---

## Step 3 — Orbit classification

The three orbit families have a concrete meaning:

| Orbit family | Structural meaning | Spring-mass analogue |
|---|---|---|
| Singularity | Fixed point under the generator dynamics. Self-absorbing. | Still — all motion has ceased |
| Satellite | Near the fixed point; norm divisible by 9. Transient attractor. | Transient — settling toward steady state |
| Cosmos | Full 24-cycle orbit; norm not divisible by 3. Stable limit cycle. | Steady oscillation — sustained periodic motion |

This classification is stable: it does not depend on which (b, e) pair you choose within a
family, only on the algebraic structure of the family itself. All three cosmos states for mod-9
are equivalent under the generator dynamics.

---

## Step 4 — Generator path (the control sequence)

Two generators drive the path from `still` to `steady_oscillation`:

| Step | Generator | From | To | Physical meaning |
|------|-----------|------|----|-----------------|
| 1 | `excite` | still (singularity) | transient (satellite) | inject energy into the system |
| 2 | `tune` | transient (satellite) | steady_oscillation (cosmos) | adjust to resonance frequency |

The path length is **k = 2**. This is the QA counterpart of a two-step optimal control sequence.

The named generators `excite` and `tune` are domain vocabulary. In the formal cert they compile
down to primitive QA generator moves (σ, λ, μ, or ν). The important thing is that:

1. Both generators are named (EC2 — the validator rejects unnamed transitions)
2. The path is witnessed by BFS (EC9 — reachability witness required for full-rank claim)
3. No shorter path exists (EC10 — minimality witness proves no k=1 path exists)

**The minimality witness** for this example:
- depth 1 frontier: only `transient` is reachable from `still` in one step
- `steady_oscillation` is not in the depth-1 frontier
- Therefore k=2 is tight (no shortcut exists)

This is the discrete counterpart of time-optimal control.

---

## Step 5 — Stability interpretation

Classical Lyapunov stability: V = ½mx'² + ½kx² decreases along trajectories, with V = 0 at
equilibrium.

QA counterpart: the Q(√5) norm `f(b, e)` plays the role of the Lyapunov function. The
**orbit contraction factor** `ρ` plays the role of the Lyapunov decrease rate:

```
ρ = ∏(1 - κ_t)²   (product over one full orbit)
L_{t+L} = ρ · L_t
```

For mod-9 with standard parameters: `ρ = 0.001582 < 1`. This means every orbit reduces the
loss by a factor of ~632. Ten orbits gives ~10⁻²⁸ reduction. This is the **Finite-Orbit Descent
Theorem**.

The validator checks EC7: `orbit_contraction_factor < 1.0`. This is the formal requirement that
the system is contracting (stable in the Lyapunov sense) under QA dynamics.

The equilibrium state must map to the **singularity** orbit family (EC8): the fixed point of the
dynamics is `still`, which is `(9, 9)` — correctly classified as singularity. A system whose
equilibrium maps to cosmos or satellite would have a misidentified fixed point.

---

## Step 6 — Controllability and arithmetic obstruction check

**Classical controllability**: the Kalman rank condition is satisfied. The cert declares
`classical_controllability: "full_rank"`.

**QA arithmetic check (EC11)**: this is the step classical analysis skips.

```
target state: steady_oscillation = (b=1, e=2)
target_r = b · e = 1 · 2 = 2
inert prime for mod 9: p = 3
v₃(2) = 0   (3 does not divide 2)
v₃(2) ≠ 1   → NOT obstructed
```

Since `v₃(target_r) ≠ 1`, the target is **reachable** in the arithmetic sense. The cert
correctly declares `obstructed: false`.

**Why this matters**: for a different encoding, the arithmetic check can fail even when the
Kalman rank passes. See the failure example below.

---

## Step 6b — What failure looks like (the obstruction fixture)

The cert family includes a deliberate failure fixture:
`engineering_core_fail_arithmetic_obstruction.json`

In that fixture, `steady_oscillation` is encoded as `(b=1, e=3)` instead of `(b=1, e=2)`.

```
target_r = 1 · 3 = 3
v₃(3) = 1   (3 divides 3 exactly once)
v₃(3) = 1   → OBSTRUCTED
```

The Kalman rank check still passes (same transition graph, same generator names). But the
arithmetic is now telling you something the rank check cannot: **no generator sequence can reach
a state with `r = 3`** because `v₃(3) = 1` and 3 is inert in Z[φ]. The target is
arithmetically forbidden.

The validator catches this as `ARITHMETIC_OBSTRUCTION_IGNORED` — the cert declared
`obstructed: false` but arithmetic says `obstructed: true`.

QA detects arithmetic obstructions that classical controllability tests do not see.

---

## Step 7 — The cert link

Everything above is encoded in a machine-checkable JSON certificate. To verify it yourself:

```bash
cd /path/to/signal_experiments/qa_alphageometry_ptolemy

# Verify the PASS fixture
python qa_engineering_core_cert/qa_engineering_core_cert_validate.py \
  --cert qa_engineering_core_cert/fixtures/engineering_core_pass_spring_mass.json

# Verify both FAIL fixtures
python qa_engineering_core_cert/qa_engineering_core_cert_validate.py \
  --cert qa_engineering_core_cert/fixtures/engineering_core_fail_arithmetic_obstruction.json

python qa_engineering_core_cert/qa_engineering_core_cert_validate.py \
  --cert qa_engineering_core_cert/fixtures/engineering_core_fail_invalid_encoding.json

# Run all three as a self-test (JSON output)
python qa_engineering_core_cert/qa_engineering_core_cert_validate.py --self-test
```

Expected output for `--self-test`:
```json
{"ok": true, "passed": 3, "failed": 0, "total": 3, ...}
```

---

## Full ladder summary

| Layer | What you did | QA concept | Formal location |
|-------|-------------|------------|-----------------|
| 1 | Described the physical system | Classical model | `FOUNDATIONS_OF_ENGINEERING_AND_APPLIED_MATH_FOR_QA.md` |
| 2 | Encoded states as (b, e) pairs | State encoding EC1 | cert [121] fixture |
| 3 | Classified each state by f(b,e) | Orbit family EC5 | `QA_AXIOMS.md` |
| 4 | Named the generator transitions | Generator path EC2 | `02_control_theory/STEERING_GUIDE.md` |
| 5 | Mapped Lyapunov stability to ρ | Orbit contraction EC6–EC8 | `02_control_theory/CONTROL_THEOREMS.md` |
| 6 | Checked arithmetic reachability | Obstruction check EC11 | cert [111], cert [121] |
| 7 | Ran the machine validator | Certificate validation | cert [121] self-test |

This is the complete loop from physical intuition to verified machine certificate.

---

## Applying this pattern to your own system

1. **Identify three distinguishable regimes** — equilibrium, transient, target. These become your
   singularity, satellite, and cosmos states.
2. **Choose (b, e) encodings** such that `f(b,e) mod m` classifies correctly. Use the orbit table
   in `05_reference/QUICK_REFERENCE.md`.
3. **Name the generators** that drive the transitions. These can be physical operations,
   experimental interventions, or control inputs.
4. **Check the obstruction** before claiming reachability: compute `target_r = b·e` and verify
   `v_p(target_r) ≠ 1` for all inert primes of your modulus.
5. **Build the cert** following the schema in
   `qa_alphageometry_ptolemy/qa_engineering_core_cert/schemas/qa_engineering_core_cert_v1.schema.json`.
6. **Run the validator**. It will tell you exactly which checks fail and why.

For your specific engineering domain, see `06_classical_engineering_map/ENGINEERING_DOMAINS_QUICK_MAP.md`.
