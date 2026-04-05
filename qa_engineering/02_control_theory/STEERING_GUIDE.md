# QA Steering Guide: How to Move a System Toward a Target

This is the practical guide for QA engineering. Given a system currently in some state, how do you move it to a target state? This is the **steering problem**.

---

## The Steering Problem (Formal)

Given:
- A current state `s₀ = (b₀, e₀)` and its orbit family (singularity / satellite / cosmos)
- A target state or orbit family `s_target`
- A generator set Σ available to you

Find: A minimal sequence of generators `[g₁, g₂, ..., gₖ]` such that applying them to `s₀` reaches `s_target` (or an element of the target orbit family), with all invariants preserved.

---

## Step 1: Classify Your Current State

Before steering, know where you are:

```python
def classify_state(b, e, m=9):
    """Classify (b,e) into singularity/satellite/cosmos orbit family."""
    f = (b*b + b*e - e*e) % m
    if b == 0 and e == 0:
        return "singularity"
    v3 = 0
    temp = f if f != 0 else m
    while temp % 3 == 0:
        v3 += 1
        temp //= 3
    if v3 >= 2:
        return "satellite"
    return "cosmos"
```

Or for the 4-tuple version: compute the Q(√5) norm `f(b,e) = b² + be - e²` and check its 3-adic valuation:
- `v₃(f) ≥ 2` → satellite
- `v₃(f) = 0` → cosmos
- `(b,e) = (0,0)` mod m → singularity

---

## Step 2: Check Reachability First

**Before planning, check if the target is arithmetically reachable.**

The Obstruction Spine rule: if your target has `v_p(r) = 1` for an inert prime p, it is **arithmetically forbidden** — no generator sequence will reach it. Check this before doing any BFS.

The inert primes for QA (mod-9): p = 3. For mod-24: p = 3, p = 7.

Example: target r = 6 with p = 3:
- `v₃(6) = 1` (6 = 2 × 3, once divisible by 3)
- 3 is inert in Z[φ]
- Therefore: target is unreachable, prune immediately, nodes_expanded = 0

---

## Step 3: Plan the Generator Sequence

If the target is reachable, use BFS over the orbit graph:

```
BFS from s₀:
  frontier = [s₀]
  visited = {s₀}
  parent = {}

  while frontier not empty:
    s = frontier.pop()
    if s is in target orbit family:
      return reconstruct_path(parent, s)
    for each generator g in Σ:
      s_new = apply(g, s)
      if s_new is not FAILURE and s_new not in visited:
        parent[s_new] = (s, g)
        visited.add(s_new)
        frontier.append(s_new)
```

BFS guarantees the **shortest** generator sequence (minimal k).

---

## Step 4: The Canonical Steering Trajectory

For most SVP engineering applications, the target is **cosmos** (full resonance) starting from **singularity** (quiescence). The canonical path is:

```
singularity → satellite → cosmos
```

with path length k = 2. This is not a special case — it is a theorem (Control Stack Theorem, cert [117]).

**The two generator sequence**:
1. A **first generator** that moves from singularity into satellite (transitional activation)
2. A **second generator** that moves from satellite into cosmos (full resonance achieved)

What those generators are depends on your domain:
- Cymatics: `increase_amplitude` → `set_frequency`
- Seismology: `increase_gain` → `apply_lowpass`
- Neural network: `σ` (increment e) → `μ` (swap, rebalance)
- Your domain: identify what moves your system from quiescence to transition, then to full activation

---

## Step 5: Certify Your Plan

A QA steering plan is not complete until it is **certified**. This means:

1. **Planner cert** (`QA_CYMATIC_PLANNER_CERT.v1` or equivalent):
   - Documents the BFS search: algorithm, depth bound, frontier sizes
   - Provides the plan witness: ordered generator sequence + intermediate states
   - If no plan exists: documents the obstruction class (why no path within bound)

2. **Control cert** (`QA_CYMATIC_CONTROL_CERT.v1` or equivalent):
   - Executes the plan step by step
   - Verifies each transition is legal (no illegal moves)
   - Confirms the final state is in the target orbit family

3. **Compiler cert** (`QA_PLAN_CONTROL_COMPILER_CERT.v1`):
   - Hash-pins both the planner cert and the control cert
   - Verifies they agree on: initial state, target state, generator sequence, path length, orbit family
   - This is what makes the plan tamper-evident

---

## Failure Handling During Steering

If a generator fails mid-sequence, do not ignore it. Record:
- Which generator failed (`move`)
- The failure type (`fail_type`)
- The invariant state at failure (`invariant_diff`)

Common steering failures and their meanings:

| Failure | During steering | Meaning |
|---------|----------------|---------|
| `OUT_OF_BOUNDS` | σ or λ applied at edge of Caps | Reduce step size or switch generator |
| `PARITY` | ν applied to odd state | State is not in the ν-domain; use σ first to reach an even state |
| `GOAL_NOT_REACHED` | Control cert | Plan executed but final state is wrong family; check BFS depth bound |
| `NO_PLAN_WITHIN_BOUND` | Planner cert | Target not reachable within max_depth; increase depth or check obstruction |
| `ILLEGAL_TRANSITION` | Control cert | A generator was applied outside its legal precondition; fix the plan |

---

## Steering in Your Own Domain

To apply QA steering to a new domain:

1. **Map your states** to `(b, e)` pairs. Every distinct configuration of your system corresponds to a QA state.

2. **Map your operations** to generators. What actions can you take? Which ones increment a coordinate (σ)? Which swap roles (μ)? Which scale up (λ) or down (ν)?

3. **Map your failure modes** to QA failure types. What goes wrong in your domain? Does it correspond to going out of bounds? A parity violation? An invariant break?

4. **Register your domain** in the compiler's `DOMAIN_TO_FIXTURES` mapping.

5. **Run the planner** on your mapped state space. The orbit structure you discovered will immediately tell you what is and isn't achievable.

---

## Signal Injection (Dynamic Steering)

For continuous systems (audio, sensor streams, neural network gradients), use **signal injection**:

1. The external signal influences the `b` state variable
2. The modified `(b, e)` generates a proposed new state
3. QA computes the resonance coupling via `np.einsum('ik,jk->ij', tuples, tuples)`
4. Neighbor pull is computed using the weighted adjacency
5. State updates with noise annealing: `noise * (NOISE_ANNEALING ** t)`
6. Modular arithmetic keeps everything within Caps(N, N)

This creates a **self-organizing system** where external signals and internal QA dynamics interact. The weight matrix updates based on tuple resonance (not pre-defined), producing adaptive coupling.

See `run_signal_experiments_final.py` for the full implementation.

---

## Source References

- Obstruction spine: cert families [111]–[116]
- Compiler law: cert family [106]
- Control stack theorem: cert family [117]
- Signal injection: `run_signal_experiments_final.py:71-100`
- Applied domain examples: `../03_applied_domains/`
