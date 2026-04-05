# Exercise 02 — RC Circuit (Charging Capacitor)

**Difficulty**: intermediate
**Time**: 15–20 minutes
**Prerequisite**: complete Exercise 01 first
**What makes this different**: you will hit EC11 and recover from it

---

## The system

An RC circuit charges a capacitor through a resistor:

```
V(t) = V₀ (1 − e^{−t/RC})
```

Three recognisable states:

| State label | Physical description |
|-------------|----------------------|
| `uncharged` | Capacitor fully discharged. No voltage across plates. |
| `charging` | Current flowing. Voltage rising exponentially toward V₀. |
| `charged` | Capacitor at V₀. Current approaches zero. Stable. |

Two transitions:
- **apply_voltage**: connect supply across uncharged capacitor (`uncharged` → `charging`)
- **reach_V0**: voltage converges to V₀; transient ends (`charging` → `charged`)

**Modulus**: 9

---

## What is already filled in

| Label | (b, e) | Orbit family |
|-------|--------|--------------|
| `uncharged` | (9, 9) | singularity |
| `charging` | (3, 6) | satellite |
| `charged` | **(1, 3)** | cosmos |

Generators: `apply_voltage`, `reach_V0`

---

## Step 1 — Verify the first two states

Confirm the orbit families for `uncharged` and `charging` using `f(b, e) = b·b + b·e - e·e`:

```
f(9, 9) = ___ + ___ − ___ = ___ ≡ ___ mod 9
(9,9) ≡ (0,0) mod 9 → singularity ✓

f(3, 6) = ___ + ___ − ___ = ___ ≡ ___ mod 9
f ≡ 0 mod 9, v₃(f) ≥ 2 → satellite ✓
```

These match Exercise 01. Same encoding, same orbit families — the QA structure carries across domains.

---

## Step 2 — Check the arithmetic obstruction for `charged = (1, 3)`

This is the critical step. **Do not skip it.**

```
charged = (b=1, e=3)
target_r = b · e = 1 · 3 = ___

Inert prime for mod 9: p = 3
v₃(___) = number of times 3 divides ___

Does 3 divide target_r exactly once?   ☐ yes (v₃ = 1 → OBSTRUCTED)
                                         ☐ no  (not obstructed → continue)
```

**What do you get?**

---

## Step 3 — What happens if you ignore the obstruction

If you ignore the result above and build the cert with `(b=1, e=3)` and `obstructed: false`, the validator catches it:

```
[FAIL] qa.cert.engineering_core.rc_circuit.exercise02.v1
  result   : FAIL
  errors:
    - Recomputed engineering failures ['ARITHMETIC_OBSTRUCTION_IGNORED'] but result=PASS — inconsistency
```

This is the same error as the arithmetic obstruction fixture in cert [121]. Classical Kalman rank analysis would certify this system as fully controllable — the transition graph is complete, the generator names are valid, everything looks fine. But `v₃(3) = 1` and 3 is inert in Z[φ], so the target is arithmetically unreachable regardless.

**QA is telling you something the circuit equations cannot.**

---

## Step 4 — Re-encode the `charged` state

You need to find a `(b, e)` encoding for the `charged` state that:

1. Gives orbit family `cosmos` (v₃(f(b,e)) = 0)
2. Gives `target_r = b·e` with `v₃(target_r) ≠ 1`

Try `(b=1, e=2)`:

```
f(1, 2) = ___ + ___ − ___ = ___ ≡ ___ mod 9
v₃(f mod 9) = ___   → orbit family: ___

target_r = 1 · 2 = ___
v₃(___) = ___   → obstructed: ___
```

Does this work? ☐ yes ☐ no

---

## Step 5 — Build and validate the corrected cert

Replace `charged` with `(b=1, e=2)` in the JSON below and fill in your generator name if you
want to use something other than `reach_V0`.

```json
{
  "schema_version": "QA_ENGINEERING_CORE_CERT.v1",
  "cert_type": "engineering_core",
  "certificate_id": "qa.cert.engineering_core.rc_circuit.exercise02.v1",
  "title": "Exercise 02 — RC charging circuit mapped to QA orbit traversal",
  "created_utc": "2026-03-24T00:00:00Z",

  "inherits_from": "QA_CORE_SPEC.v1",
  "spec_scope": "family_extension",
  "core_kernel_compatibility": {
    "state_space_compatible": true,
    "generator_alphabet_compatible": true,
    "invariants_preserved": true,
    "logging_contract_compatible": true,
    "gate_policy_respected": [0, 1, 2, 3, 4, 5]
  },

  "classical_system": {
    "description": "RC circuit: capacitor charging through resistor V(t) = V0(1 - exp(-t/RC)). Three-state model mapped to mod-9 QA orbits.",
    "modulus": 9,
    "state_encoding": [
      { "label": "uncharged", "b": 9, "e": 9, "orbit_family": "singularity" },
      { "label": "charging",  "b": 3, "e": 6, "orbit_family": "satellite"   },
      { "label": "charged",   "b": 1, "e": 2, "orbit_family": "cosmos"      }
    ],
    "transitions": [
      { "from": "uncharged", "to": "charging", "generator": "apply_voltage" },
      { "from": "charging",  "to": "charged",  "generator": "reach_V0"      }
    ],
    "failure_modes": [
      { "label": "overvoltage",   "qa_fail_type": "OUT_OF_BOUNDS" },
      { "label": "dielectric_breakdown", "qa_fail_type": "INVARIANT" }
    ],
    "target_condition": {
      "label": "charged",
      "orbit_family": "cosmos"
    },
    "equilibrium_state": "uncharged"
  },

  "stability_claim": {
    "lyapunov_function": "f(b,e) = b*b + b*e - e*e (Q(sqrt5) norm; decreasing toward equilibrium along generator paths with rho < 1)",
    "orbit_contraction_factor": 0.001582,
    "contraction_verified": true
  },

  "controllability_claim": {
    "classical_controllability": "full_rank",
    "reachability_witness": {
      "algorithm": "BFS",
      "depth_bound": 24,
      "path": [
        { "state": "uncharged", "orbit_family": "singularity" },
        { "state": "charging",  "orbit_family": "satellite",  "move": "apply_voltage" },
        { "state": "charged",   "orbit_family": "cosmos",     "move": "reach_V0"      }
      ],
      "path_length_k": 2
    }
  },

  "obstruction_check": {
    "modulus": 9,
    "inert_primes": [3],
    "target_r": 2,
    "v_p_values": { "3": 0 },
    "obstructed": false
  },

  "validation_checks": [],
  "fail_ledger": [],
  "result": "PASS"
}
```

```bash
python qa_engineering_core_cert_validate.py \
  --cert rc_circuit_cert.json
```

---

## Expected result

```
[PASS] qa.cert.engineering_core.rc_circuit.exercise02.v1
  result   : PASS
  checks   : 0/0
```

Key values:
- Orbit sequence: `singularity → satellite → cosmos`
- Path length: `k = 2`
- `target_r = 2` (after re-encoding), `v₃(2) = 0` → not obstructed

**If you still get FAIL after re-encoding, check**: is `charged` set to `(b=1, e=2)` in *both*
`state_encoding` and `obstruction_check.target_r`? Both must be updated.

---

## What you just learned

<details>
<summary>Full answers (check your work)</summary>

**Step 1**:
```
f(9,9) = 81 + 81 - 81 = 81 ≡ 0 mod 9 → singularity ✓
f(3,6) = 9 + 18 - 36 = -9 ≡ 0 mod 9, v₃(-9)=2 → satellite ✓
```

**Step 2**: `target_r = 3`, `v₃(3) = 1` → **OBSTRUCTED**. The (1,3) encoding is arithmetically forbidden.

**Step 3**: The validator outputs `ARITHMETIC_OBSTRUCTION_IGNORED`. Classical controllability cannot catch this.

**Step 4**:
```
f(1,2) = 1 + 2 - 4 = -1 ≡ 8 mod 9, v₃(8)=0 → cosmos ✓
target_r = 1·2 = 2, v₃(2) = 0 → not obstructed ✓
```
`(1,2)` works. `(1,3)` does not.

</details>

---

## Why (1,3) failed but (1,2) works

| Encoding | target_r | v₃ | Result |
|----------|----------|----|--------|
| (b=1, e=3) | 3 | 1 | **OBSTRUCTED** — forbidden |
| (b=1, e=2) | 2 | 0 | not obstructed — allowed |

Both states are **cosmos** — `f(1,3) ≡ 4 mod 9` and `f(1,2) ≡ 8 mod 9` both have `v₃ = 0`.
EC5 (orbit classification) passes for both. Only EC11 (arithmetic reachability) distinguishes them.

This is the core structural fact:

> **classification ≠ reachability.**
> A state can be correctly classified as cosmos and still be arithmetically unreachable.

Orbit family answers: *what kind of state is this?*
Arithmetic admissibility answers: *can this state be reached?*

These are independent questions. QA checks both. Classical control theory checks only the first.

---

## Going further

- **Find all obstructed cosmos states for mod 9**: which (b,e) pairs classify as cosmos but have `v₃(b·e) = 1`? (Hint: look for pairs where `b·e` is exactly divisible by 3 once.)
- **Try mod 24**: the inert primes are {3, 7}. Which target_r values are now forbidden?
- **Exercise 03**: map an RLC circuit (three energy storage elements → four states). Use the template.
