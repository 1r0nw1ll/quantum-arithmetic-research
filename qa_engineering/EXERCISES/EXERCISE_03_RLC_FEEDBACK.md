# Exercise 03 — RLC Feedback Circuit (Minimality Witness)

**Difficulty**: intermediate-advanced
**Time**: 20–30 minutes
**Prerequisite**: complete Exercises 01 and 02
**New skill**: proving that k=2 is the *shortest possible* path, not just *a* valid path (EC10)

---

## The new idea: reachable ≠ provably shortest

In Exercises 01 and 02 you found a valid path of length k=2. That's reachability (EC9).

This exercise introduces a harder question:

> **Is k=2 the minimum? Or could k=1 get there?**

In classical control theory, time-optimal control asks this. In QA, EC10 requires a
**minimality witness** — a proof that no shorter path exists.

The minimality witness is not optional when an `optimization_claim` is present in the cert. A
cert that claims optimality without proving it will be rejected.

---

## The system

An RLC circuit (resistor + inductor + capacitor in series) has a richer transient than an RC
circuit. It exhibits damped oscillation before settling:

```
L·I'' + R·I' + (1/C)·I = V'(t)
```

Three states:

| State label | Physical description |
|-------------|----------------------|
| `quiescent` | No current. Circuit at rest. All reactive elements uncharged. |
| `ringing` | Damped oscillation. Inductor and capacitor exchanging energy; envelope decaying. |
| `settled` | Oscillation damped out. Current stable at DC value. Steady state. |

Three transitions:

| Generator | From | To | Physical meaning |
|-----------|------|----|-----------------|
| `energise` | `quiescent` | `ringing` | Apply step input; L and C begin transient exchange |
| `damp` | `ringing` | `settled` | R dissipates oscillation energy; envelope falls |
| `lock` | `settled` | `settled` | Confirm steady state; no-op path (needed for completeness) |

**Modulus**: 9

**State encoding** (given):

| Label | (b, e) | Orbit family |
|-------|--------|--------------|
| `quiescent` | (9, 9) | singularity |
| `ringing` | (3, 6) | satellite |
| `settled` | (1, 2) | cosmos |

---

## Step 1 — Verify all three states

Complete the arithmetic:

```
f(9, 9) = ___ ≡ ___ mod 9  →  orbit family: ___________
f(3, 6) = ___ ≡ ___ mod 9  →  orbit family: ___________
f(1, 2) = ___ ≡ ___ mod 9  →  orbit family: ___________
```

---

## Step 2 — Arithmetic obstruction check

```
settled = (b=1, e=2)
target_r = ___
v₃(target_r) = ___
obstructed = ___
```

(You have seen this before. It should now feel automatic.)

---

## Step 3 — The reachability path (EC9)

The path from `quiescent` to `settled` takes k=2 steps:

```
quiescent  →[energise]→  ringing  →[damp]→  settled
```

This is EC9: a valid BFS path exists. But is it the *shortest*?

---

## Step 4 — The minimality witness (EC10) — this is the new part

EC10 requires proof that no path of length k=1 exists. That means: from `quiescent`, can
`energise` alone (or any single generator) reach `settled` in one step?

**BFS depth-1 frontier from `quiescent`**:

From `quiescent`, there is exactly one defined transition: `energise` → `ringing`.

| From | Generator | To |
|------|-----------|-----|
| quiescent | energise | ringing |

Is `settled` in the depth-1 frontier? ☐ yes ☐ no

Therefore: the minimum path length is k = ___.

**Complete the minimality witness**:

```
proved_no_path_shorter_than: ___
excluded_shorter_lengths: [___]
frontier_sizes:
  depth_1: ___   (how many distinct states reachable in 1 step from quiescent?)
  depth_2: ___   (how many distinct states reachable in 2 steps?)
```

---

## Step 5 — Build the cert

Fill in the blanks (`[BLANK]`) and paste the completed JSON into a file.

```json
{
  "schema_version": "QA_ENGINEERING_CORE_CERT.v1",
  "cert_type": "engineering_core",
  "certificate_id": "qa.cert.engineering_core.rlc_feedback.exercise03.v1",
  "title": "Exercise 03 — RLC feedback circuit with minimality witness",
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
    "description": "RLC series circuit: damped oscillation before settling. L·I'' + R·I' + (1/C)·I = V'(t).",
    "modulus": 9,
    "state_encoding": [
      { "label": "quiescent", "b": 9, "e": 9, "orbit_family": "singularity" },
      { "label": "ringing",   "b": 3, "e": 6, "orbit_family": "satellite"   },
      { "label": "settled",   "b": 1, "e": 2, "orbit_family": "cosmos"      }
    ],
    "transitions": [
      { "from": "quiescent", "to": "ringing", "generator": "energise" },
      { "from": "ringing",   "to": "settled", "generator": "damp"     },
      { "from": "settled",   "to": "settled", "generator": "lock"     }
    ],
    "failure_modes": [
      { "label": "overload",       "qa_fail_type": "OUT_OF_BOUNDS" },
      { "label": "resonance_lock", "qa_fail_type": "PHASE_VIOLATION" }
    ],
    "target_condition": {
      "label": "settled",
      "orbit_family": "cosmos"
    },
    "equilibrium_state": "quiescent"
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
        { "state": "quiescent", "orbit_family": "singularity" },
        { "state": "ringing",   "orbit_family": "satellite",  "move": "energise" },
        { "state": "settled",   "orbit_family": "cosmos",     "move": "damp"     }
      ],
      "path_length_k": 2
    },
    "optimization_claim": {
      "objective": "minimal_k",
      "minimality_witness": {
        "proved_no_path_shorter_than": [BLANK: your answer from Step 4],
        "excluded_shorter_lengths": [[BLANK: your answer from Step 4]],
        "frontier_sizes": {
          "depth_1": [BLANK: your answer from Step 4],
          "depth_2": [BLANK: your answer from Step 4]
        }
      }
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
  --cert rlc_feedback_cert.json
```

---

## Expected result

```
[PASS] qa.cert.engineering_core.rlc_feedback.exercise03.v1
  result   : PASS
  checks   : 0/0
```

**If you get `CONTROLLABILITY_QA_MISMATCH`**: the `minimality_witness` is missing or
`optimization_claim` is malformed. Check that `proved_no_path_shorter_than` is an integer
(not a string), `excluded_shorter_lengths` is a list, and `frontier_sizes` has both keys.

---

## Full answers

<details>
<summary>Click to reveal (try it yourself first)</summary>

**Step 1**:
```
f(9,9) = 81+81-81 = 81 ≡ 0 mod 9  →  singularity ✓
f(3,6) = 9+18-36 = -9 ≡ 0 mod 9   →  satellite ✓
f(1,2) = 1+2-4  = -1 ≡ 8 mod 9    →  cosmos ✓
```

**Step 2**: `target_r = 2`, `v₃(2) = 0`, `obstructed = false`

**Step 4 — minimality witness**:
- Depth-1 frontier from `quiescent`: only `ringing` (via `energise`). Size = 1.
- `settled` is not in depth-1 frontier.
- Depth-2 frontier: `settled` (via `energise` → `damp`). Size = 1.
- Therefore: `proved_no_path_shorter_than = 2`, `excluded_shorter_lengths = [1]`,
  `frontier_sizes = {"depth_1": 1, "depth_2": 1}`

</details>

---

## The distinction this exercise teaches

| Check | Question it answers | Passes for this cert? |
|-------|--------------------|-----------------------|
| EC9 | Is the target reachable at all? | Yes — BFS path k=2 found |
| EC10 | Is k=2 the shortest possible path? | Yes — depth-1 frontier doesn't contain `settled` |
| EC11 | Is the target arithmetically admissible? | Yes — v₃(2)=0 |

All three are independent. EC9 being true does not imply EC10. EC10 does not imply EC11.
A cert that passes EC9 but skips EC10 is making an unproved optimality claim.

---

## Going further

- **What if you removed the `lock` self-loop?** Would the cert still pass? (Yes — it's not on
  the primary path. But it is part of the classical model and should be declared.)
- **What if there were two paths of length 2?** Would the minimality witness change? (No — EC10
  only requires that no path shorter than k exists, not that k-length paths are unique.)
- **Exercise 04**: map a system from your own engineering domain using only the translation
  template. No scaffolding provided. Aim for gallery quality.
