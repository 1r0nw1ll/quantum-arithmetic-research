# Exercise 01 — Thermostat

**Difficulty**: beginner
**Time**: 10–15 minutes
**What you need**: this file, `QUICK_REFERENCE.md`, a calculator (or pen and paper)
**What you'll produce**: a completed QA cert for a thermostat system

---

## The system

A thermostat controls a heating system with three recognisable states:

| State label | Physical description |
|-------------|----------------------|
| `off` | Heater off. Room at rest. Below setpoint. |
| `heating` | Heater running. Room temperature rising toward setpoint. |
| `at_temperature` | Heater cycling at minimum. Room stable at setpoint. |

Two control actions drive the system:
- **call_for_heat**: the thermostat switches the heater on (`off` → `heating`)
- **reach_setpoint**: room reaches setpoint; heater drops to maintenance cycling (`heating` → `at_temperature`)

This is the simplest possible three-state thermal control system. You will map it to QA.

---

## What is already filled in

The following values have been chosen for you. Your job is to fill in the **three blanks**.

**Modulus**: 9

**State encoding**:

Each state is represented as a pair of numbers `(b, e)`. Two things describe each state:
- **QA family name** — the named family it belongs to (Ninbonacci, Tribonacci, Fibonacci, Lucas, or Phibonacci)
- **Orbit classification** — the structural class (singularity, satellite, or cosmos)

You will work both out for the `heating` state in Blank A.

| Label | (b, e) | QA family name | Orbit classification | Calculated value |
|-------|--------|---------------|---------------------|-----------------|
| `off` | (9, 9) | Ninbonacci | singularity | 0 |
| `heating` | (3, 6) | **[BLANK A — family name]** | **[BLANK A — classification]** | **[BLANK A — calculated result]** |
| `at_temperature` | (1, 2) | Fibonacci | cosmos | 8 |

**Transitions**:

| Generator | From | To |
|-----------|------|----|
| `call_for_heat` | `off` | `heating` |
| **[BLANK B]** | `heating` | `at_temperature` |

**Obstruction check**:

```
at_temperature = (b=1, e=2)
target_r = b · e = 1 · ___ = ___   [BLANK C]
v₃(target_r) = ___                  [BLANK C]
obstructed = ___                    [BLANK C]
```

---

## Your task

### Blank A — classify the `heating` state

Compute `f(3, 6)` using `f(b, e) = b·b + b·e - e·e`:

```
f(3, 6) = (3 × 3) + (3 × 6) - (6 × 6)
        = ___ + ___ - ___
        = ___
        ≡ ___ mod 9
```

Now classify:
- Is (3, 6) ≡ (0, 0) mod 9? → singularity
- Is f mod 9 = 0? → satellite
- Is f mod 9 not divisible by 3? → cosmos

**QA family name for `heating`**: _______________

**Orbit classification for `heating`**: _______________

Fill in both columns in the table above.

---

### Blank B — name the second generator

The transition from `heating` to `at_temperature` is the moment the room reaches setpoint and
the heater drops to maintenance cycling. Choose a name that describes this action from the
perspective of the thermostat system.

**Your generator name**: _______________

(There is no single correct answer — the name should be meaningful to your domain.
The spring-mass example used `tune`. You might use `reach_setpoint`, `stabilize`, `cycle`, or
anything else that describes the physical action.)

---

### Blank C — arithmetic obstruction check

```
at_temperature = (b=1, e=2)
target_r = b · e = 1 · 2 = ___

Inert prime for mod 9: p = 3
v₃(___) = number of times 3 divides ___

Does 3 divide target_r?   ☐ yes (once: v₃=1 → OBSTRUCTED)
                           ☐ no  (v₃=0 → not obstructed)

obstructed = ___
```

---

## Expected answers

<details>
<summary>Click to reveal (try it yourself first)</summary>

**Blank A**:
```
f(3, 6) = 9 + 18 - 36 = -9 ≡ 0 mod 9
v₃(0 mod 9) — more precisely: v₃(-9) = v₃(9) = 2 ≥ 2 → satellite
```
QA family name: **Tribonacci** | Orbit classification: **satellite** | Calculated value: **0**

**Blank B**: any non-empty string is valid. `reach_setpoint` is the most literal.

**Blank C**:
```
target_r = 1 · 2 = 2
v₃(2) = 0   (3 does not divide 2)
obstructed = false
```

</details>

---

## Build the cert

Once you have all three blanks filled, paste this JSON into a new file, substitute your generator
name for `[BLANK_B]`, and run the validator.

**Before saving**: make sure you save it as a plain `.json` file — not a `.txt` file, and without
any code fence markers (no ` ```json ` at the top, no ` ``` ` at the bottom). If you copy from a
chat window, delete those lines before saving. The file should start with `{` on the first line.

```json
{
  "schema_version": "QA_ENGINEERING_CORE_CERT.v1",
  "cert_type": "engineering_core",
  "certificate_id": "qa.cert.engineering_core.thermostat.exercise01.v1",
  "title": "Exercise 01 — thermostat mapped to QA orbit traversal",
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
    "description": "Thermostat: three-state thermal control system mapped to mod-9 QA orbits.",
    "modulus": 9,
    "state_encoding": [
      { "label": "off",             "b": 9, "e": 9, "orbit_family": "singularity" },
      { "label": "heating",         "b": 3, "e": 6, "orbit_family": "satellite"   },
      { "label": "at_temperature",  "b": 1, "e": 2, "orbit_family": "cosmos"      }
    ],
    "transitions": [
      { "from": "off",     "to": "heating",        "generator": "call_for_heat" },
      { "from": "heating", "to": "at_temperature", "generator": "[BLANK_B]"     }
    ],
    "failure_modes": [
      { "label": "overheat",   "qa_fail_type": "OUT_OF_BOUNDS" },
      { "label": "short_cycle","qa_fail_type": "PARITY"        }
    ],
    "target_condition": {
      "label": "at_temperature",
      "orbit_family": "cosmos"
    },
    "equilibrium_state": "off"
  },

  "stability_claim": {
    "lyapunov_function": "f(b,e) = b*b + b*e - e*e (Q(sqrt5) norm; decreasing toward equilibrium)",
    "orbit_contraction_factor": 0.001582,
    "contraction_verified": true
  },

  "controllability_claim": {
    "classical_controllability": "full_rank",
    "reachability_witness": {
      "algorithm": "BFS",
      "depth_bound": 24,
      "path": [
        { "state": "off",            "orbit_family": "singularity" },
        { "state": "heating",        "orbit_family": "satellite",  "move": "call_for_heat" },
        { "state": "at_temperature", "orbit_family": "cosmos",     "move": "[BLANK_B]"     }
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

**Run it** (from the `qa_engineering/` folder):
```bash
python qa_engineering_core_cert_validate.py --cert your_thermostat_cert.json
```

**Expected output**:
```
[PASS] qa.cert.engineering_core.thermostat.exercise01.v1
  result   : PASS
  checks   : 0/0
```

---

## What you just did

You completed the full QA engineering loop for a real (if simple) system:

| Layer | What you did |
|-------|-------------|
| State | Mapped three physical regimes to (b,e) pairs |
| Dynamics | Named the generator transitions |
| Orbit classification | Computed f(b,e) and identified satellite vs cosmos |
| Obstruction check | Verified target is arithmetically reachable |
| Certificate | Built a machine-checkable JSON cert |
| Validation | Ran the validator and got PASS |

---

## Expected result

When you run the validator on your completed cert, you should see:

```
[PASS] qa.cert.engineering_core.thermostat.exercise01.v1
  result   : PASS
  checks   : 0/0
```

Key values to confirm:
- QA family sequence: `Ninbonacci → Tribonacci → Fibonacci`
- Orbit classification sequence: `singularity → satellite → cosmos`
- Path length: `k = 2`
- `target_r = 2`, `v₃(2) = 0` → not obstructed

**If you do NOT get PASS, check these in order:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `STATE_ENCODING_INVALID` | a b or e value is 0 or > 9 | Use {1,...,9} — no zeros |
| `ORBIT_FAMILY_CLASSIFICATION_FAILURE` | orbit_family doesn't match f(b,e) | Recheck your Blank A computation |
| `TRANSITION_NOT_GENERATOR` | generator field is empty or `"[BLANK_B]"` not replaced | Fill in your generator name |
| `ARITHMETIC_OBSTRUCTION_IGNORED` | target_r has v₃=1 | target_r=2 is correct; verify (b=1, e=2) |
| `LYAPUNOV_QA_MISMATCH` | lyapunov_function doesn't mention f | Copy the string from the skeleton exactly |

---

## Going further

- **Try a failure**: change `at_temperature` to `(b=1, e=3)`. Recompute target_r. Run the validator. What error do you get?
- **Map your own system**: use `QA_SYSTEM_TRANSLATION_TEMPLATE.md` with a system from your own domain.
- **Read the spring-mass example**: `03_applied_domains/SPRING_MASS_WORKED_EXAMPLE.md` for a fuller walkthrough including the stability and minimality claims.
