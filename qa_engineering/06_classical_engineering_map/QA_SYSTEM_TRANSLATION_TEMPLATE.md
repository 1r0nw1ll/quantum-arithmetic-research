# QA System Translation Template

Use this template to map **any classical engineering system** to QA. Work through each section
in order. When complete, you have everything needed to build a cert [121] JSON certificate.

See `03_applied_domains/SPRING_MASS_WORKED_EXAMPLE.md` for a fully-filled example.

---

## The 10-minute version (quick start)

If you just want to try QA on your system:

1. Pick three states: equilibrium, transient, target
2. Assign (b, e) values in `{1,...,N}`
3. Check orbit types using `f(b,e) = b·b + b·e - e·e`
4. Name two transitions (state₁ → state₂ → state₃)
5. Compute `target_r = b·e` and check `v_p(target_r)` for inert primes
   - If `v_p = 1` → **STOP and re-encode the target**
6. Copy the JSON skeleton in Section 7 and fill it in
7. Run the validator

Then come back and fill the full template properly.

---

## 0. System identification

**System name**: _______________________________________________

**Physical domain**: ☐ mechanical ☐ electrical ☐ acoustic/cymatics ☐ seismic ☐ biological
☐ thermal ☐ fluid ☐ signal/communication ☐ neural ☐ financial ☐ other: _______________

**Governing equation or model** (brief):
```
[write here — e.g. mx'' + cx' + kx = F(t), or dx/dt = Ax + Bu, or ...]
```

**Modulus** (choose 9 for theoretical work, 24 for applied experiments): ___

---

## 1. State — identify three operating regimes

Every QA system needs exactly three structurally distinct states.
If you are unsure how to choose (b, e), start with the spring-mass example values
(`still`=(9,9), `transient`=(3,6), `steady_oscillation`=(1,2)) and adjust to fit your regime names.

The three required types:
`singularity` (fixed point / equilibrium), `satellite` (transient / near-resonance),
`cosmos` (stable limit cycle / target regime).

| QA label | (b, e) encoding | Orbit family | Physical description |
|----------|-----------------|--------------|----------------------|
| __________ | (__, __) | singularity | [your equilibrium: what does "at rest" mean?] |
| __________ | (__, __) | satellite | [your transient: what does "settling" look like?] |
| __________ | (__, __) | cosmos | [your target: what does "operational" look like?] |

**Encoding constraints**:
- All values must be in `{1,...,N}` where N = your modulus. **No zeros.**
- For modulus 9: valid values are 1–9. For modulus 24: valid values are 1–24.
- If your system has more than three states, map the most important three first. Additional
  states can be added to the cert later.

**Verification** (fill in before proceeding):

```
f(b, e) = b·b + b·e - e·e

singularity state (__, __):
  f = ___ ≡ ___ mod N
  (b,e) ≡ (0,0) mod N?  ☐ yes → singularity ✓   ☐ no → re-encode

satellite state (__, __):
  f = ___ ≡ ___ mod N
  v₃(f mod N) ≥ 2?       ☐ yes → satellite ✓     ☐ no → re-encode

cosmos state (__, __):
  f = ___ ≡ ___ mod N
  v₃(f mod N) = 0?        ☐ yes → cosmos ✓        ☐ no → re-encode
```

---

## 2. Dynamics — name the generator transitions

What physical operations or interventions drive the system from one regime to another?

| Step | Generator name | From state | To state | Physical operation |
|------|---------------|------------|----------|--------------------|
| 1 | ______________ | singularity | satellite | [e.g. "excite", "energize", "perturb"] |
| 2 | ______________ | satellite | cosmos | [e.g. "tune", "resonate", "stabilize"] |

**Notes**:
- Generator names must be non-empty strings (EC2).
- Use domain-specific vocabulary that your users will recognise (e.g. "fire", "propagate", "couple").
- If your system has intermediate states, list additional transitions below.

**Additional transitions** (optional):

| Step | Generator name | From | To | Physical operation |
|------|---------------|------|----|--------------------|
| | | | | |

---

## 3. Failure modes — map to QA fail types

What can go wrong when trying to drive this system?

| Failure label | QA fail type | Physical description |
|--------------|--------------|----------------------|
| ____________ | ☐ OUT_OF_BOUNDS ☐ PARITY ☐ PHASE_VIOLATION ☐ INVARIANT ☐ REDUCTION | |
| ____________ | ☐ OUT_OF_BOUNDS ☐ PARITY ☐ PHASE_VIOLATION ☐ INVARIANT ☐ REDUCTION | |

**QA fail type guide**:
- `OUT_OF_BOUNDS`: state exceeds domain limits (e.g. overdrive, saturation, runaway)
- `PARITY`: operation requires even values but odd values present (e.g. halving when indivisible)
- `PHASE_VIOLATION`: transition violates phase/timing constraint
- `INVARIANT`: norm f(b,e) changes in an illegal way
- `REDUCTION`: generator cannot reduce state as intended

---

## 4. Stability claim

**Lyapunov function** (must mention f, b, e, or a QA invariant):
```
[e.g. "f(b,e) = b*b + b*e - e*e (Q(sqrt5) norm; decreasing toward equilibrium along generator paths)"]
```

**Orbit contraction factor** (must be < 1.0):
```
ρ = ___________   (from Finite-Orbit Descent Theorem; use 0.001582 for mod-9 standard parameters)
```

**Equilibrium state label** (must map to singularity):
```
[the label you gave your singularity state above]
```

---

## 5. Controllability claim

**Classical controllability**:
☐ `full_rank` (Kalman rank condition satisfied)
☐ `partial` (some states unreachable classically)
☐ `unknown`

**Reachability witness** (required if full_rank):

BFS path from singularity to cosmos:
```
[state 1: your singularity label] → orbit_family: singularity
[state 2: your satellite label]   → orbit_family: satellite  (move: [generator name 1])
[state 3: your cosmos label]      → orbit_family: cosmos     (move: [generator name 2])
```

Path length k = ___ (minimum number of generator moves)

**Minimality witness** (optional but recommended):
- Proved no path shorter than k=___
- Excluded shorter lengths: ___
- How: ☐ BFS frontier exhaustion ☐ algebraic argument ☐ other: ___________

---

## 6. Arithmetic obstruction check (EC11 — do not skip)

This is the check classical analysis misses. Do it before claiming reachability.

**Target state**: _______________  (b = ___, e = ___)

```
target_r = b · e = ___ · ___ = ___

Inert primes for your modulus:
  mod  9: inert primes = {3}
  mod 24: inert primes = {3, 7}

For each inert prime p:
  v_p(target_r) = number of times p divides target_r

  p=3: v₃(___) = ___   Is this equal to 1?  ☐ yes → OBSTRUCTED (stop — re-encode target)
                                              ☐ no  → ok so far
  p=7: v₇(___) = ___   Is this equal to 1?  ☐ yes → OBSTRUCTED (stop — re-encode target)
  (mod 24 only)                               ☐ no  → ok

Obstruction result:  ☐ obstructed=true   ☐ obstructed=false
```

**If obstructed=true**: your target state is arithmetically unreachable regardless of what the
Kalman rank check says. You must choose a different (b, e) encoding for the target state. Refer
to `05_reference/QUICK_REFERENCE.md` for the allowed quadrea spectrum.

---

## 7. Build the cert

Once all sections above are filled and verified, copy this skeleton and fill in your values:

```json
{
  "schema_version": "QA_ENGINEERING_CORE_CERT.v1",
  "cert_type": "engineering_core",
  "certificate_id": "qa.cert.engineering_core.[your_system_name].v1",
  "title": "[Your system name] mapped to QA orbit traversal",
  "created_utc": "[ISO 8601 timestamp]",

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
    "description": "[one sentence describing your system]",
    "modulus": [9 or 24],
    "state_encoding": [
      { "label": "[singularity label]", "b": [b], "e": [e], "orbit_family": "singularity" },
      { "label": "[satellite label]",   "b": [b], "e": [e], "orbit_family": "satellite"   },
      { "label": "[cosmos label]",      "b": [b], "e": [e], "orbit_family": "cosmos"      }
    ],
    "transitions": [
      { "from": "[singularity label]", "to": "[satellite label]", "generator": "[name 1]" },
      { "from": "[satellite label]",   "to": "[cosmos label]",    "generator": "[name 2]" }
    ],
    "failure_modes": [
      { "label": "[failure label]", "qa_fail_type": "[QA fail type]" }
    ],
    "target_condition": {
      "label": "[cosmos label]",
      "orbit_family": "cosmos"
    },
    "equilibrium_state": "[singularity label]"
  },

  "stability_claim": {
    "lyapunov_function": "[your lyapunov function string]",
    "orbit_contraction_factor": [rho < 1.0],
    "contraction_verified": true
  },

  "controllability_claim": {
    "classical_controllability": "full_rank",
    "reachability_witness": {
      "algorithm": "BFS",
      "depth_bound": 24,
      "path": [
        { "state": "[singularity label]", "orbit_family": "singularity" },
        { "state": "[satellite label]",   "orbit_family": "satellite",  "move": "[name 1]" },
        { "state": "[cosmos label]",      "orbit_family": "cosmos",     "move": "[name 2]" }
      ],
      "path_length_k": [k]
    }
  },

  "obstruction_check": {
    "modulus": [9 or 24],
    "inert_primes": [3],
    "target_r": [target_r],
    "v_p_values": { "3": [v3 value] },
    "obstructed": [true or false]
  },

  "validation_checks": [],
  "fail_ledger": [],
  "result": "PASS"
}
```

**To validate**:
```bash
python qa_engineering_core_cert_validate.py \
  --cert your_cert.json
```

---

## 8. Common errors and how to fix them

| Error | Cause | Fix |
|-------|-------|-----|
| `STATE_ENCODING_INVALID` | b or e = 0, or exceeds modulus | Re-encode; use {1,...,N} not {0,...,N-1} |
| `ORBIT_FAMILY_CLASSIFICATION_FAILURE` | f(b,e) mod m doesn't match claimed family | Recompute f, use QUICK_REFERENCE orbit table |
| `ARITHMETIC_OBSTRUCTION_IGNORED` | target_r has v_p=1 for an inert prime | Choose different (b,e) for target; avoid r divisible by 3 once (mod 9) |
| `LYAPUNOV_QA_MISMATCH` | lyapunov_function doesn't mention f, b, or e | Include "f(b,e)" in the string |
| `CONTROLLABILITY_QA_MISMATCH` | full_rank declared but no reachability_witness | Add BFS path |
| `TRANSITION_NOT_GENERATOR` | generator field is empty string | Name every transition |

---

## Reference

- `QUICK_REFERENCE.md` — orbit table, allowed quadrea spectrum, hash formula
- `SPRING_MASS_WORKED_EXAMPLE.md` — completed version of this template
- `CLASSICAL_TO_QA_MAP.md` — full classical ↔ QA equivalence table
- `ENGINEERING_DOMAINS_QUICK_MAP.md` — domain-specific translation hints
- Cert [121] schema: `qa_alphageometry_ptolemy/qa_engineering_core_cert/schemas/qa_engineering_core_cert_v1.schema.json`
- Cert [121] validator: `qa_engineering_core_cert_validate.py`
