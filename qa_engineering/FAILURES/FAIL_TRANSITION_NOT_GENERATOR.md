# FAIL: TRANSITION_NOT_GENERATOR

**Trigger**: a transition has an empty, whitespace-only, or missing `generator` field.

---

## Broken cert (minimal)

```json
"transitions": [
  { "from": "off", "to": "heating", "generator": "" },
  { "from": "heating", "to": "steady", "generator": "   " }
]
```

## Validator output

```
[FAIL]
  errors:
    - Recomputed engineering failures ['TRANSITION_NOT_GENERATOR'] but result=PASS — inconsistency
```

## Why it fails

In QA, every state transition must be driven by a **named generator**. An unnamed transition has
no QA identity — it cannot be certified, logged, or audited. It is equivalent to an unmodelled
control input, which QA does not permit.

This commonly happens when:
- copying a transition table from a classical state diagram that uses arrow labels like `"→"` or
  `"input"`
- leaving the `generator` field as a placeholder during drafting
- using the template without filling in Section 2

## The fix

Every transition must have a non-empty, non-whitespace generator name:

```json
"transitions": [
  { "from": "off",     "to": "heating", "generator": "call_for_heat" },
  { "from": "heating", "to": "steady",  "generator": "reach_setpoint" }
]
```

The name should be:
- **domain-specific** (use the language of your system)
- **a verb or verb phrase** (it is an action, not a state)
- **unique within the cert** (if two transitions share a name, they are the same generator)

Common examples by domain:

| Domain | Generator names |
|--------|----------------|
| Thermal | `call_for_heat`, `reach_setpoint`, `cool_down` |
| Electrical | `apply_voltage`, `charge`, `discharge`, `switch_on` |
| Mechanical | `excite`, `tune`, `damp`, `lock` |
| Seismic | `nucleate`, `propagate`, `attenuate` |
| Acoustic | `drive`, `resonate`, `decay` |
| Neural | `activate`, `converge`, `reset` |

---

**Cert family reference**: EC2 in `qa_engineering_core_cert/qa_engineering_core_cert_validate.py`
