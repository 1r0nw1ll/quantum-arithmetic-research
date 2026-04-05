# Exercise Track — Tier 4 Engineering Progression

Five exercises. Each teaches exactly one new capability. Read this before building or assigning
any exercise.

---

## Progression map

| # | System | Skill target | Likely fail mode | Expected artifact | Gallery? |
|---|--------|-------------|-----------------|-------------------|----------|
| 01 | Thermostat | Encode three states, classify orbits, pass EC11 | `STATE_ENCODING_INVALID` (zero-indexing) | PASS cert, k=2 | No — too scaffolded |
| 02 | RC circuit | Hit EC11, recover, understand classification ≠ reachability | `ARITHMETIC_OBSTRUCTION_IGNORED` | PASS cert after re-encoding | No — obstruction is planted |
| 03 | RLC / feedback | Prove shortest path (minimality witness, EC10); three transitions | `CONTROLLABILITY_QA_MISMATCH` (missing minimality_witness) | PASS cert with minimality witness | Optional |
| 04 | Builder's own domain | Map an independent system end-to-end using the template with no scaffolding | Any — builder encounters their own domain's failure | PASS cert for a system they chose | **Yes — primary gallery source** |
| 05 | Two connected subsystems | Compose certs: inheritance, family extension, [107]→[121] chain | `INVALID_KERNEL_REFERENCE` or `SPEC_SCOPE_MISMATCH` | Two certs with explicit inheritance edge | Yes — advanced gallery |

---

## Skill targets (one sentence each)

**01**: A builder can encode any three-regime system into QA states and get a validator PASS.

**02**: A builder understands that orbit family classification (EC5) and arithmetic reachability (EC11) are independent checks, and can recover from an obstruction by re-encoding.

**03**: A builder can prove that a k-step path is the *shortest possible* path, not just *a* valid path.

**04**: A builder can take any system from their own domain, apply the translation template without guidance, and produce a valid cert.

**05**: A builder understands how individual system certs compose into a larger certified architecture via QA inheritance.

---

## Exercise 04 as gallery submission

Exercise 04 is the curriculum's inflection point. The builder is working on their own system —
there is no pre-planted answer. The expected outcome is not a specific cert but a valid one.

**Gallery submission protocol** (to be formalised):
1. Builder produces a PASS cert using the translation template
2. Builder runs `--self-test` (or `--cert`) and records the output
3. Builder submits the JSON cert + a one-paragraph description to `GALLERY/`
4. Cert is verified by running the validator against it — no human review needed

This turns Exercise 04 into:

> **learn → practice → build → submit → become a reference for others**

---

## Dependency chain

```
01 (encoding) → 02 (obstruction) → 03 (minimality) → 04 (independence) → 05 (composition)
```

Each exercise assumes the previous is complete. Do not assign 03 before 02 is understood,
because EC10 (minimality) only makes sense once EC11 (obstruction) is clear — both are about
the difference between *possible* and *provably correct*.

---

## Files

| Exercise | File | Status |
|----------|------|--------|
| 01 | `EXERCISE_01_THERMOSTAT.md` | complete |
| 02 | `EXERCISE_02_RC_CIRCUIT.md` | complete |
| 03 | `EXERCISE_03_RLC_FEEDBACK.md` | complete |
| 04 | `EXERCISE_04_YOUR_DOMAIN.md` | complete |
| 05 | `EXERCISE_05_COMPOSITION.md` | pending |
