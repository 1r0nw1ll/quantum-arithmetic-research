# Exercise 04 — Your Domain

**Difficulty**: independent
**Time**: as long as it takes
**Prerequisite**: complete Exercises 01–03
**New skill**: mapping a system from your own domain end-to-end, without scaffolding
**Outcome**: a gallery-quality cert you produced yourself

---

## What this exercise is

Exercises 01–03 gave you a system and told you where the traps were.

This one does not.

You choose the system. You choose the encoding. You find the traps. You produce the cert.

The only tools you need are already in this on-boarding pack:

- `06_classical_engineering_map/QA_SYSTEM_TRANSLATION_TEMPLATE.md` — your primary tool
- `FAILURES/` — when the validator rejects you
- `03_applied_domains/SPRING_MASS_WORKED_EXAMPLE.md` — if you need to see a complete example again
- `05_reference/QUICK_REFERENCE.md` — orbit table, inert primes, hash formula

---

## Your task

**1. Choose a system from your own domain.**

It must have three recognisable operating regimes. If you cannot identify three distinct regimes
in your system, the system may be too simple (only two states) or too complex (model it at a
higher level of abstraction first).

Some starting points by background:

| Background | System candidates |
|-----------|-------------------|
| Mechanical | pendulum, gear train, hydraulic actuator |
| Electrical | motor drive, switching regulator, PLL |
| Acoustic / SVP | Chladni plate mode transition, resonator tuning, vibrational cascade |
| Seismic | fault nucleation cycle, wave propagation regime |
| Biological | neuron firing cycle, circadian rhythm, cell cycle |
| Thermal | heat exchanger (cold/heating/equilibrated) |
| Signal processing | filter state (transient/ringing/settled) |
| Finance | market regime (bear/transitional/bull) |
| Neural network | training phase (random/descending/converged) |
| Computer / network | connection state (closed/handshake/established) |

**2. Work through the full translation template.**

Every section. Do not skip Section 6 (arithmetic obstruction check).

**3. Run the validator.**

When it fails (and it may), use `FAILURES/` to diagnose and fix.

**4. When you have a PASS, complete the gallery submission checklist below.**

---

## Gallery submission checklist

When the validator returns PASS, fill in this checklist. Keep it with your cert.

```
System name: _________________________________________________

Domain: _____________________________________________________

Governing equation or model (brief):
_______________________________________________________________

Three states:
  Singularity: ________________  (b=___, e=___)
  Satellite:   ________________  (b=___, e=___)
  Cosmos:      ________________  (b=___, e=___)

Two (or more) generators:
  ______________________  →  ______________________
  ______________________  →  ______________________

Target state: ________________
target_r = ___ · ___ = ___
v_p(target_r) = ___  →  obstructed: ___

Why this encoding is valid (one sentence):
_______________________________________________________________

Validator output (paste the PASS line):
_______________________________________________________________
```

**To submit to the gallery**: place your cert JSON and this completed checklist in `GALLERY/`.
The cert will be verified by running the validator — no human review required.

---

## What makes a gallery-quality cert

A cert is gallery quality if:

1. The system description is recognisable to someone in your domain
2. The three state labels use domain vocabulary, not generic QA labels
3. The generators are named after real physical or operational actions
4. The orbit classification arithmetic is correct (check with `QUICK_REFERENCE.md`)
5. The obstruction check is present and correct
6. The validator returns PASS

A cert that passes the validator is formally correct. A cert that also has good domain labels and
a clear system description is a useful reference for other builders in the same domain.

---

## What to do if you get stuck

**Cannot identify three distinct regimes?**
Start with the most obvious distinction in your system: on/off, active/inactive, stable/unstable.
Let those be singularity and cosmos. The satellite state is the transition between them — it
is often the least obvious but most physically interesting regime.

**Orbit classification keeps failing?**
Use the verification table in `FAILURES/FAIL_ORBIT_CLASSIFICATION.md`. Common working encodings
for mod-9: singularity=(9,9), satellite=(3,6), cosmos=(1,2).

**EC11 obstruction keeps appearing?**
Your target (b,e) gives target_r = b·e with v₃(b·e) = 1. Change one of b or e so that their
product is not divisible by 3, or is divisible by 3 more than once (v₃ ≥ 2 → satellite, not
target). Allowed cosmos target_r values for mod 9: {1, 2, 4, 5, 7, 8} — these have v₃ = 0.

**Minimality witness (EC10)?**
Only required if you include `optimization_claim` in your cert. If you omit
`optimization_claim`, EC10 is not checked. Add it only if you want to claim your path is
shortest.

---

## Note on difficulty

Exercise 04 is intentionally open-ended. The validator will tell you exactly what is wrong.
Your job is to fix it and re-run. There is no time pressure and no single correct answer —
any system that produces a PASS cert is a valid completion of this exercise.

The exercise is complete when the validator returns PASS and the gallery checklist is filled in.
