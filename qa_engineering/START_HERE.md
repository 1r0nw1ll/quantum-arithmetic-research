# Start Here — QA Engineering Tier 4 On-boarding Beta

This is the entry path. Follow it in order.

---

## 1. Read the foundations doc (20 minutes)

`01_foundations/FOUNDATIONS_OF_ENGINEERING_AND_APPLIED_MATH_FOR_QA.md`

Five sections: state, dynamics, control, invariants, computation.
Each explains a core idea in plain language and shows what it means in QA.
Each ends with a pointer to the exact files and certs where the formal version lives.

If you want the short version first: read the mental model block at the top of that doc.

---

## 2. Complete Exercise 01 — Thermostat (15 minutes)

`EXERCISES/EXERCISE_01_THERMOSTAT.md`

Three blanks. You compute orbit classification by hand, name a generator, and verify the
arithmetic obstruction check. You end with a PASS cert. This is your first completed cert.

---

## 3. Complete Exercise 02 — RC Circuit (20 minutes)

`EXERCISES/EXERCISE_02_RC_CIRCUIT.md`

You are given an encoding that looks valid but fails EC11. You compute the obstruction,
see the validator reject the cert, re-encode, and get a PASS.

The key idea: orbit family classification (EC5) and arithmetic reachability (EC11) are
independent checks. A state can be correctly classified and still be unreachable.

---

## 4. Complete Exercise 03 — RLC Circuit (25 minutes)

`EXERCISES/EXERCISE_03_RLC_FEEDBACK.md`

You prove that k=2 is the *shortest possible* path, not just *a* valid path.
This introduces the minimality witness (EC10) and the distinction between:

- reachable (EC9)
- provably shortest (EC10)
- arithmetically admissible (EC11)

---

## 5. Map your own system — Exercise 04 (your pace)

`EXERCISES/EXERCISE_04_YOUR_DOMAIN.md`

Choose any system you know well — a physical phenomenon, an instrument, a pattern you've
observed. Work through the full translation template. Run the validator until you get PASS.
Fill in the gallery checklist.

The template is your main tool: `06_classical_engineering_map/QA_SYSTEM_TRANSLATION_TEMPLATE.md`
When the validator rejects you: `FAILURES/`

---

## 6. Submit to the gallery

`GALLERY/`

Place your cert JSON and completed checklist in `GALLERY/`. Validator PASS is the only
requirement. Your cert becomes a reference for other builders in your domain.

---

## Reference material (keep open while working)

- `05_reference/QUICK_REFERENCE.md` — orbit table, inert primes, hash formula
- `FAILURES/` — look up any validator error by fail type
- `03_applied_domains/SPRING_MASS_WORKED_EXAMPLE.md` — a complete end-to-end walkthrough

---

## The complete picture

If you want to understand the formal foundations behind any of this:

- Axioms and generator algebra: `01_foundations/QA_AXIOMS.md`
- Proved theorems: `02_control_theory/CONTROL_THEOREMS.md`
- Classical engineering equivalence table: `06_classical_engineering_map/CLASSICAL_TO_QA_MAP.md`
- The cert family this on-boarding pack is built on: cert [121] in
  `qa_alphageometry_ptolemy/qa_engineering_core_cert/`

---

*Exercise 05 (composition and inheritance) is in development. It will cover how individual
system certs connect to the broader QA cert architecture via [107]→[121] inheritance chains.
Complete Exercises 01–04 and submit a gallery cert first.*
