# Family [195] QA_PEZZULO_LEVIN_BOOTSTRAP_CERT.v1

## One-line summary

Pezzulo & Levin's 7-stage "biological route from chemistry to cognition and creativity" mapped to QA architecture levels. Intelligence ratchet = Pisano fixed point π(9)=24 [192]. Five design principles (autonomy, self-assembly, rebuilding, constraints, signaling) map to QA axioms and operations.

## Mathematical content

### 7-stage pipeline ↔ QA levels

| Stage | Pezzulo-Levin | QA Level |
|-------|--------------|----------|
| 1 | Chemistry/Physics | A1 (no-zero discrete arithmetic) |
| 2 | Metabolic space | Single-step QA dynamics |
| 3 | Transcriptional space | Orbit classification via v₃(f) |
| 4 | Anatomical morphospace | Orbit structure + E8 alignment |
| 5 | Behavioral space | Observer projection (Theorem NT) |
| 6 | Abstract/cultural | Multi-modulus operation, L₂ |
| 7 | Creativity | L₃ modulus change via π(9)=24 |

### Intelligence ratchet

π(9) = 24: the Pisano period operator applied to mod-9 produces mod-24. The arithmetic generates the conditions for its own enhancement. This is L₃ (modulus-changing) bootstrapping — certified in [192] as the minimum non-trivial Pisano fixed point.

### Five design principles ↔ QA operations

| Principle | QA Operation |
|-----------|-------------|
| Autonomy | A1 boundary ({1,...,N}, no zero) |
| Self-assembly | Orbit emergence from f(b,e) classification |
| Continuous rebuilding | T1 path time (constructive, not storage) |
| Embodied constraints | S1 + S2 (integer arithmetic constraints) |
| Pervasive signaling | Resonance coupling (tuple inner products) |

## Dependencies

- [191] QA_BATESON_LEARNING_LEVELS_CERT.v1
- [192] QA_DUAL_EXTREMALITY_24_CERT.v1

## Sources

- Pezzulo & Levin, "Bootstrapping Life-Inspired Machine Intelligence" (arXiv:2602.08079, 2026)

## Verification Note (2026-07-04)

Fetched and read the actual paper directly (arXiv:2602.08079, Pezzulo &
Levin, "Bootstrapping Life-Inspired Machine Intelligence: The Biological
Route from Chemistry to Cognition and Creativity" — title matches this
cert's one-line summary almost verbatim). Confirmed real.

**Five design principles table matches exactly**: the paper's abstract
lists "multiscale autonomy, growth through self-assemblage of active
components, continuous reconstruction of capabilities, exploitation of
physical and embodied constraints, and pervasive signaling enabling
self-organization and top-down control from goals" — matching this
cert's Autonomy/Self-assembly/Continuous rebuilding/Embodied constraints/
Pervasive signaling table in the same order.

**7-stage pipeline table does NOT match the paper's own explicit
hierarchy**: the paper's Figure 1 ("Living beings... navigate across
nested problem spaces") names exactly six problem spaces, in order:
**Metabolic → Physiological → Transcriptional → Morphospace → 3D
(Behavioral) → Linguistic**. This cert's 7-stage table omits
"Physiological" entirely and instead brackets the list with "Chemistry/
Physics" (stage 1) and "Creativity" (stage 7) — neither of which is a
named box in Figure 1 (though both are supported by surrounding prose:
the paper does describe embryogenesis starting from "chemistry and
physics," and frames the whole arc as leading toward "creativity" in its
title). This isn't a fabrication, but the 7-stage table should be
understood as **this cert's own QA-motivated compression of the paper's
narrative**, not a literal citation of Figure 1's stage list — and it
specifically drops "Physiological," a stage the paper does name
explicitly. Not fixed in the validator (would require a QA-level design
decision for the missing stage, outside this audit's scope) — flagged
here so the doc doesn't overstate precision it doesn't have.

**Pisano fixed point independently reproduced**: π(9)=24 and π(24)=24
computed directly (stdlib Fibonacci-mod-m), confirming the genuine
fixed-point claim underlying the "intelligence ratchet."
