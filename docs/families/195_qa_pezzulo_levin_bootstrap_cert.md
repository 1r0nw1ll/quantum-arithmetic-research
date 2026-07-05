# Family [195] QA_PEZZULO_LEVIN_BOOTSTRAP_CERT.v1

## One-line summary

Pezzulo & Levin's "biological route from chemistry to cognition and creativity" mapped to QA architecture levels across 8 stages (matching the paper's own Figure 1 nested-problem-space hierarchy exactly, including "Physiological"). Intelligence ratchet = Pisano fixed point π(9)=24 [192]. Five design principles (autonomy, self-assembly, rebuilding, constraints, signaling) map to QA axioms and operations.

## Mathematical content

### 8-stage pipeline ↔ QA levels

Corrected 2026-07-04 (see Verification Note): matches the paper's own
Figure 1 nested-problem-space hierarchy — Metabolic → Physiological →
Transcriptional → Morphospace → 3D(Behavioral) → Linguistic — bracketed
by "Chemistry/Physics" (precedes the figure, supported by the paper's
own prose) and "Creativity" (the paper's title's endpoint).

| Stage | Pezzulo-Levin | QA Level | Bateson Level |
|-------|--------------|----------|---------------|
| 1 | Chemistry/Physics | A1 (no-zero discrete arithmetic) | L₀ |
| 2 | Metabolic | Single-step QA dynamics T(b,e) | L₀ |
| 3 | Physiological | Orbit-attractor convergence (T1 path time) | L₀ to L₁ |
| 4 | Transcriptional | Orbit classification via v₃(f) | L₁ |
| 5 | Anatomical (morphospace) | Orbit structure + E8 alignment | L₁ to L₂ₐ |
| 6 | Behavioral (3D) | Observer projection (Theorem NT) | L₂ₐ |
| 7 | Abstract/cultural (linguistic) | Multi-modulus operation, L₂ | L₂ₐ and L₂ᵦ |
| 8 | Creativity | L₃ modulus change via π(9)=24 | L₃ |

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

**7-stage pipeline table did NOT match the paper's own explicit
hierarchy, and this has been fixed**: the paper's Figure 1 ("Living
beings... navigate across nested problem spaces") names exactly six
problem spaces, in order: **Metabolic → Physiological → Transcriptional
→ Morphospace → 3D (Behavioral) → Linguistic**. The cert's original
7-stage table omitted "Physiological" entirely. Fixed by inserting it as
stage 3, between Metabolic and Transcriptional, matching Figure 1
exactly: "Physiological" (bioelectric/ion-channel homeostatic set-point
maintenance) is mapped to **orbit-attractor convergence via the T1 path-
time axiom** — a perturbed trajectory returning to its periodic orbit,
which is a genuinely distinct QA concept from both "Metabolic" (a single
T(b,e) step) and "Transcriptional" (full orbit-family classification via
v₃(f)), and reuses the same T1 concept already present in this cert's own
"Rebuilding/regeneration" design-principle mapping (biologically
consistent: physiological homeostasis and morphological regeneration are
both "return to attractor after perturbation"). All 8 stages, the
validator (`EXPECTED_STAGE_COUNT=8`, extended `BATESON_ORDER` with
`"L_0 to L_1": 0.5`), both fixtures, and this doc were updated together;
self-test reruns clean. "Chemistry/Physics" (stage 1) and "Creativity"
(stage 8) remain as bookends outside Figure 1's explicit boxes — both
are directly supported by the paper's own prose (embryogenesis starting
from "chemistry and physics"; the paper's title ending at "creativity")
so are kept, clearly labeled as such.

**Pisano fixed point independently reproduced**: π(9)=24 and π(24)=24
computed directly (stdlib Fibonacci-mod-m), confirming the genuine
fixed-point claim underlying the "intelligence ratchet."
