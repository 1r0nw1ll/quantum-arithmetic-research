# [521] QA Phase-Conjugate Morphogenetic Memory Cert

**Family ID**: 521
**Slug**: `qa_morphogenetic_memory_cert_v1`
**Status**: Active
**Registered**: 2026-07-09

## Claim (demonstrated + mechanism-certified)

Michael Levin's bioelectric morphogenesis — the **target morphology stored as an
attractor**, with tissue navigating back to the correct body plan from damaged or
perturbed states (regeneration) — is realized as the cert **[519]** phase-conjugate
associative memory (on cert **[518]**'s exact conjugator). Body plans are 2D QA
phase fields; a **damaged (amputated) field regrows to the correct target** by
content-addressable recall.

**Key mechanism** (certified deterministically, no external data): under a
**systemic bioelectric shift** (`probe → qa_add(probe, φ)`, a global field
offset), naive nearest-overlap regeneration is fooled, while **phase-locked
recall** — scan the compensation phase maximising overlap, read the plan in that
frame (the [518] phase-conjugate mirror self-locking to the perturbing medium) —
restores the correct morphology. The validator proves this on two
spatially-different synthetic body plans (not global shifts of each other).

## Empirical record (reference implementation)

`qa_morphogenetic_memory.py` (m=24, 16×16 field, axiom-linter clean):

| Property | Result |
|----------|--------|
| Regeneration + correct-plan vs contiguous amputation | **1.000 up to 87.5%** amputated (12.5% tissue left) |
| Correlated morphologies (24 variants of 4 plans, 75% shared) | **1.000 up to 62.5%** amputation |
| No-chimera rate (regrown field is an exact stored plan) | **1.000** |
| Systemic perturbation (global shift): naive vs phase-locked | **0.000 vs 1.000** (φ≥3) |

- **Regeneration**: the memory regrows the exact correct body plan from a
  contiguous amputation of up to 87.5% of the field.
- **Which-morphology**: a damaged plan converges to its **own** stored plan.
- **No chimera**: the regenerated field is always a clean stored plan, never a
  mixture — Levin's "correct target, not a monster."
- **Systemic tolerance**: a global bioelectric offset destroys naive regeneration
  but phase-locked recall restores the correct morphology — the [518]
  distortion-correction property.

## Honest limits

- The high robustness reflects the **high-redundancy large-alphabet field**
  (near-orthogonal plans over 256 cells); it degrades as plans approach identical
  or amputation nears 100%.
- Amputating a **whole distinguishing region** *and* applying a systemic shift that
  maps the remainder onto another plan is genuinely ambiguous — an honest failure
  mode, not hidden.
- Distinct from `qa_brainca_morphogenesis_v3.py` (organizer-cell CA on a single
  target); this stores **multiple** body plans with content-addressable regeneration.

## Checks

| Check | Meaning |
|-------|---------|
| `REGEN` | an amputated body plan regenerates to the true plan |
| `WHICH_PLAN` | regeneration selects the true plan, not another |
| `NO_CHIMERA` | the regenerated field is an exact stored plan |
| `SYSTEMIC_PHASE_LOCK` | phase-locked recall regenerates through a systemic shift, all φ |
| `NAIVE_SYSTEMIC_FAILS` | naive regeneration is fooled by a systemic shift (control) |
| `OVERLAP_MATCH` | phase-conjugate overlap == exact match count |
| `A1_RANGE` | every state in `{1,...,m}` |
| `SRC` / `F` | mapping ref present; pass/fail fixtures behave as declared |

**Fixtures**: 2 PASS (mechanism + empirical) + 2 FAIL. **Self-test**:
deterministic, integer-only, pure stdlib.

## Primary Sources

- Levin, M. (2021). "Technological Approach to Mind Everywhere." *Front. Syst.
  Neurosci.* 15:768201. DOI 10.3389/fnsys.2021.768201
- Pezzulo, G. & Levin, M. (2015). "Re-membering the body: top-down control of
  regeneration." *Integr. Biol.* 7:1487-1517. DOI 10.1039/c5ib00221d
- Hopfield, J.J. (1982). *PNAS* 79(8):2554-2558. DOI 10.1073/pnas.79.8.2554
- Soffer, B.H. et al. (1986). *Opt. Lett.* 11(2):118-120. DOI 10.1364/OL.11.000118

## Companion

- Certs **[518]** (exact conjugator), **[519]** (associative memory), **[520]**
  (EEG brain-state recall) — the phase-conjugation cluster.
- Reference impl: `qa_morphogenetic_memory.py`.

**Author**: Will Dale + Claude, 2026-07-09.
