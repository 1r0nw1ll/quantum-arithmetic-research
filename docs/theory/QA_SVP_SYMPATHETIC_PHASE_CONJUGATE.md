<!-- PRIMARY-SOURCE-EXEMPT: reason=internal synthesis mapping Keely/SVP sympathetic vibration to the QA phase-conjugate operator; Keely primary sources are cited below (svpwiki, byte-verified in the prior Keely/SVP provenance audit) and the operator's sources in companion certs [518]/[519]. -->

# QA Phase-Conjugate Resonance as a Formalization of Sympathetic Vibration

**Status**: v1 (evidence-first) | **Date**: 2026-07-09 | **Authors**: Will Dale + Claude

This maps **one specific, falsifiable** claim of John Keely's Sympathetic Vibratory
Physics (SVP) onto the QA phase-conjugate operator we have built and certified. It
is written **after** the supporting demonstration (`qa_sympathetic_resonance.py`),
not before it. It deliberately does **not** endorse SVP wholesale, and it is
explicit about what is formalized, what is interpretive, and what is open.

## The formalized claim (with evidence)

> **Sympathetic vibration = phase-conjugate resonance.** Two systems tuned to the
> same "chord" couple and exchange energy *regardless of their instantaneous phase*;
> systems of different chords do not couple (sympathetic **selectivity**).

In QA: a tuning is a phase pattern `p ∈ {1,…,m}^N`; sympathetic coupling is the
phase-conjugate lock `C_symp(i,j) = max_ψ overlap(qa_add(p_i, ψ), p_j)/N`.

Evidence (`qa_sympathetic_resonance.py`, m=24, N=64):
- **Same chord, any phase offset → couples**: `C_symp = 1.000`.
- **Different chord → does not couple**: `C_symp = 0.093` (the correct
  max-over-shifts false-coupling floor; *not* 1/m — the null 95th percentile is
  0.125). Selectivity gap **+0.891**, cleanly separable.
- **Naive (no-lock) coupling cannot see same-chord-at-different-phase** (`C_naive
  = 0.060 ≈` chance). This is the crux: naive similarity is *not* sympathetic
  vibration — the phase-conjugate **lock** is what makes co-tuned systems couple
  across phase.

This is the **same-medium specificity** already certified four independent ways —
[518] (exact distortion correction), [520] (EEG artifact-lock), [521]
(morphogenetic regeneration), and the channel equalizer. SVP's "sympathetic
association between co-tuned bodies" and phase conjugation's "a system responds
only to its matched medium" are the same mathematical structure.

## How it ties the scalar-EM cluster together

| layer | certs / artifacts | role |
|-------|-------------------|------|
| EM substrate | [510]–[514] QA-Maxwell derivation | rigorous finite-cochain Maxwell in QA (guarding, not asserting, the scalar/longitudinal question) |
| phase-conjugate signature | [155] Bearden pumped-PC-mirror (QCI opposite-sign, weak) | the *emergent* signature |
| explicit operator | [518] FWM conjugator + distortion-correction theorem | the exact operator |
| memory / applications | [519] memory, [520] EEG, [521] morphogenesis, equalizer | same-medium specificity, four ways |
| **SVP phenomenology** | **this doc + `qa_sympathetic_resonance.py`** | sympathetic vibration = phase-conjugate resonance |

Phase conjugation is a real solution structure of the QA-Maxwell substrate; SVP
sympathetic vibration is the phenomenology that the phase-conjugate lock
formalizes. That is the through-line from rigorous EM to Keely.

## Scope: formalized vs interpretive vs open

**Formalized (evidenced here):** selective, phase-invariant sympathetic coupling —
co-tuned systems couple across phase, different tunings do not — as the
phase-conjugate lock / same-medium specificity.

**Interpretive (not claimed as validated):** SVP's broader ontology — aetheric
media, over-unity / free-energy, "focalization of will," etc. None of that follows
from, or is asserted by, this mapping.

**Open (deferred, not overclaimed):** a QA formalization of Keely's **neutral
center**. The obvious candidate (the additive-identity pattern) does **not** serve
— under a global phase scan it degenerates to a constant and merely measures a
chord's modal phase, not coupling (found and removed during review). A correct
neutral-center formalization is future work.

## Provenance and caveats

- Keely primary quotations used across this project were byte-verified against
  svpwiki.com in the prior Keely/SVP provenance audit (see
  `memory/project_keely_svp_vibes_provenance_audit.md`); "Vibes" (Dale Pond's AI
  tool) is **not** peer review and is not treated as theorem-grade here.
- The mapping is a *structural* correspondence backed by a falsifiable
  demonstration; it is a step toward the terminal-goal SVP formalization, not a
  claim that SVP as a whole is validated.

## Primary Sources

- Keely, J.E.W. — laws of sympathetic vibration; archived at svpwiki.com
  (byte-verified, prior audit). Companion: cert [155], [201], [184]–[188].
- Soffer, B.H. et al. (1986). *Opt. Lett.* 11(2):118-120. DOI 10.1364/OL.11.000118
- Yariv, A. (1978). *IEEE J. Quantum Electron.* 14(9):650-660. DOI 10.1109/JQE.1978.1069870
- Certs [510]–[514] (QA-Maxwell), [518]/[519]/[520]/[521] (phase-conjugation).

## Artifacts

- `qa_sympathetic_resonance.py` — the falsifiable demonstration.
- `docs/theory/QA_PHASE_CONJUGATE_APPLICABILITY.md` — the operating boundary.
