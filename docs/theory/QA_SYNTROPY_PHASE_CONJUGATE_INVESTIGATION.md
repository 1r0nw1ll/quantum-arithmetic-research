# QA Syntropy / Phase-Conjugation Investigation — Companion Record for Cert [518]

**Status**: Complete | **Date**: 2026-07-08 | **Authors**: Will Dale + Claude

This is the empirical companion record that cert **[518]** (QA FWM Phase
Conjugate) and cert **[155]** (QA Bearden Phase Conjugate) reference. It
documents the full investigation arc — including the negative results — so the
certs do not cite evidence that lives only in ephemeral scratch. Reproduction
scripts: `experiments/syntropy_phase_conjugate/`.

## Question

Fantappiè's *syntropy* (advanced/time-reversed wave solutions — Wheeler–Feynman
absorber theory, Cramer's transactional interpretation) is physically realized by
**optical phase conjugation**, whose **distortion-correction theorem** (Yariv
1978; Zel'dovich–Pilipetsky–Shkunov 1985; Agarwal–Friberg) states that a
phase-conjugated wave returned through the *same* distorting medium exactly undoes
the distortion. Does QA implement this — **emergently**, from its self-organizing
dynamics, or only via an **explicit** conjugate-generating operator?

Answer, in one line: **emergently, no; explicitly, exactly.**

## Part A — The Pesin anchor (positive control)

Largest Lyapunov exponent of the Lorenz system by the Benettin method:
**λ₁ = 0.9015 nats/time** vs the literature value 0.906 — reproduced essentially
exactly. Pesin's identity `h_KS = Σ λ⁺` is the KS-entropy bridge the syntropy
framing invokes; the anchor is solid. (`syntropic_projector_probe.py`,
`syntropic_projector_dynamics.py`.)

## Part B — Emergent QA dynamics do NOT implement syntropic projection (4 nulls)

| Test | Operationalization | Result |
|------|--------------------|--------|
| Static orbit-family fold | classify delay-embedded points by `orbit_family_s9` (Eisenstein norm), excess collapse vs size-matched random partition | **Null / anti-syntropic**: excess −0.127 bits, z = +1.23 (wrong direction), p = 0.88; the excess is zero-or-negative across delay τ ∈ {1,4,8,16,32} and both binning modes (the sole syntropic-sign point, τ=16, is not significant) |
| QA additive-recurrence match | under τ=1 embed, does the trajectory obey `bin(x_{t+2}) = (bin(x_t)+bin(x_{t+1})−1) mod m + 1`, vs time-shuffle null | **Autocorrelation artifact**: Lorenz/Rössler z≈+5.3, periodic +7.1, logistic-map & white-noise null — but **AR(1) red noise (zero positive Lyapunov) z=+29.5**, ~5× any chaos → the effect tracks smoothness, not chaos or QA structure |
| Resonance-flow concentration | canonical einsum `⟨T_t,T_{t+1}⟩` resonance entropy vs shuffle | **Refuted**: resonance stream is *higher* entropy than shuffle (z≈+100), the anti-syntropic direction — detects autocorrelation with the wrong sign |
| Order out of noise | steady-state E8 alignment / harmonic index of the driven `QASystem` across sources | **Real null**: E8 ≈ 0.82–0.83 for coherent/Lorenz/white/free-run (and red noise ≈ 0.826 in the initial fixed-injection run), **flat across injection strength 0.1→4.0** (40×) — intrinsic geometric order dominates; the drive cannot move it |

There is one *weak* positive: in a signal-buried-in-noise coherence-extraction
test with a leakage-free cross-validated distributed decode, the self-organizing
coupling recovers coherence beyond pure injection — `COUPL_ON > COUPL_OFF` at
every noise level (0.088 vs 0.055, p=0.037 at σ=2), growing with noise. Part C
shows this is **generic, medium-agnostic spatial denoising**, not phase
conjugation. (`qa_syntropy_extraction.py`, `qa_syntropy_order.py`.)

## Part C — The decisive test: same-medium specificity (emergent) fails

The distortion-correction theorem's *fingerprint* is same-medium specificity:
recovery works only when the return path matches the distortion path. Adapting the
`QASystem` coupling to a signal through a fixed FIR channel D (the "hologram"),
then reading reconstruction quality through the **same** D vs a statistically
identical but different D′ (`qa_phase_conjugate_distortion.py`):

- **No same-medium specificity**: SAME ≈ DIFF, SAME even *lower* at 3 of 4 noise
  levels (p > 0.75).
- **Adaptation hurts**: a fresh (un-adapted) coupling reconstructs *better* than
  the adapted one at every noise level (e.g. 0.164 vs 0.060 at σ=0.5). The
  opposite of holographic recording — the dissipative system + noise annealing
  freezes into an attractor and loses plasticity.

Conclusion: the emergent coupling does generic medium-agnostic denoising, not
phase-conjugate distortion correction.

## Part D — Convergence with cert [155]'s own certified record

Cert [155] (QA Bearden Phase Conjugate) certifies the *structural parallel*
(Bearden's pumped phase-conjugate mirror ↔ the QCI opposite-sign "global tightens
/ local scatters" gap), but its pre-registered empirical domains are weak-to-null:

- `bearden_injection` (domain 2): **WEAK** — partial_r ≈ −0.13, p ≈ 0.0006
- `bearden_stress_proxy`: **NULL** on one dataset; the large one significant but
  r ≈ −0.076 (tiny)
- `bearden_injection_domain3`: **NULL — sign flipped** (attack gap went positive,
  opposite domain 2)

Both the fresh tests and QA's own certs agree: QA carries a **real but weak**
phase-conjugate signature (r ≈ 0.07–0.15) that fails every strong, discriminating
test (same-medium specificity; consistent-sign cross-domain replication).

## Part E — The explicit operator succeeds exactly (cert [518])

The physics says phase conjugation is not emergent — it is generated by an
explicit nonlinear element (four-wave mixing: two pumps + signal → conjugate).
Supplying that operator explicitly (`qa_fwm_conjugator.py`, cert [518]):

```
fwm(pf,pb,s) = qa_mod(pf + pb − s)          # FWM phase-sum relation
```

with conjugate pumps `pb = qa_neg(pf)` yields `qa_neg(s)` exactly, and the
distortion-correction theorem holds **identically**: same-medium recovery fidelity
**1.000** vs **1/m = 0.042** chance for a wrong medium — verified exhaustively and
on a 72×72 image-recovery demo.

## Bottom line

- The syntropy/phase-conjugation *theory* is sound and correctly predicts a weak
  coherence signature in QA's emergent dynamics.
- That emergent signature is only a weak shadow — it fails the strong tests, and
  the "order out of noise" and same-medium specificity claims are genuinely null
  on the self-organizing dynamics.
- The **explicit** four-wave-mixing operator realizes the distortion-correction
  theorem exactly (cert [518]). Phase conjugation belongs in QA as a *constructed
  primitive*, not an emergent property.

## Primary Sources

- Hellwarth, R.W. (1977). *J. Opt. Soc. Am.* 67(1):1-3. DOI 10.1364/JOSA.67.000001
- Yariv, A. (1978). *IEEE J. Quantum Electron.* 14(9):650-660. DOI 10.1109/JQE.1978.1069870
- Zel'dovich, Pilipetsky, Shkunov (1985). *Principles of Phase Conjugation.* Springer. ISBN 978-3-540-13458-4
- Agarwal & Friberg, "Scattering theory of distortion correction by phase conjugation," *J. Opt. Soc. Am.*
- Pesin, Ya.B. (1977). "Characteristic Lyapunov exponents and smooth ergodic theory." *Russ. Math. Surv.* 32(4):55-114.

## Related certs

- **[518]** QA FWM Phase Conjugate (the explicit operator; this record is its companion)
- **[155]** QA Bearden Phase Conjugate (the weak emergent QCI signature)
