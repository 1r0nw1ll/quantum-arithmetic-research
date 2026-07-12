<!-- PRIMARY-SOURCE-EXEMPT: reason=internal robustness analysis of the QA phase-conjugate operator cluster; the underlying certs ([522] seismic EGF, sympathetic resonance) and their data are cited below. -->
# Phase-Conjugate Cluster: Modulus Robustness and the mod-24 Correction

Evidence-first robustness note on the QA phase-conjugate operators
(`qa_seismic_egf_specificity.py` = cert [522], `qa_sympathetic_resonance.py` = SVP,
`qa_acoustic_delaylock.py` = real acoustic). All quantize a phase to **mod-24**. This
records (a) that the certified results do **not** depend on that choice, and (b) that
"24" is a resolution knob, not an orbit-derived modulus — correcting a conflation.

## The mod-24 conflation

The operators map a phase to `{1..24}` "because QA is mod 24." But **24 is the Pisano
*period* of the mod-9 golden orbit** (the 24-cycle Cosmos) — an *iteration count*, not a
phase-quantization modulus. Quantizing a cross-spectral phase into 24 bins on that basis
borrows the number without its meaning. The honest test is whether the result depends on
it. It does not.

## Evidence (all on the operators' own data)

| operator | data | finding across M and raw phase |
|---|---|---|
| [522] seismic EGF (`qa_seismic_egf_robustness.py`) | real, 13 stations | C_match ≫ C_mis for **raw phase and every sampled M** {6,9,12,24,48}; spread 0.21–0.27. Robust; mid-M marginally best. |
| acoustic (`qa_acoustic_delaylock.py`) | real notes | AUC **flat 0.77–0.79** for M = 4…96; coarser marginally better. |
| SVP sympathetic (`qa_sympathetic_modulus_sweep.py`) | synthetic | same-chord couples at 1.000 for **every M**; separation grows with M (0.78→0.95) → mod-24 **suboptimal**. |

In none of the three is 24 the justified or optimal choice, and in all three the **core
result survives every M and raw phase**.

## Conclusions

1. **The certified phase-conjugate specificity is robust to the quantization modulus.**
   It is not an artifact of mod-24 — which *strengthens* the certs against the natural
   skeptic objection "you tuned the modulus."
2. **"mod-24" should be documented as a resolution/denoising parameter, not an
   orbit-derived QA modulus.** It is the Pisano period of the mod-9 orbit, and its
   optimum is task-dependent (mid-M for real noisy seismic, large-M for clean synthetic,
   flat for acoustic). Presenting it as a meaningful modulus is a borrowed-number claim a
   skeptic would (correctly) flag.

## Scope

This is a robustness/framing note; it does **not** modify the certified validators or
their scope-notes (Gate 0). Recommendation for the cluster: state the phase-quantization
modulus as a resolution parameter and cite this robustness result, so the certs rest on
the phase specificity itself (which survives raw phase) rather than on the number 24.
