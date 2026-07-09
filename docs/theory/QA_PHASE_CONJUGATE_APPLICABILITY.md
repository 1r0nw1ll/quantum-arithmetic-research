<!-- PRIMARY-SOURCE-EXEMPT: reason=internal methods/applicability characterization of the QA phase-conjugate memory; the operator's sources (Soffer 1986, Owechko 1987, Hopfield 1982) are cited in companion certs [518]/[519]/[520]/[521]. -->

# QA Phase-Conjugate Memory — Applicability Boundary (corrected)

**Status**: Complete | **Date**: 2026-07-09 | **Authors**: Will Dale + Claude

This is the corrected characterization of *when* the QA phase-conjugate associative
memory (certs [518]/[519]) and its phase-lock distortion-tolerance apply. It
**supersedes** the earlier claim (commit 2482f60f) that the climate ENSO failure
was caused by low dimensionality — that was **wrong** and is retracted here.

## The prompt

Cert [520] (EEG brain-state recall) and [521] (morphogenetic memory) show strong
recall and a strong phase-lock advantage under a systemic shift. A third domain,
climate ENSO regime recall (`qa_climate_enso_recall.py`, real 5-channel NOAA data),
**failed**: recall only 0.66 (chance 0.45) and the phase-lock advantage vanished.
What actually causes the failure?

## Controlled investigation (`qa_phase_conjugate_dimension_sweep.py`)

Each hypothesis tested by holding everything else fixed:

| hypothesis | test | result |
|---|---|---|
| **low dimension** | vary pattern dim N = 4…256 | **falsified** — recall + phase-lock perfect at every N, incl. N=4 |
| **poor separability** | overlapping classes (distinct_frac 0.1…1.0) | **falsified** — works even at heavy overlap |
| **crowding** | 12…600 stored patterns | **falsified** — 600 patterns still phase-lock 1.000 |
| **label-metric misalignment** | class defined by `rel` of N dims, rest noise | **CONFIRMED** — at rel=1, recall 0.74, phase-lock ≈ naive (reproduces climate) |
| **continuum-threshold classes** | discrete attractors vs thresholds on a continuum | **CONFIRMED** — discrete phase-lock 1.00 vs continuum 0.15 |

## The two real drivers

**1. Clean recall requires label-metric alignment.** The label-relevant signal must
dominate the stored pattern. ENSO is defined by a threshold on **one** channel
(ONI); the other four (NAO/AO/PDO/AMO) are label-irrelevant, so the nearest pattern
in full 5-D phase space often has the wrong ENSO label. Confirmed on real data:
**weighting ONI heavily recovers recall 0.664 → 0.952** (`qa_climate_enso_recall.py`
`[3]`), while the synthetic `rel_dims=1` reproduces the failure exactly.

**2. Phase-lock distortion-tolerance requires *discrete attractor* classes.** With
classes that are **thresholds on a continuum** (ENSO = bins of continuous ONI), the
class is encoded along a global-phase-like axis — the very degree of freedom
phase-lock compensates. Compensating the systemic shift therefore also erases the
class signal; phase-lock cannot distinguish "systemic shift" from "different class."
Synthetic: **discrete classes give phase-lock 1.00, continuum classes 0.15** (same
N, K, crowding). ONI-weighting does **not** fix this (climate phase-lock stays ≈0.31)
because the continuum structure is untouched.

## The reusable rule

Apply the phase-conjugate memory + phase-lock where:
- the class label **aligns with the pattern metric** (label-relevant signal is a
  substantial fraction of the pattern), **and**
- classes are **discrete, well-separated attractors**, not thresholds on a continuum.

Both hold for EEG brain-states ([520]) and morphological body plans ([521]) — discrete,
high-signal patterns — so phase-lock works there. Climate ENSO violates both, which is
why it fails; it is a **misaligned / continuum-structured target**, not a fundamental
limit of the method (the "verify target alignment before ceiling" principle).

## Artifacts

- `qa_phase_conjugate_dimension_sweep.py` — the controlled sweeps (recall,
  generalization, crowding, label-relevance, class-structure).
- `qa_climate_enso_recall.py` — real-data demonstration + the ONI-weighting confirmation.
- Related: certs [518], [519], [520], [521].
