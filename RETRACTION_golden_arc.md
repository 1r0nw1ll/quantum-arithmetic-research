# ERRATA & CORRECTION — the "golden structure of QA" arc

Two errors, in opposite directions. This document records both and the corrected position.
Commits `ce1c1a21`..`0fcf00c8`, corrected by `qa_golden_exact.py`.

## Correction 2 (2026-07-12) — the retraction itself over-reached

The retraction below (Correction 1) claimed the golden arc was **continuous** mathematics
"because φ is irrational, so it cannot be a discrete QA object." **That is a category error:
it conflates IRRATIONAL with INEXACT.**

φ = (1+√5)/2 is the regular-pentagon ratio — **exactly straightedge-and-compass
constructible** — and ℤ[φ] is **exact integer arithmetic** on pairs (a,b)=a+bφ (add
componentwise; multiply via φ²=φ+1; ×φ is (a,b)→(b,a+b)). Even the **ordering** of ℤ[φ] is
exactly decidable in integers, with √5 never evaluated (sign of a+bφ = compare (2a+b)² to 5b²).

`qa_golden_exact.py` rebuilds the cut-and-project Fibonacci quasicrystal in **pure exact integer
arithmetic, zero floats** (linter-clean with **no** `RT1_OBSERVER_FILE` exemption): 261 points,
exactly two tiles (1 and φ), tile ratio verified = φ by the integer identity φ·(1,0)=(0,1). The
golden structure is therefore an **exactly discrete-constructible object** — consistent with
Volk's / QA's actual exact-geometry ethos (exact integer m:n winding ratios, exact geometric
construction; the whitepaper maps Fibonacci winding into (b,e,d,a)).

**So the arc's real error was NARROWER than the retraction claimed:** it computed the golden
structure with drift-prone **floats** (`PHI = (1+math.sqrt(5))/2`, `np.cos`, …) and hid them
behind `RT1_OBSERVER_FILE`, instead of exact ℤ[φ]. That is a fixable *implementation* flaw, not
a proof that the object is "continuous / not QA." The mathematics in the arc was correct; the
representation was sloppy.

## Corrected net position

- The golden structure (Fibonacci, ℤ[φ], quasicrystal, cut-and-project, E8/icosian) is an
  **exact, discrete, constructible** object. NOT retracted as "continuous."
- The arc's genuine flaw: **float implementation + RT1-exemption** where exact ℤ[φ] was
  available and correct. Remediated by `qa_golden_exact.py` (the pattern others should follow).
- **Phase L still holds, narrowly:** reducing **mod m** destroys the structure (Fibonacci mod m
  is Pisano-periodic). This is a fact about the mod-m *reduction*, NOT about φ's nature. The
  unreduced exact ℤ[φ] object retains the structure. So the honest distinction is:
  QA-as-exact-geometry (Volk) *has* the golden structure; QA-as-strict-mod-m-reduction loses it.
- The overclaim that remains retracted: presenting **float diffraction readouts as QA's
  structure** without doing the exact construction, and using the observer-exemption to do it.
- `qa_nonunit_experiment.py` is separately flawed (it used a float regression — the operation
  QA forbids — and classified by ring-unit-ness, which is beside QA's exactness point).
- Prior empirical work (EEG / seismic / climate) is independent of this arc.

## The rule, corrected

`RT1_OBSERVER_FILE` is for reading out a genuine QA state, not for making a continuous *float*
the content. But the fix for a golden/φ computation is **not** to declare it "not QA" — it is
to do it in **exact ℤ[φ] integer arithmetic** (as `qa_golden_exact.py` does). Irrational ≠
inexact: an exactly-constructible algebraic quantity is a legitimate discrete QA object.

---

## Correction 1 (2026-07-12, superseded above) — original retraction

[Kept for provenance. Its core claim ("golden structure is continuous, therefore not QA") is
**withdrawn** by Correction 2 above; irrational ≠ inexact.]

Audit: of the 20 arc files, 15 are centrally φ/√5/real and all carry `RT1_OBSERVER_FILE`. I
concluded the golden structure was continuous and not QA. That conclusion was wrong — the
structure is exactly ℤ[φ]-constructible (see Correction 2). What was correct in the original
retraction: the arc *implemented* it in floats behind the observer-exemption, and mod-m
reduction destroys the structure (Phase L).
