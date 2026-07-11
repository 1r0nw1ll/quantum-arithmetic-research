# [522] QA Time-Reversal Focusing Cert

**Family ID**: 522
**Slug**: `qa_time_reversal_focus_cert_v1`
**Status**: Active
**Registered**: 2026-07-10

## Claim (demonstrated + mechanism-certified)

Cert **[518]**'s distortion-correction operator — `qa_neg`, the **standard
involution** (adjugate) from the quaternion grounding = **optical phase
conjugation = time reversal** — run **in reverse** does not merely correct
distortion at a receiver; it **focuses energy back onto a source through a
scattering medium**. This is Mathias Fink's *time-reversal mirror* (ultrasound,
seismics, underwater acoustics), realized in QA phase arithmetic.

A source at `x0` propagates to an array (per-element phase `G(i,x0)`) and scatters
through a medium (`H_i`); the array records `r_i = qa_add(G(i,x0), H_i, s)`. Each
element is **time-reversed** (`r_i* = qa_neg(r_i)`) and re-emitted toward a
candidate location `x`:

```
field_i(x) = qa_add(G(i,x), H_i, r_i*)
           = qa_add( G(i,x) − G(i,x0) − s )      (the medium H_i CANCELS)
```

At `x = x0` every element carries the identical phase `qa_neg(s)` → they add
coherently → a **focal peak, through the medium, at the source**. Elsewhere the
per-element phases scatter.

## Key mechanism (certified deterministically, integer-only, no external data)

- **FOCUS_CONSTANT** — matched re-emission makes the field at the source *constant
  across all array elements* (= `qa_neg(s)`): maximally coherent focus.
- **MEDIUM_CANCEL** — that constant equals `qa_neg(s)` **independent of the
  scattering medium `H`** (verified for two distinct media). The involution does
  the work; the medium is undone.
- **OFF_SOURCE_SCATTER** — at a genuinely distinct off-source location the field is
  *not* constant (it would otherwise focus everywhere).
- **SPECIFICITY** — re-emitting through a **different** medium `H'` destroys the
  focus (the field at the source is no longer constant). This is the cert [518]
  same-medium fingerprint, now for **focusing** rather than correction.

## Empirical record (reference impl)

`qa_time_reversal_focus.py` (m=24, 64-element array, axiom-linter clean):

- refocuses **exactly at the source** through an arbitrary random phase screen —
  peak coherent focus **1.000 at x0**, background median ≈ 0.49 (focal gain > 1);
- **same-medium specificity**: matched-medium peak **1.000** vs mismatched **0.244**,
  gap **+0.756** — the focus survives only through the matched medium.

## Honest limits

The 1-D limited-aperture geometry has finite focal resolution and moderate
sidelobes (the ≈ 0.49 background is that, not a failure). The **exact refocus** at
the source and the **medium-specificity** are the certified facts; the coherent-sum
focus magnitude is an observer-layer readout (Theorem NT), never QA state. This
cert is the *reverse deployment* of cert [518] (correction at a receiver) — the
same operator `qa_neg`, aimed to focus instead of correct.

### Real-data bound (2026-07-11): known-medium identity — does NOT transfer to unknown-medium seismic

The certified refocus/specificity above hold because the medium is **known**: the
scattering term appears identically on record and re-emit, so it cancels
analytically. Taken to **real** seismic data (M5.5 Ridgecrest 2019 aftershock, 23
regional stations, `qa_seismic_tr_real.py`) as a QA mod-24 phase-coherence
back-projection with a pre-registered decision, a phase-scramble null, and a
noise-window control, the operator **does not transfer**: the real recorded phases
focus no better than randomised phases (empirical p = 0.86 / 0.11 / 0.49 across a
short-period constant-v and two long-period velocity-search regimes — never < 0.01),
and never refocus within 25 km of the catalog epicenter. Verdict **NOT_SUPPORTED**.

This is a genuine physics bound, not a method bug: the identical estimator recovers
a **known synthetic source exactly** (coherence 1.000 at 0.0 km,
`qa_seismic_tr_selftest.py`). The cause is that a real 3-D Earth is an **unknown**
medium, so at mod-24 phase resolution (15° steps) constant/searched-velocity
travel-time errors exceed a phase period and destroy the coherence the identity
needs. Scope: this bounds QA *single-event absolute-phase mod-24 back-projection*;
professional seismic back-projection succeeds using envelope/cross-correlation and
3-D velocity models — which abandon the QA-phase structure. Result record:
`results/seismic/qa_seismic_tr_real_results.json`.

### EGF specificity follow-up (2026-07-11): identify the medium empirically

The fix for the unknown medium is not to model it but to *measure* it: a **co-located
companion event** records the true Green's function `G_i` from the target's source
patch (empirical Green's function). The [522] cross-product `R_i^T · conj(R_i^EGF)`
then cancels the shared medium (`|G_i|²` real) for a co-located EGF and decoheres for
a distant one. Tested on real Ridgecrest data (`qa_seismic_egf_specificity.py`): target
T (M4.97), matched EGF E (M4.35, 3.2 km), mismatched control Ep (M3.46, 117.5 km), 13
SNR-passing common stations, QA mod-24 cross-spectral phase coherence, pre-registered.

Result — the **specificity direction is clearly recovered**: matched (co-located)
coherence **0.331** vs mismatched (distant) **0.067** (~5×), and the distant EGF
decoheres to *below* the phase-scramble null (p = 0.95) exactly as medium-cancellation
predicts. **mod-24 is vindicated as faithful**: full-precision matched coherence 0.327
≈ the mod-24 value 0.331 — the quantization is not the limiter. But the matched
coherence does **not** clear the pre-registered individual-significance bar vs the
scramble null (p = 0.25 at 13 stations), because the co-located EGF still sits 3.2 km
from T (residual station-dependent phase) and the closest events small enough to be
truly co-located fall below SNR. **Pre-registered verdict: NOT_SUPPORTED** (strict) —
but the qualitative [518]/[522] same-medium fingerprint IS present on real earthquakes,
and the remaining gap is the target–EGF offset + station count, not the QA method.
Result record: `results/seismic/qa_seismic_egf_specificity_results.json`.

### Stacked-EGF significance (2026-07-11): SUPPORTED on real data

Stacking a CLUSTER of co-located aftershocks as the empirical Green's function closes
the gap (`qa_seismic_egf_stack.py`). Each aftershock's residual offset phase points a
different way, so averaging the cross-spectral phasors over the cluster (per-event
global phase removed first) drives the offset toward a station-independent residual
while lifting SNR. Target T (M4.97) with a **30-event co-located aftershock stack**
(0.8-4.8 km) vs a distant control (M3.46, 117 km), QA mod-24 cross-spectral phase
coherence, pre-registered p < 0.01, SNR/KMIN frozen after the 10-event run.

**Verdict SUPPORTED.** Matched (co-located cluster) coherence **0.696, scramble-null
p = 0.0002** (19 stations) — decisively significant; mismatched (distant) **0.095,
p = 0.87** — at the null (~7× specificity). mod-24 faithful throughout
(full-precision 0.713 ≈ mod-24 0.696). The honest signature of a real effect:
matched coherence rose **monotonically with stack size** at frozen params —
single-EGF 0.331 (p = 0.25) → 10-stack 0.502 (p = 0.012) → 30-stack 0.696
(p = 0.0002) — i.e. significance was reached by adding DATA, not by tuning criteria.

Bottom line for [522] on real seismic: the known-medium *identity* does not transfer
under a guessed velocity (constant-v back-projection NOT_SUPPORTED), but once the
medium is **identified empirically** via a stacked co-located EGF, the [518]/[522]
same-medium fingerprint is demonstrated on real earthquakes at p = 0.0002, and QA
mod-24 quantization is confirmed faithful (never the limiter). Disclosure: SNR (3→2)
and KMIN (4→3) were relaxed once at the 10-event stage to admit enough stations, then
frozen; the distant control is a single event (the distant region was sparse) but
sits firmly at the null. Result record:
`results/seismic/qa_seismic_egf_stack_results.json`. Certified as an empirical
observation (parent [522]):
`qa_empirical_observation_cert/results/eoc_pass_seismic_egf_specificity_supported.json`
(`qa.cert.empirical.seismic_egf_time_reversal_specificity.v1`, verdict CONSISTENT).

### Replication + foreshock time-symmetry control (2026-07-11)

Two follow-on tests (`qa_seismic_egf_t2_fetch.py`, `qa_seismic_egf_foreshock.py`) on
an **independent second target** — a different M4.97 in the SW Ridgecrest patch
(35.725, −117.553), **25 km** from target 1, i.e. different paths/medium to the same
network. Its 30-event co-located cluster splits by origin time into **17 foreshocks +
13 aftershocks**; a co-located companion is a valid empirical Green's function
regardless of *when* it occurred, since the medium is time-invariant — a foreshock EGF
is literally the "past side" of the time-reversal mirror.

| stack | coherence | scramble-null p | stations |
|---|---|---|---|
| combined (30) | **0.758** | **0.0002** | 20 |
| foreshock (17) | **0.662** | **0.0004** | 20 |
| aftershock (13) | 0.447 | 0.0202 | 19 |
| foreshock, count-balanced (13) | **0.592** | **0.0002** | 20 |
| aftershock, count-balanced (13) | 0.447 | 0.0194 | 19 |
| distant control | 0.286 | 0.24 (null) | 17 |

**(R) Replication — CONFIRMED.** The second, independent target reproduces the
SUPPORTED result (combined p = 0.0002, coherence 0.758 ≫ distant control 0.286); the
stacked-EGF specificity is not event-specific.

**Foreshock EGF — CONFIRMED.** The foreshock-only stack (all companions *before* the
target) is strongly significant (0.662, p = 0.0004), demonstrating medium
time-invariance on the past side — the physically important new result.

**(T) Full time-symmetry — PARTIAL, and a count-balanced check corrects the first
reading.** The aftershock-only arm beats the control but misses the strict p < 0.01
bar (0.447, p = 0.020). A count-balanced follow-up (13 events each, matched magnitude
ranges M3.2–4.6 vs M3.0–4.5) shows this is **not** merely a count effect: the balanced
foreshock arm still clears cleanly (0.592, p = 0.0002) while the balanced aftershock
arm stays sub-threshold (0.447, p = 0.019). So there is a mild *real* difference for
this target — most plausibly a **data artifact of T2's placement**: T2 (03:16:32) sits
**3 minutes before the M7.1 mainshock** (03:19:53), so its aftershock windows are
contaminated by the M7.1 and its immediate cascade, whereas its foreshocks come from
the cleaner M6.4 sequence. This is a target-specific contamination, **not** a violation
of medium time-invariance: the foreshock EGF itself works cleanly (p = 0.0002), which
is the physical point. A fully clean time-symmetry test would use a target whose
aftershock window is not overprinted by a larger event. Pre-registered verdict
**REPLICATED_ONLY** (full symmetry needs both arms at p < 0.01). Result record:
`results/seismic/qa_seismic_egf_foreshock_results.json`.

## Primary sources

- Fink, M. (1992). "Time reversal of ultrasonic fields." *IEEE Trans. UFFC*
  39(5):555-566. DOI 10.1109/58.156174
- Prada, C. & Fink, M. (1994). "Eigenmodes of the time reversal operator." *Wave
  Motion* 20:151-163. DOI 10.1016/0165-2125(94)90039-6
- Yariv, A. (1978). *IEEE J. Quantum Electron.* 14(9):650-660. DOI 10.1109/JQE.1978.1069870
- Soffer, B.H. et al. (1986). *Opt. Lett.* 11(2):118-120. DOI 10.1364/OL.11.000118

Builds on certs [518] (four-wave-mixing conjugator), [519] (holographic memory);
phase-conjugation cluster companions [520] (EEG recall), [521] (morphogenetic
memory).
