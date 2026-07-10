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
