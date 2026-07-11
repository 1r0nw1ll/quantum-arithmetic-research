<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; primary sources (Feinberg 1982 DOI 10.1364/OL.7.000486; Cronin-Golomb et al. 1984 DOI 10.1109/JQE.1984.1072018; Zel'dovich 1985 ISBN 978-3-540-13458-4) are cited in mapping_protocol_ref.json and the validator. -->
# [523] QA Self-Pumped Phase Conjugate Cert

**Family ID**: 523
**Slug**: `qa_self_pumped_conjugate_cert_v1`
**Status**: Active
**Registered**: 2026-07-11

## Claim (exact, falsifiable)

The **self-pumped** phase conjugator — the internal-reflection ("cat") mirror that
supplies **no external pump** (Feinberg 1982; Cronin-Golomb et al. 1984) — is realized
in the QA additive group on the A1 alphabet `{1,...,m}`. Where cert **[518]**
*supplies* two pump beams, here the incident beam fans and internally reflects,
generating its **own** counter-propagating pump pair `p_b = qa_neg(p_loop)`, and above
a reflectivity threshold the loop self-oscillates into the phase conjugate:

```
self_pumped(p, s) = fwm(p, qa_neg(p), s) = qa_mod(p + qa_neg(p) - s) = qa_neg(s)
```

The key that makes this work with no external reference:

> `fwm(p, qa_neg(p), s) = qa_neg(s)` is **independent of the pump `p`**.

Only the *conjugate-pair condition* matters, and the loop geometry (internal
reflection: the return beam is `qa_neg` of the forward loop field) enforces it
automatically — so whatever self-pump the loop settles on, the output is the correct
conjugate. That is exactly why Feinberg's cat mirror self-starts from noise with no
external beam.

## Checks (validator, integer-only; gain/amplitude are observer-layer, Theorem NT)

| Check | Meaning |
|-------|---------|
| `C1 PUMP_INDEPENDENCE` | `fwm(p, qa_neg(p), s) == qa_neg(s)` for **all** `p,s` (exhaustive m×m) |
| `C2 SELF_STARTING` | output stays `qa_neg(s)` as the self-pump wanders from a noise seed (= C1 in action) |
| `C3 THRESHOLD` | loop amplitude `A' = g·A/(1+A)` self-oscillates iff `g>1` (fixed point `A*=g−1`); decays to `A=0` for `g≤1`; `g_c=1` reflectivity threshold, `g=1` marginal |
| `C4 SELF_PUMPED_DC` | aberrate by `phi` → self-conjugate (any self-pump, no external beam) → return through same `phi` = `qa_neg(s)` for all `s,phi`; a different screen leaves `qa_mod(-s+phi'-phi)` (same-medium specificity) |
| `C5 A1_RANGE` | every phase operation stays in `{1,...,m}` |

Reference implementation: `qa_self_pumped_conjugator.py` (repo root, axiom-linter
clean) — PUMP_INDEPENDENCE 576/576, SELF_STARTING 200/200 (output locked through a
wandering self-pump while amplitude self-builds), threshold sweep (`g_c=1`), and
self-referenced distortion correction over all `s,phi`.

**Fixtures**: 3 PASS (same-medium recovery; different-medium residual; pump-independent
alternate) + 2 FAIL (missing field; wrong conjugate). **Self-test**: exhaustive
C1–C5 on `{1,...,24}` + fixtures; deterministic, integer-only, stdlib.

## Relation to the phase-conjugate cluster

- **[518]** externally pumped (supplies the pumps) — this is its **self-starting
  cousin** (the pump is internal).
- **[519]** phase-conjugate holographic memory — a resonator *driven by stored
  patterns*; here the pump is self-generated from the signal, with a threshold.
- **[522]** time-reversal focusing — the [518] operator reversed.

## Honest scope

The gain `g` and amplitude `A` (the self-oscillation threshold) are observer-layer
reals (Theorem NT), never QA state; the phase layer is integer throughout. The QA
model captures the *logic* of self-pumping — pump-independence, self-starting, a
reflectivity threshold, self-referenced distortion correction — not the continuous
photorefractive dynamics of a physical cat mirror.

## Primary sources

- Feinberg, J. (1982). "Self-pumped, continuous-wave phase conjugator using internal
  reflection." *Opt. Lett.* 7(10):486–488. DOI 10.1364/OL.7.000486
- Cronin-Golomb, M., Fischer, B., White, J.O., Yariv, A. (1984). "Theory and
  applications of four-wave mixing in photorefractive media." *IEEE J. Quantum
  Electron.* 20(1):12–30. DOI 10.1109/JQE.1984.1072018
- Zel'dovich, B.Ya., Pilipetsky, N.F., Shkunov, V.V. (1985). *Principles of Phase
  Conjugation.* Springer. ISBN 978-3-540-13458-4

Builds on cert [518]; cluster companions [519], [520], [521], [522].
Author: Will Dale + Claude 2026-07-11.
