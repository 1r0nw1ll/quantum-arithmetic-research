# Family [215] QA_RESONANCE_BIN_CORRESPONDENCE_CERT.v1

## One-line summary

At modulus `m`, the equivalence class `[k]_m = { x ∈ ℝ : quantize(x, m) = k }` is isomorphic to a resonance tolerance bandwidth centered on integer eigenvalue `k`. Hensel lift `m → m·p` (for prime divisor `p` of `m`) is **progressive bandwidth narrowing**. This supplies a first candidate for the permissibility-filter gap flagged in `docs/theory/QA_SYNTAX_SVP_SEMANTICS.md` (Dale Pond + Vibes, letter of 2026-04-05).

## The claim

**QA syntax** (integer equality under `mod m`) corresponds to **SVP semantics** (sympathetic transmission window width) via the simple relation:

```
window width at modulus m  ∝  2π / m
```

Finer `m` ⇒ narrower window ⇒ stricter resonance. Coarser `m` ⇒ broader window ⇒ more permissive coupling. This is the first formal "filter function" matching Vibes's requirement that structural sufficiency (QA) must be restricted by physical permissibility (SVP) to yield transmissive coupling.

## Why this matters

Memory `a9307705` (Dale Pond + Vibes letter, 2026-04-05) stated:

> Vibes has not given us the filter function; he has given us a principled reason to expect it exists.

This cert names a filter. The function is not unique — other permissibility filters may compose on top (phase alignment, harmonic concordance) — but the bin-width correspondence is **a structurally well-formed candidate that can be empirically tested**, not a post-hoc reconciliation.

## Witnesses

### W1 — Arnold-tongue width ≡ QA bin width (empirical)

Two coupled phase oscillators:
```
dφ₁/dt = 1         + K · sin(φ₂ − φ₁)
dφ₂/dt = 1 + δ     + K · sin(φ₁ − φ₂)
```

The classical Arnold-tongue prediction: phase-locking occurs when `|δ| ≤ K` (linear-approximation tongue width). Its QA analogue: the phase difference `φ₂ − φ₁`, when quantized to `m` bins of width `2π/m`, concentrates in a single bin once `K` exceeds a critical value `K*(m)`.

**Prediction**: `K*(m)` is **monotone increasing in `m`** — finer bins require stronger coupling to pull the phase into a single bin.

**Empirical verification** (in validator `RBC_BINWIDTH` check, δ=0.05, n_steps=4000):

| m | Critical K* (locked_frac ≥ 0.9) |
|---|---|
| 6 | 0.060 |
| 18 | 0.080 |
| 48 | 0.100 |

Monotone increase, as predicted. The Arnold tongue is QA-interpretable: the tongue's angular half-width at a given K is the bin width at the corresponding modulus.

### W2 — Hensel lift is progressive narrowing (external reference)

Orbit-family count on `S_{3^k}` grows with `k`: Pudelko / Hensel prediction and empirical verification in `qa_brainca_selforg_v2.py` and `qa_hensel_selforg_experiment.py`:

- mod 3 → 2 families
- mod 9 → 5 families  
- mod 27 → further splitting

More families = finer distinctions = narrower permissibility windows at each level. This is the Hensel lift read as a *tightening* of the filter function, not a wholly new filter.

### W3 — Integer-only round-trip (axiom compliance)

Round-trip `continuous → TypeA quantize → int64 bin → bin-center → re-bin` preserves the integer bin assignment exactly, with **zero use of `fractions.Fraction`** (which auto-reduces and would violate S2 integer-state axiom). Witness: 500 Gaussian samples round-trip cleanly at both m=9 and m=24.

## Compliance

| Axiom | Status | Note |
|---|---|---|
| A1 (no-zero) | ✓ | `qa_bin` shifts by +1 to avoid the 0 state |
| A2 (derived coords) | n/a | Cert is about quantization, not (d, a) derivation |
| S1 (no `**2`) | ✓ | Arnold sim uses `sin`, no squaring |
| S2 (no float state) | ✓ | All bin state `int64`; floats cross only at observer boundary |
| T1 (path time) | ✓ | Time = integer step count in the simulation |
| T2 (firewall) | ✓ | Continuous oscillator is observer layer; bins are QA layer; boundary crossed at `np.floor(phase · m / (2π))` |

## Links

- Theory: `docs/theory/QA_SYNTAX_SVP_SEMANTICS.md` — the syntax/semantics complementarity frame that this cert instantiates
- Related: [191] Bateson Learning Levels (filtration structure), [153] Keely Triune, [154] QCI T-operator coherence
- Source letter: OB `a9307705-6ff0-4422-bd62-9889dcd0f1b2`

## Running

```bash
cd qa_alphageometry_ptolemy/qa_resonance_bin_correspondence_cert_v1
python qa_resonance_bin_correspondence_cert_validate.py                              # run pass fixture
python qa_resonance_bin_correspondence_cert_validate.py fixtures/rbc_fail_no_hensel.json  # should FAIL
python qa_resonance_bin_correspondence_cert_validate.py --self-test                  # JSON form used by meta-validator
```

## Source

Will Dale + Claude (Opus 4.6), 2026-04-12. Grounded in Arnold (1961) phase-locking theory and the Dale Pond + Vibes permissibility letter of 2026-04-05 (OB `a9307705`).
