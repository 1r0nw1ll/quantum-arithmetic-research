# [519] QA Phase-Conjugate Holographic Associative Memory Cert

**Family ID**: 519
**Slug**: `qa_phase_conjugate_memory_cert_v1`
**Status**: Active
**Registered**: 2026-07-08

## Claim (demonstrated, falsifiable)

The exact QA phase conjugator of cert **[518]** composes into a
**content-addressable associative memory** whose stored patterns are attractors.
Patterns and probes are QA phase vectors in `{1,...,m}^N`. The phase-conjugate
overlap of a probe `x` with stored pattern `k` is

```
C_k = #{ i : qa_add(x_i, qa_neg(P_k_i)) == m }        # == phase-match count
```

(grounded in [518]: two phases match iff their conjugate sum is the additive
identity `m`). Recall is **holographic playback** — each site is reconstructed by
superposing the stored patterns weighted by their sharpened overlap (argmax
phase), iterated to a fixed point. For sufficiently separated pattern families,
stored patterns behave as attractors and a corrupted or partial probe flows to
the nearest — the discrete analog of an optical phase-conjugate resonator (Soffer
1986; Owechko 1987) and of Michael Levin's "target morphology reached as an
attractor from many perturbed starting states." These are **demonstrated
empirical properties under favorable separation, not guaranteed for arbitrary
memories** — near-duplicate/highly-correlated patterns can outvote the
self-pattern, and recall degrades toward chance as patterns approach identical
(see the honest limit below). Phase-locked recall returns the pattern in the
probe's (possibly distorted) frame.

## Empirical record (reference implementation)

`qa_phase_conjugate_memory.py` (repo root, axiom-linter **clean**, m=24, N=120):

| Property | Result |
|----------|--------|
| Stored patterns are exact fixed points | **True** |
| Recall vs corruption | **1.000** to 80%; 0.995 @90%; **0.855 @95%** (breaks near the alphabet noise floor) |
| Capacity (random patterns, 20% corruption) | **1.000** at load K/N up to **4.27** (512 patterns) |
| Correlated patterns (70% shared, 4 prototypes) | **1.000** to K=128 |
| Content-addressable (probe @35% → exact nearest) | **1.000** |
| Distortion tolerance — global phase screen | naive **0.000**, **phase-locked 1.000** |

**Why capacity is so high (honest mechanism):** random patterns over the mod-24
alphabet are near-orthogonal — expected crosstalk overlap between distinct
patterns is only ~N/m per pair — so a correct probe's overlap dominates even at
large K. This is a genuine consequence of the large phase alphabet, **not a claim
of unbounded capacity**: recall degrades toward chance as stored patterns are
made nearly identical, or as corruption reaches the noise floor (hence 0.855 at
95% corruption).

**The [518] connection — distortion-tolerant recall.** A global phase screen `phi`
on the probe breaks naive recall entirely (0.000), because every site is shifted.
**Phase-locked recall** scans the global compensation phase that maximizes overlap
— the discrete analog of a phase-conjugate mirror self-adjusting to the distorting
medium (the distortion-correction theorem of [518]) — and recovers the pattern
(1.000 for all `phi`). This is content-addressable recall that is robust to an
unknown global distortion.

## Checks

| Check | Meaning |
|-------|---------|
| `PC_OVERLAP` | overlap via `qa_add(x,qa_neg(P))==m` equals the exact match count |
| `FIXED_POINTS` | stored patterns recall to themselves exactly |
| `RECALL` | a corrupted probe recalls its exact stored pattern |
| `CONTENT_ADDRESSABLE` | probe nearest to pattern `s` recalls exactly `s` |
| `DISTORTION_TOLERANT` | phase-locked recall recovers through a global phase screen |
| `NAIVE_DISTORTION_FAILS` | naive recall does *not* recover through the screen (control) |
| `A1_RANGE` | every output lies in `{1,...,m}` |
| `SRC` / `F` | mapping ref present; pass/fail fixtures behave as declared |

**Fixtures**: 2 PASS + 2 FAIL
**Self-test**: deterministic, integer-only, pure stdlib (small fixed memory).

## Axiom compliance

Phase state (patterns, probe, reconstruction) is integer in `{1,...,m}` (A1/S2);
the additive identity is `m`, never 0. Correlation and vote weights are float
observer-layer scores (like E8-alignment / harmonic-index), never fed back as QA
state — the argmax that writes state is over integer phases (Theorem NT). No `**2`
(S1). The reference implementation passes the axiom linter.

## Primary Sources

- Soffer, B.H., Dunning, G.J., Owechko, Y., Marom, E. (1986). "Associative
  holographic memory with feedback using phase-conjugate mirrors." *Opt. Lett.*
  11(2):118-120. DOI 10.1364/OL.11.000118
- Owechko, Y. (1987). "Nonlinear holographic associative memories." *IEEE J.
  Quantum Electron.* 25(3):619-634.
- Hopfield, J.J. (1982). "Neural networks and physical systems with emergent
  collective computational abilities." *PNAS* 79(8):2554-2558. DOI 10.1073/pnas.79.8.2554
- Levin, M. (2021). "Technological Approach to Mind Everywhere." *Front. Syst.
  Neurosci.* (target morphology as an attractor).

## Companion

- Cert **[518]** — QA FWM Phase Conjugate (the exact operator this memory composes).
- `docs/theory/QA_SYNTROPY_PHASE_CONJUGATE_INVESTIGATION.md` — the investigation arc.

**Author**: Will Dale + Claude, 2026-07-08.
