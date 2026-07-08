# [518] QA FWM Phase Conjugate Cert

**Family ID**: 518
**Slug**: `qa_fwm_phase_conjugate_cert_v1`
**Status**: Active
**Registered**: 2026-07-08

## Claim (exact, falsifiable)

The degenerate four-wave-mixing (FWM) phase-sum relation

```
theta_c = theta_f + theta_b - theta_s
```

— two pump waves plus a signal, with the signal entering *conjugated* — is
realized **exactly** in the QA additive group on the A1 alphabet `{1,...,m}`:

```
qa_mod(x)    = ((x - 1) % m) + 1        # A1: values in {1,...,m}, never 0
qa_add(a,b)  = qa_mod(a + b)            # additive identity = m (No-Zero rep of 0)
qa_neg(a)    = qa_mod(-a)               # phase conjugation (an involution)
fwm(pf,pb,s) = qa_mod(pf + pb - s)      # the FWM phase-sum relation
```

With **conjugate pumps** `pb = qa_neg(pf)`, `fwm(pf, pb, s) = qa_neg(s)` exactly,
for every `pf` and `s`. The **distortion-correction theorem** (Yariv 1978;
Zel'dovich–Pilipetsky–Shkunov 1985; Agarwal–Friberg scattering proof) then holds
exactly: aberrate a signal by a phase screen `phi`, phase-conjugate it, and return
it through the **same** `phi` — the result is `qa_neg(s)` exactly (same-medium
recovery). Return through a **different** screen `phi'` leaves exactly the residual
`qa_mod(-s + phi' - phi)`, so exact recovery occurs *only* where `phi' = phi`.
This same-medium specificity is the fingerprint that separates true
phase-conjugate reconstruction from generic denoising.

## Why this cert exists (the emergent-vs-explicit distinction)

A companion empirical investigation (2026-07-08) tested whether QA's *emergent*
self-organizing dynamics spontaneously perform phase-conjugate distortion
correction. They do **not**:

- The self-organizing `QASystem` coupling performs only **generic,
  medium-agnostic denoising** — the same-medium specificity test came back null,
  and pre-adaptation actually *degrades* reconstruction (a fresh coupling beats an
  adapted one).
- The rolling-QCI opposite-sign "Bearden" signature of cert **[155]** is weak and
  inconsistent (domain 2 WEAK, partial_r ≈ −0.13; domain 3 NULL with a sign flip).

Real optical phase conjugation is not emergent — it is produced by an *explicit*
nonlinear element (four-wave mixing: two pumps + signal → conjugate). This cert
supplies exactly that explicit operator, and it reproduces the theorem exactly
where the emergent versions capture only a weak shadow.

## Empirical record (reference implementation)

`qa_fwm_conjugator.py` (repo root, axiom-linter **clean**):

- **Medium-mismatch sweep**: recovery fidelity 1.000 at 0% mismatch → 0.043
  (= chance, 1/24) at 100% mismatch — clean same-medium specificity.
- **Controls**: no-correction and a non-conjugate mixer both at chance (~0.044);
  the proper FWM conjugator = 1.000. Conjugation is load-bearing.
- **Robustness**: graceful degradation under imperfect pump conjugation and
  return-medium jitter.
- **Image demo** (`qa_fwm_conjugator_demo.png`): a "QA"+rings pattern scrambled
  unrecognizable by the phase screen, reconstructed perfectly (fidelity 1.000)
  through the same medium and left as noise (0.073) through a wrong one — the
  classic phase-conjugation "recover the image through frosted glass" result, in
  pure modular arithmetic.

## Checks

| Check | Meaning |
|-------|---------|
| `FWM_CONJUGATE`   | conjugate pumps give `fwm = qa_neg(s)` for all `pf,s` (exhaustive) |
| `DC_SAME_MEDIUM`  | same-medium recovery `= qa_neg(s)` for all `s,phi` (any pump) |
| `DC_DIFF_RESID`   | different-medium residual `= qa_mod(-s+phi'-phi)`; exact only when `phi'=phi` |
| `GROUP_IDENTITY`  | `m` is the unique additive identity, never 0 (A1) |
| `CONJ_INVOLUTION` | `qa_neg` is an involution with exactly two fixed points (`m`, `m/2`) |
| `CONTROL_NONCONJ` | a non-conjugate second pump does not reconstruct |
| `A1_RANGE`        | every operation output lies in `{1,...,m}` |
| `SRC`             | `mapping_protocol_ref.json` present + correct protocol version |
| `F`               | pass/fail fixtures behave as declared |

**Fixtures**: 3 PASS + 2 FAIL
**Self-test**: exhaustive over `{1,...,m}` (m=24), deterministic, integer-only.

## Axiom compliance

All arithmetic is integer on the A1 alphabet `{1,...,m}`; the additive identity is
`m` (the No-Zero representative of 0), never 0. There is no float QA state. The
observer boundary (intensity ↔ phase) is crossed exactly twice, and only in the
reference image demo, satisfying Theorem NT. `qa_fwm_conjugator.py` passes the
axiom linter.

## Primary Sources

- Hellwarth, R.W. (1977). "Generation of time-reversed wave fronts by nonlinear
  refraction." *J. Opt. Soc. Am.* 67(1):1-3. DOI 10.1364/JOSA.67.000001
- Yariv, A. (1978). "Phase conjugate optics and real-time holography." *IEEE J.
  Quantum Electron.* 14(9):650-660. DOI 10.1109/JQE.1978.1069870
- Zel'dovich, B.Ya., Pilipetsky, N.F., Shkunov, V.V. (1985). *Principles of Phase
  Conjugation.* Springer. ISBN 978-3-540-13458-4
- Agarwal, G.S. & Friberg, A.T. "Scattering theory of distortion correction by
  phase conjugation." *J. Opt. Soc. Am.* (distortion-correction theorem)

## Companion

- Cert **[155]** — QA Bearden Phase Conjugate (the weak *emergent* opposite-sign
  QCI signature this explicit operator supersedes).
- Certs **[510]-[514]** — QA Maxwell / scalar-EM cluster (adjacent EM physics).

**Author**: Will Dale + Claude, 2026-07-08.
