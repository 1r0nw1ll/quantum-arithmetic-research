# [221] QA Nuclear Magic Spin-Extension Cert

**Schema:** `QA_NUCLEAR_MAGIC_SPIN_EXTENSION_CERT.v1`
**Status:** draft, 2026-04-13
**Originator:** Will Dale
**Validator:** `qa_alphageometry_ptolemy/qa_nuclear_magic_spin_extension_cert_v1/qa_nuclear_magic_spin_extension_cert_validate.py`
**Theory note:** [`docs/theory/QA_NUCLEAR_MAGIC_SPIN_EXTENSION.md`](../theory/QA_NUCLEAR_MAGIC_SPIN_EXTENSION.md)
**Atomic companion:** [[220]](220_qa_madelung_d_ordering_cert.md)

## Claim

Under axiom **D1** (Dirac fermion spin: `σ ∈ {1, 2}` with `j = l + (2σ − 3)/2`) and physics input **P1** (nuclear ratio `r = α/ℏω ∈ [1/3, 1/2)`):

1. The fractional-½ promotion rule on QA `(b, e, σ)` with condition `(σ = 2, b = e + 1, l ≥ l*)`, where `l* = ⌈1/r⌉`, gives `l* = 3` by integer ceiling.
2. The ½ promotion amount is the Dirac spin unit, derived from D1 (not a free parameter).
3. Magic-shell criterion: `N_eff ∈ {0, 1, 2}` OR `N_eff` half-integer.
4. Cumulative populations at magic closures reproduce **exactly** the experimental nuclear magic numbers `{2, 8, 20, 28, 50, 82, 126}`.

## Physics input transparency

One physical ratio: `r = α/ℏω` in a narrow window `[1/3, 1/2)`. Nuclear empirical values (Mayer-Jensen 1950; Bohr-Mottelson 1969) place `r ≈ 0.3–0.4`, inside the window. The integer ceiling `⌈1/r⌉ = 3` is forced for the entire window — not a fit, a discrete consequence of a measured ratio.

Atomic `r ≈ 0.01–0.02` gives `l* ≈ 50–100`, never reached in the aufbau. This explains why atomic Madelung [220] needs no spin extension while nuclear shells do.

## What this cert does NOT claim

- No derivation of `r` from lower axioms. `r` is a property of the strong interaction (QCD) and requires empirical input or QCD-level derivation.
- No prediction for exotic nuclei (drip lines, hypernuclei, neutron matter) where effective `r` differs.
- No explanation of Madelung anomalies (Cr, Cu, lanthanides) at the atomic level.
- No claim that QA derives nuclear shell magic numbers purely structurally — the cert is **axiomatic D1 + one physical ratio P1**.

## Fixtures

- `fixtures/nms_pass_magic_7.json` — full magic breakdown + witnesses
- `fixtures/nms_fail_wrong_threshold.json` — negative control (claims l*=2, produces wrong magic)

## Checks

| ID | Scope |
|---|---|
| NMS_1 | schema_version matches |
| NMS_D1 | Dirac axiom well-formed (sigma domain, j formula, population formula) |
| NMS_HO | all declared (b, e, sigma) have HO N = 2b − e − 2 computed correctly |
| NMS_PROMOTION | promotion rule declared and ½ matches Dirac unit |
| NMS_THRESHOLD | declared l* satisfies l* = ⌈1/r_upper⌉, with r_window narrow (width ≤ 1/6) |
| NMS_MAGIC | validator independently reproduces magic sequence {2, 8, 20, 28, 50, 82, 126} |
| NMS_P1 | physics input P1 present with r window |
| NMS_SRC | attribution to Mayer-Jensen + Bohr-Mottelson + Will Dale |
| NMS_WITNESS | ≥ 3 witnesses including exact_match, threshold_derivation, dirac_derivation |
| NMS_F | fail_ledger well-formed |

## Cross-references

- [`docs/theory/QA_NUCLEAR_MAGIC_SPIN_EXTENSION.md`](../theory/QA_NUCLEAR_MAGIC_SPIN_EXTENSION.md)
- [`docs/theory/QA_MADELUNG_D_ORDERING.md`](../theory/QA_MADELUNG_D_ORDERING.md) — atomic analogue, no spin extension
- [[220]](220_qa_madelung_d_ordering_cert.md), [[217]](217_qa_fuller_ve_diagonal_decomposition_cert.md), [[218]](218_qa_haramein_scaling_diagonal_cert.md), [[219]](219_qa_fibonacci_resonance_cert.md)

## References

- Mayer, M. G. (1950). Phys. Rev. 78, 16–21.
- Jensen, J. H. D. (1950). Phys. Rev. 78, 22.
- Bohr, A., Mottelson, B. R. (1969). *Nuclear Structure, Vol. I.* W. A. Benjamin.
- Phys3002 lecture notes (Soton): `05_shell.pdf` — spin-orbit is ~20× atomic; 1f₇/₂ is first crossing.

## Correction history

- **v0**: "threshold l ≥ 3 is empirically calibrated" — Will pushed back (if calibrated, not done).
- **v1 (this cert)**: threshold derived from single physical ratio `r ∈ [1/3, 1/2)`; `⌈1/r⌉ = 3` by integer ceiling. Promotion ½ from Dirac D1 axiom. Scope explicit: "axiomatic + one physical ratio" rather than "fully axiomatic."
