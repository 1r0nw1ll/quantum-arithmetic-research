# [222] QA Madelung Anomaly Boundary Cert

**Schema:** `QA_MADELUNG_ANOMALY_BOUNDARY_CERT.v1`
**Status:** draft, 2026-04-13
**Originator:** Will Dale
**Validator:** `qa_alphageometry_ptolemy/qa_madelung_anomaly_boundary_cert_v1/qa_madelung_anomaly_boundary_cert_validate.py`
**Theory note:** [`docs/theory/QA_MADELUNG_ANOMALY_BOUNDARY.md`](../theory/QA_MADELUNG_ANOMALY_BOUNDARY.md)
**Prior:** [[220]](220_qa_madelung_d_ordering_cert.md) Madelung d-Ordering (axiomatic backbone)

## Claim

Every known Madelung anomaly in the ground-state electron configuration of a neutral atom satisfies

`|d(source subshell) − d(destination subshell)| ≤ 1`

where `d = n + l` is the QA d-coordinate under `(b, e) = (n, l)`.

Verified on 20/20 known anomalies: 10 at `|Δd| = 0` (intra-class f↔d / d↔p), 10 at `|Δd| = 1` (inter-adjacent-class s↔d).

## Null test

Uniform random 2-subshell picks from the first 20 Madelung positions:
- Baseline `|Δd| ≤ 1` rate: 36.8%
- Observed: 100% (20/20)
- Enrichment: **2.71×**
- Binomial p-value: **p = 2.1 × 10⁻⁹**

## Scope — necessary, not sufficient

QA identifies **where** anomalies can live (the boundary zone). Atomic-physics mechanisms (exchange, relativistic corrections, half/full-shell stabilisation) determine **which** zone members actually anomalise.

- All 20 known anomalies lie in zone — necessity supported
- Ti, V, Mn, Fe, Co, Ni (d=4↔5); Ta, W, Re, Os, Ir (d=6↔7) lie in zone but follow Madelung — zone is not sufficient
- Claim is falsifiable: any future experimentally-established anomaly with `|Δd| ≥ 2` breaks the claim

## Fixtures

- `fixtures/mab_pass_20_anomalies.json` — 20 known neutral-atom anomalies with zone verification
- `fixtures/mab_pass_ions_42.json` — extension to 42 cation/anion ground states (21 anomalous, 21 Madelung-exact)
- `fixtures/mab_pass_superheavy_54.json` — out-of-sample check against 54 predicted superheavy (Z=119–172) configurations (Fricke/Nefedov/Pyykkö relativistic calculations)
- `fixtures/mab_fail_fake_high_dd_anomaly.json` — negative control (fabricated `|Δd| = 3` entry)

## Verification Note (2026-07-04)

Independently checked the 20 neutral-atom entries against real electron
configurations (NIST Atomic Spectra Database; Sato et al. 2015 for Lr) —
all 20 confirmed correct (Cr, Cu, Nb, Mo, Ru, Rh, Pd's unique zero-5s case,
Ag, La, Ce, Gd's half-filled-4f7 exception, Pt, Au, Ac, Th's zero-5f case,
Pa, U, Np, Cm, Lr). The 6 counterexamples (Ti, V, Mn, Fe, Ta, W) also
checked correct.

**Hardening found and fixed**: the validator previously only checked the
*fixture's own* internal arithmetic consistency (`d=n+l`, `delta_d`
formula) — it never compared the fixture's anomaly list against an
independent ground truth. A future edit that silently dropped or
fabricated an entry would have passed undetected. Added
`MAB_REFERENCE_MATCH`: a hardcoded, independently-verified table of the 20
real `(n_src, l_src, n_dst, l_dst)` values, checked exactly against any
fixture whose anomaly Z-set matches this reference set (ion/superheavy
fixtures, which cover different Z-sets and are appropriately hedged as
"extension"/"out-of-sample prediction" rather than settled fact, are
unaffected). Confirmed the check both passes the real data and catches a
deliberately corrupted entry.

## Checks

| ID | Scope |
|---|---|
| MAB_1 | schema version |
| MAB_ANOMALIES | list well-formed, ≥ 1 entry, (n, l) integer coords |
| MAB_REFERENCE_MATCH | 20-neutral-atom fixtures match a hardcoded, independently-verified reference table exactly |
| MAB_MAPPING | each entry's `d` equals `n + l`; `delta_d` equals `|d_src − d_dst|` |
| MAB_ZONE | all entries satisfy `|Δd| ≤ 1` |
| MAB_COVERAGE | distribution matches enumeration; total equals listed anomalies |
| MAB_NULL | null_test block well-formed; p-value < 0.01 |
| MAB_COUNTEREX | ≥ 1 counterexample listed (zone member that follows Madelung) |
| MAB_SRC | NIST / Sato / Will Dale attribution |
| MAB_WITNESS | ≥ 3 witnesses including zone_coverage, null_significance, necessity_not_sufficiency |
| MAB_F | fail_ledger well-formed |

## References

- NIST Atomic Spectra Database
- Sato, T. K. et al. (2015). *Measurement of the first ionization potential of lawrencium, element 103.* Nature 520, 209–211.
- [[220]](220_qa_madelung_d_ordering_cert.md), [[221]](221_qa_nuclear_magic_spin_extension_cert.md)

## Correction history

- **v1 (this cert)**: initial structural claim + null test. Scope: necessary not sufficient; falsifiable.
