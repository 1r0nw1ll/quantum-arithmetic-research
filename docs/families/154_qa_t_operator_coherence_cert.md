# Family [154] QA_T_OPERATOR_COHERENCE_CERT.v1

## One-line summary

The QA T-operator's prediction error, measured as a rolling coherence index (QCI), carries genuine forward-looking structural information across finance, EEG, and audio domains.

## Mathematical content

### Mechanism

1. Multi-channel signal → topographic k-means → discrete microstates
2. Microstate transitions → QA (b,e) states mod m
3. T-operator T(b,e) = (e, ((b+e-1) % m) + 1) predicts next state
4. Rolling prediction accuracy over window W = **QA Coherence Index (QCI)**
5. QCI anticorrelates with future system instability

### Core result (Finance, Tier A hardened)

| Metric | Value |
|--------|-------|
| QCI vs future vol (OOS) | r = -0.3175, p < 10⁻⁶ |
| Partial r (controlling lagged RV) | -0.2154, p < 10⁻⁸ |
| Robustness grid | 67/80 significant (84%) |
| Permutation test | real χ²=23.48 > max null 12.88 |
| Early warning | χ²=155, precision=54%, recall=48% |

**Interpretation**: When cross-asset dynamics deviate from QA Fibonacci-shift predictions (low QCI), future volatility increases. The T-operator error signal carries independent forward-looking information that lagged realized volatility does not capture.

### Cross-domain evidence

| Domain | Signal type | Result | p-value |
|--------|------------|--------|---------|
| Finance | Forward prediction | partial r=-0.22 beyond RV | <10⁻⁸ |
| EEG | Contemporaneous classification | ΔR²=+0.210, 10/10 patients | 2.9×10⁻³³ |
| Audio | Structural detection | partial r=+0.752 beyond lag-1 AC | 0.020 |

### Why this works (Keely Triune connection)

Low QCI = system departing from QA-predicted dynamics = loss of structural coherence. In Keely's framework: departure from the DOMINANT (neutral balance) toward instability. The T-operator coherence measures how closely the system follows its own inherent arithmetic — when it stops following, stress is emerging.

## Checks

| ID | Description |
|----|-------------|
| TC_1 | schema_version correct |
| TC_OBS | observer pipeline fully declared |
| TC_QCI | QCI construction specified |
| TC_OOS | out-of-sample protocol |
| TC_PARTIAL | partial correlation significant beyond baseline |
| TC_ROBUST | robustness grid >50% significant |
| TC_W | ≥2 domain witnesses |
| TC_F | finance domain present |

## Source grounding

- **Ben Iverson QA framework**: T-operator = Fibonacci shift on (Z/mZ)²
- **Scripts 30-37** (frozen, hashes in FROZEN_HASHES_30_37.txt): full empirical pipeline
- **Cert [153]** Keely Triune: singularity/satellite/cosmos = dominant/enharmonic/harmonic
- **Cert [128]** Spread Period: pi(9)=24 = cosmos orbit period
- **EEG scripts**: eeg_orbit_classifier.py, eeg_chbmit_scale.py
- **Audio scripts**: qa_audio_residual_control.py
- **`docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md`** (2026-04-16): independent public-data replication, sign flip finding — see Verification Note
- **`docs/theory/QA_FINANCE_EXISTING_ART_AUDIT.md`** (2026-05-24): most recent finance status audit, notes QCI superseded by [209] — see Verification Note

## Connection to other families

- **[153] Keely Triune**: QCI measures departure from DOMINANT balance
- **[122] Empirical Observation**: finance result extends the empirical bridge
- **[145]-[146] Path Shape/Scale**: QCI is a path-level coherence measure
- **[147] Synchronous Harmonics**: coprime sync → T-operator is the sync mechanism

## Fixture files

- `fixtures/tc_pass_finance_hardened.json` — full finance Tier A result with OOS, partial corr, robustness, permutation
- `fixtures/tc_pass_cross_domain.json` — three-domain evidence (finance + EEG + audio)

## Verification Note (2026-07-06)

This cert (`created_utc: 2026-04-01`) certifies the finance QCI result as
"Tier A hardened" — partial r=-0.2154, p<10⁻⁸ beyond lagged realized
vol, 67/80 robustness grid, permutation-beating. The private backing
scripts (`~/Desktop/qa_finance/30-42`) are off-limits per project rules
(frozen, private) and this machine doesn't currently have that directory
at all, so the private numbers themselves cannot be independently
re-run. However, **two follow-up audits already exist in this repo**
(both post-dating this cert) that materially change how "Tier A
hardened" should be read, and neither was folded back into this cert
before now:

**1. Sign does not reproduce on public data** —
`docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md` (2026-04-16) ran the
publicly-runnable `qci_v2_real_finance.py` (real yfinance data, same
6-asset/K=6/mod-24/window=63 pipeline the docstring claims is "the same
pipeline as validated script 35") end to end: real data, n=1171 OOS
days, 4 surrogate null families (phase-randomized, AR1, block-shuffled,
row-permuted; 200 each). Result: **r=+0.4355, partial r=+0.2556** —
opposite sign from this cert's -0.3175/-0.2154, and roughly double the
magnitude. The signal beats all 4 nulls on raw r (2/4 on partial r), so
*some* real structure survives — but the sign, which is what
"low QCI → high future vol" depends on, does not. The report's own
verdict: "Any downstream use of QCI that depends on sign is **not
supported** by the public replica." Candidate causes it identifies:
structural pipeline drift from the true script 35, private-result
instability under the arbitrary CLUSTER_MAP relabeling (a k-means label
permutation nuisance parameter, 720 possible maps), or a genuinely
map-dependent sign (the "benign" reading, which still means published
sign numbers aren't reproducible without the exact map).

**2. Architecturally superseded** —
`docs/theory/QA_FINANCE_EXISTING_ART_AUDIT.md` (2026-05-24, the most
recent finance audit in the repo) independently classifies
`qci_v2_real_finance.py` as "tested-mixed (architectural cluster-map
dependency; superseded as a state-assignment by [209] generator
inference)" — i.e. the K-means-then-hand-tuned-CLUSTER_MAP state
assignment this cert certifies is no longer the project's own
recommended approach for finance QA state construction.

**What was fixed**: added `reproducibility_caveats` to the finance
witness in both fixtures citing both findings, added `source_grounding`
citations to both fixtures, and hardened the validator with a new
`TC_REPRO` check — any witness declaring `"status": "...hardened..."`
must now carry a non-empty `reproducibility_caveats` field, verified to
reject a witness that claims hardened status without one. This doesn't
retract the cert (EEG and audio evidence are unaffected, and the public
replica does show real above-null structure), but it stops "Tier A
hardened" from reading as a settled, reproduced result when the
project's own later work found otherwise.

**Not investigated further in this pass** (flagged, not resolved): the
EEG witness numbers here (`mean dR2=+0.210, Fisher p=2.9e-33`) don't
obviously match a differently-worded EEG finding elsewhere in this
project's memory (`Fisher p=0.006`, "per-window topographic transitions
+ per-patient z-score") — these may be different statistics on the same
underlying pipeline (Fisher-combined dR² vs. seizure/baseline orbit
Fisher test) or may be a real discrepancy; worth a dedicated follow-up
audit rather than resolving inline here.

Also noted: this family has no FAIL fixture (both `fixtures/*.json`
files are PASS), unlike every other cert family in this project's
`qa_alphageometry_ptolemy/` — not fixed in this pass, but worth adding a
FAIL fixture (e.g. missing `reproducibility_caveats` on a hardened
witness, which would now be caught by `TC_REPRO`) in a future visit.
