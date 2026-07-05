# Family [219] QA_FIBONACCI_RESONANCE_CERT.v1

## One-line summary

Mean-motion resonances across 8+ planetary systems preferentially select Fibonacci ratios (2:1, 3:2, 5:3, 8:5) over non-Fibonacci ratios of the same order — consistent with the T-operator as a universal dynamical preference.

## Mathematical content

### The observation

Among order-1 resonances (|p-q|=1), there are 9 coprime ratios: 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9. Only 2 are Fibonacci (2:1 and 3:2) = 22% expected under uniform selection.

**Nature selects Fibonacci 77% of the time** (33/43 instances, p < 10⁻⁶).

### Cross-validation

| System | Fibonacci | Non-Fib | Fib% |
|--------|-----------|---------|------|
| Solar system (29 total) | 21 | 8 | 72% |
| Exoplanets (31 total) | 24 | 7 | 77% |
| **Combined** | **45** | **15** | **75%** |

### Statistical tests

| Test | Result | p-value |
|------|--------|---------|
| Unique ratios binomial (8/14 vs 31%) | SIGNIFICANT | 0.040 |
| All instances binomial (45/60 vs 31%) | SIGNIFICANT | < 10⁻⁶ |
| Fisher combined (order-stratified) | SIGNIFICANT | < 10⁻⁶ |
| Exoplanet-only (24/31 vs 31%) | SIGNIFICANT | < 10⁻⁶ |
| Order-1 only (33/43 vs 22%) | SIGNIFICANT | < 10⁻⁶ |

### QA interpretation

The T-operator = Fibonacci shift [[0,1],[1,1]]. Fibonacci ratios F_n/F_m are convergents of the golden ratio φ. If the T-operator governs orbital dynamics, periods locked to Fibonacci ratios sit at dynamically deeper attractors — not just easier to form (low order) but harder to escape once captured.

Standard perturbation theory explains why low-order resonances are preferred (strength scales as e^|p-q|). It does **not** explain why 2:1 and 3:2 dominate over 4:3, 5:4, ..., 10:9 within the same order class. QA provides a structural selection principle that fills this gap in the three-body problem.

## Tier classification

**Tier 2→3 candidate.** Cross-validated on independent datasets (solar system + 7 exoplanet systems spanning billions of years and light-years apart). All five statistical tests significant. Honest caveats documented.

### Caveats

1. **Detection bias**: 2:1 and 3:2 produce strongest transit timing variations — easiest to detect
2. **Catalogue completeness**: not exhaustive; future discoveries could change ratios
3. **4:3 is "almost Fibonacci"**: 4 sits between Fibonacci numbers 3 and 5
4. **Expected rate depends on max ratio considered**: p≤10 gives 31% expected
5. **Instance weighting**: 3:2 appears 22 times; unique-ratio test (p=0.040) is more conservative
6. **Formation physics**: resonance capture depends on tidal dissipation history

## Witness systems

Solar System, TRAPPIST-1, HD 110067, K2-138, Kepler-80, Kepler-223, TOI-178, GJ-876, HD 158259

## Sources

- Murray & Dermott, *Solar System Dynamics* (1999)
- Luger et al., Nature Astronomy 1, 0129 (2017) — TRAPPIST-1
- Luque et al., ApJ 968, L12 (2024) — HD 110067
- Christiansen et al., AJ 155, 57 (2018) — K2-138
- Mills et al., Nature 533, 509-512 (2016) — Kepler-223

## Validator

`qa_alphageometry_ptolemy/qa_fibonacci_resonance_cert_v1/qa_fibonacci_resonance_cert_validate.py --self-test`

## Verification Note (2026-07-04)

Given this is the project's flagship/anchor empirical result, audited it
carefully against the paper (`papers/ready-for-submission/fibonacci-resonance/paper.tex`,
now further along than this doc reflects — see below) and real
astronomical data.

**Raw data spot-checked as genuine**: independently recomputed the
TRAPPIST-1 resonance chain from the actual published orbital periods
(Luger et al. 2017: b=1.510826d, c=2.421937d, d=4.049219d, e=6.101013d,
f=9.207540d, g=12.352446d, h=18.772866d) — all 6 consecutive-pair ratios
match the catalogue exactly (8:5, 5:3, 3:2, 3:2, 4:3, 3:2). Also confirmed
the Galilean-moon Laplace resonance (Io:Europa:Ganymede = 4:2:1) and
Pluto:Neptune 3:2. This is real, accurately-transcribed astronomical data,
not fabricated.

**Statistics independently reproduced, and a real bug found and fixed**:
recomputed the expected Fibonacci fraction (2/9 ≈ 0.222 for order-1; 10/32
= 0.3125 overall) and the headline binomial p-values exactly via
`scipy.stats.binomtest` and a stdlib-only `math.comb`-based exact tail —
both give **4.19 × 10⁻¹²** (all-instances) and **4.71 × 10⁻¹⁴** (order-1),
matching the published paper's stated 4.2×10⁻¹² / 4.7×10⁻¹⁴ almost
exactly. But the cert's own fixture (`fr_pass_full_catalogue.json`) had
declared `p_value` and `order1_p_value` as a stale placeholder
**`1e-06`** for both — off by 6+ orders of magnitude from the true values,
and the validator had no check that would have caught this (it computed
`fib_count`/`nonfib_count` from the catalogue but never actually compared
them, or any p-value, against the declared statistics). Fixed the fixture
to the correct precise values and added `FR_RECOMPUTE`: a real,
independent, stdlib-only recomputation of counts and both headline
p-values, cross-checked against the catalogue and declared statistics.
Confirmed it both passes the corrected fixture and catches the exact
original bug (restoring the `1e-06` placeholder now fails validation).

**One remaining, honestly-flagged imprecision**: the fixture's
`fisher_combined_chi2`/`fisher_combined_df` were also stale (58.98/df=12);
recomputing Fisher's method across orders 1–5 (excluding order 0, matching
the paper's stated methodology) gives χ²≈74.3, df=10, p≈6.5×10⁻¹², close
to but not bit-exact with the paper's stated 75.5/3.9×10⁻¹². The paper's
precise per-order combination method isn't independently reproducible
from the information available in this repo; the fixture now uses the
exactly-reproducible recomputed value as its own ground truth rather than
copying an unverifiable number. `FR_RECOMPUTE` does not gate on this
specific figure (only on the two headline p-values, which are exactly
reproducible).

**Project-status note**: the paper has progressed since this doc was
written — it's now in `papers/ready-for-submission/` (not
`in-progress/`), and includes a major **adversarial out-of-sample
replication** on 341 additional NASA Exoplanet Archive systems (order-1
Fibonacci rate 81.3%, p < 10⁻⁴⁸, Fisher combined p = 3.21×10⁻⁴⁶) not yet
reflected in this cert or doc. Independently reproduced that figure too:
`binomtest(113, 139, 2/9)` gives 2.554×10⁻⁴⁹, matching the paper's stated
2.55×10⁻⁴⁹ exactly. Worth extending this cert's fixture/catalogue to
cover the replication dataset in a future session.
