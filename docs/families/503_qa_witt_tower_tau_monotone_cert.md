# [503] QA Witt Tower τ-Monotone Discrimination Ladder Cert

## What this is

Machine-checkable certificate that the six Witt Tower empirical certs form a
monotone concordant discrimination ladder under the return-rank operator — tied
together by the physical principle that autocorrelation timescale τ determines
the degree of persistence the operator detects.

## Claim (narrow)

| Check | Claim |
|-------|-------|
| **WTM_1** | 6 domains declared: τ_rank ∈ {1,...,6} distinct, n_sig_milli > 0, cert_ids reference [491][495][490][494][492][493] |
| **WTM_2** | All C(6,2)=15 (τ_rank, n_sig_milli) pairs concordant — Kendall τ = 1 |
| **WTM_3** | Anti/null/structural split: EEG n_sig_milli < 1000; FX ∈ [1000,2000); rivers..SST ≥ 2000 |
| **WTM_4** | Ladder span > 6.0: max_n_sig_milli × 1000 > min_n_sig_milli × 6000 (integer check) |

## The Discrimination Ladder

| τ_rank | Domain | n_sig_ratio | n_sig_milli | Cert | Physical τ |
|--------|--------|-------------|-------------|------|-----------|
| 1 | EEG (interictal) | 0.72× | 720 | [491] | < 1 second |
| 2 | FX 1-min returns | 1.009× | 1009 | [495] | 1–10 minutes |
| 3 | River flow | 2.69× | 2690 | [490] | ~ 1 day |
| 4 | Precipitation | 3.05× | 3050 | [494] | 3–7 days |
| 5 | Temperature anomaly | 3.40× | 3400 | [492] | ~ weeks |
| 6 | SST (ocean) | 4.43× | 4430 | [493] | ~ months |

**n_sig_milli = n_sig_ratio × 1000** (integer milliunits; all arithmetic exact integers)

## Physical Mechanism

Hasselmann (1976) proved that ocean SST has autocorrelation timescale τ ~ months
because of the ocean's enormous heat capacity relative to the atmosphere. The return-rank
operator (n_signal/n_expected) amplifies this τ-hierarchy into a measurable discrimination
ladder: longer τ → more consecutive rank persistence → higher n_sig_ratio.

The three structural zones match the three QA orbit types:
- **Anti-persistent** (EEG, τ < 1s): neural refractory → forced mean-reversion after each spike
- **Null zone** (FX 1-min, n_sig_ratio ≈ 1): bid-ask bounce cancels persistence at 1-min scale
- **Structural persistence** (rivers → SST, τ ≥ 1 day): thermodynamic/hydraulic τ accumulates

## Monotone Concordance (WTM_2 detail)

15 pairs, all concordant:

| i | j | τᵢ < τⱼ | n_sig_millᵢ < n_sig_millⱼ |
|---|---|---------|--------------------------|
| EEG(1) | FX(2) | 1<2 ✓ | 720<1009 ✓ |
| EEG(1) | rivers(3) | 1<3 ✓ | 720<2690 ✓ |
| EEG(1) | precip(4) | 1<4 ✓ | 720<3050 ✓ |
| EEG(1) | temp(5) | 1<5 ✓ | 720<3400 ✓ |
| EEG(1) | SST(6) | 1<6 ✓ | 720<4430 ✓ |
| FX(2) | rivers(3) | 2<3 ✓ | 1009<2690 ✓ |
| ... | ... | ... | ... |
| temp(5) | SST(6) | 5<6 ✓ | 3400<4430 ✓ |

Kendall τ = 15 concordant / 15 total = **1.0**

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_witt_tower_tau_monotone_cert_v1/qa_witt_tower_tau_monotone_cert_validate.py` |
| Mapping ref | `qa_witt_tower_tau_monotone_cert_v1/mapping_protocol_ref.json` |
| PASS fixture | `fixtures/pass_tau_monotone_ladder.json` |
| FAIL: discordant pair | `fixtures/fail_discordant_pair.json` |
| FAIL: EEG not anti-persistent | `fixtures/fail_eeg_not_anti_persistent.json` |
| FAIL: insufficient span | `fixtures/fail_insufficient_span.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_tau_monotone_cert_v1
python3 qa_witt_tower_tau_monotone_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`, 1 PASS + 3 FAIL fixtures.

## QA Axiom Compliance

- **A1**: τ_rank ∈ {1,...,6} (integers, no zero)
- **T1**: n_sig_ratio = count/count (integer ratio); ordinal τ ranks are integers
- **T2**: n_sig_ratio declared as integer milliunits; continuous autocorr coefficient is observer projection, not QA state
- **S2**: no float state anywhere in the cert; all comparisons use integer arithmetic

## Primary Sources

- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8.
- Hasselmann, K. (1976). Stochastic climate models. Part I: Theory. *Tellus* 28(6):473–485. DOI:10.1111/j.2153-3490.1976.tb00696.x.

## Relation to other certs

- **[491]** QA Witt Tower EEG Anti-Persistence — τ_rank=1, n_sig_ratio=0.72×
- **[495]** QA Witt Tower 1-min FX Null — τ_rank=2, n_sig_ratio=1.009×
- **[490]** QA Witt Tower River Flow Persistence — τ_rank=3, n_sig_ratio=2.69×
- **[494]** QA Witt Tower Precipitation Return-Rank — τ_rank=4, n_sig_ratio=3.05×
- **[492]** QA Witt Tower Temperature Anomaly — τ_rank=5, n_sig_ratio=3.40×
- **[493]** QA Witt Tower Ocean SST Persistence — τ_rank=6, n_sig_ratio=4.43×

## Scope boundary

**The cert does NOT:**
- Claim QA causes the τ hierarchy
- Provide exact τ values (ordinal claim only)
- Certify the continuous autocorrelation coefficient values
- Extend to domains outside the 6 tested

**The cert DOES:**
- Certify that all 15 rank pairs are concordant (Kendall τ = 1)
- Tie 6 independent empirical certs into one falsifiable structural claim
- Establish the anti/null/structural three-zone split as a QA integer certificate
