# Cert [490]: QA Witt Tower River Streamflow Return-Rank Autocorrelation (Persistence)

**Family ID**: 490
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_streamflow_return_rank_cert_v1/`

## Claim

The return-rank a=b+2e≤6 operator reveals **positive autocorrelation (persistence)** in river recession — the structural opposite of crash-reversion. After 2 consecutive fast-recession days (log-flow-change in bottom 7%), the next day continues fast recession:

| Gauge | Location | n_signal | Expected | Excess | pers_p |
|-------|----------|---------|---------|--------|--------|
| Potomac | Little Falls, MD | 604 | ~208 | −16.15 log-% | 0.0 |
| Hudson | Green Island, NY | 504 | ~208 | −12.50 log-% | 0.0 |
| Missouri | Toston, MT | 454 | ~208 | −3.59 log-% | 0.0 |
| Eel | Scotia, CA | 675 | ~208 | −13.40 log-% | 0.0 |

4/4 negative. Pooled: **−11.95 log-%**. Persistence permutation p = 0.0 for all (0/5000 null shuffles as negative as signal). This is the **first non-finance cert** for the return-rank operator; it certifies a null on crash-reversion and a positive finding on persistence.

## Two Findings

**Finding 1 — Persistence**: After 2 consecutive fast-recession days, recession continues. The mean next-day log-flow-change is −12 to −16 log-% below the unconditional mean. The restoring force (precipitation) operates on timescales >> 1 day; within-day-type autocorrelation dominates.

**Finding 2 — Clustering**: n_signal is 2.2–3.2× the ~208 days expected under independence (P(a≤6) = 16/729 = 2.19%). This excess is direct evidence of positive autocorrelation: consecutive (b,e) pairs are correlated, so (low, low) pairs co-occur far more often than chance.

## Structural Contrast with Finance

| System | Mechanism | signal_excess | Direction |
|--------|-----------|--------------|-----------|
| Altcoins [487] | Mean-reverting microstructure | +1.34 to +2.09%/day | Crash-reversion |
| BTC/ETH [482] | Mean-reverting microstructure | +0.85 to +1.77%/day | Crash-reversion |
| Equity [488]/[489] | Mean-reverting at multi-year scale | +0.36 to +0.39%/day | Crash-reversion |
| **Rivers [490]** | **Maillet exponential recession** | **−3.6 to −16.2 log-%/day** | **Persistence** |
| Non-crypto [486] | No dominant microstructure | ~0 (null) | Null |

The operator **discriminates autocorrelation sign**: systems with negative autocorrelation after extreme events show crash-reversion; systems with positive autocorrelation (recession persistence) show the opposite.

## Physical Mechanism

River discharge after a flood peak follows Maillet's law: Q(t) = Q₀·e^(−t/τ). In log-return space, fast-recession days cluster because the recession timescale τ is days to weeks — far longer than one day. So after 2 fast-recession days, the next day is also a fast-recession day with high probability. The bottom-7% return-rank signal catches days embedded within an ongoing recession event, not the start of recovery.

The contrast with equity: equity crashes (panic selling) self-correct in 1–2 days via price discovery. River recessions self-correct in weeks via groundwater recharge. The timescale mismatch explains the sign flip.

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | Potomac excess < −1.0 log-% | PASS | −16.15 |
| C2 | n_negative == 4/4 | PASS | 4 |
| C3 | Missouri excess negative | PASS | −3.59 |
| C4 | Eel persistence_p < 0.001 | PASS | 0.0 |
| C5 | pooled excess < −1.0 log-% | PASS | −11.95 |
| C6 | n_signal elevated (>1.5× expected) | PASS | 2.2–3.2× |

## Primary Sources

- Maillet E (1905). *Essais d'hydraulique souterraine et fluviale*. Paris: Hermann. (Exponential recession law)
- Brutsaert W & Nieber J (1977). Regionalized drought flow hydrographs from a mature glaciated plateau. *Water Resources Research* 13(3):637-643. doi:10.1029/WR013i003p00637

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [482]: BTC/ETH return-rank; operator definition
- Cert [488]: US equity return-rank; crash-reversion for contrast
- Cert [486]: Non-crypto null; operator scope boundary
