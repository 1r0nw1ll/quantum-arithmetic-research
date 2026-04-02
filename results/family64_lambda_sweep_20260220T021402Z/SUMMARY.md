# Family [64] Lambda Sweep Summary

- Config: `n_epochs=20`, `n_samples=1000`, `lr=0.01`, seeds `0..4`
- Source reports: `report_lambda_*.json`

| lambda | COSMOS coherence (mean±std) | COSMOS z (mean±std) | SAT coherence (mean±std) | SAT z (mean±std) | reg_norm_final (mean±std) | statuses |
|---|---:|---:|---:|---:|---:|---|
| 1e-5 | 0.000894 ± 0.000007 | 0.391422 ± 0.642869 | 0.000890 ± 0.000025 | -0.096605 ± 0.804427 | 2.730205 ± 0.011324 | STALLED:5 |
| 1e-3 | 0.000894 ± 0.000007 | 0.391510 ± 0.642882 | 0.000890 ± 0.000025 | -0.096593 ± 0.804432 | 2.730101 ± 0.011324 | STALLED:5 |
| 1e-1 | 0.000895 ± 0.000007 | 0.400512 ± 0.643896 | 0.000891 ± 0.000025 | -0.095430 ± 0.804492 | 2.719644 ± 0.011285 | STALLED:5 |
| 1 | 0.000902 ± 0.000007 | 0.485276 ± 0.653558 | 0.000898 ± 0.000024 | -0.084543 ± 0.805090 | 2.626386 ± 0.010941 | STALLED:5 |
| 10 | 0.000950 ± 0.000004 | 1.642940 ± 0.762797 | 0.000948 ± 0.000013 | 0.074905 ± 0.790863 | 1.851849 ± 0.007951 | STALLED:5 |

## Quick Observations

- Increasing `lambda_orbit` monotonically increased COSMOS/SAT coherence means in this range.
- Best aggregate COSMOS z-score mean: `1.642940` at `lambda=10`.
- Final regularizer norm decreased strongly with higher `lambda_orbit` (`~2.73` -> `~1.85`).
- All runs ended with `STALLED` status under current convergence criterion.
- No per-seed run achieved both COSMOS and SATELLITE `z > 2` at 20 epochs.