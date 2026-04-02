# Family [64] Flat-LR Increase Comparator

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lambda_orbit=1.0
- [63] baseline lr=0.01; [64] flat lr in [0.011, 0.012]
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline (lr=0.01) | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat lr=0.011 | 0.066479 | -0.000980 +- 0.000060 | -0.236 | 0.789 | 10.371734 |
| [64] flat lr=0.012 | 0.065144 | -0.002315 +- 0.000136 | -0.222 | 0.853 | 11.173943 |

## Cross-LR Comparison ([64] flat)

- Recon shift (0.012 - 0.011): `-0.001335` (negative means 0.012 better).
- COSMOS z shift (0.012 - 0.011): `0.014`
- SAT z shift (0.012 - 0.011): `0.064`