# Family [64] Flat-LR Sweep Comparator

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lambda_orbit=1.0
- [63] baseline lr=0.01; [64] flat lr in [0.021, 0.023, 0.025]
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline (lr=0.01) | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat lr=0.021 | 0.057871 | -0.009589 +- 0.000152 | -0.090 | 1.833 | 15.925476 |
| [64] flat lr=0.023 | 0.057000 | -0.010460 +- 0.000169 | -0.044 | 2.148 | 16.521291 |
| [64] flat lr=0.025 | 0.056248 | -0.011211 +- 0.000185 | 0.017 | 2.508 | 16.988238 |