# Family [64] Flat-LR Sweep Comparator

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lambda_orbit=1.0
- [63] baseline lr=0.01; [64] flat lr in [0.015, 0.017, 0.019]
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline (lr=0.01) | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat lr=0.015 | 0.061876 | -0.005583 +- 0.000170 | -0.178 | 1.096 | 13.170674 |
| [64] flat lr=0.017 | 0.060221 | -0.007238 +- 0.000127 | -0.153 | 1.307 | 14.265413 |
| [64] flat lr=0.019 | 0.058926 | -0.008534 +- 0.000126 | -0.125 | 1.554 | 15.179989 |