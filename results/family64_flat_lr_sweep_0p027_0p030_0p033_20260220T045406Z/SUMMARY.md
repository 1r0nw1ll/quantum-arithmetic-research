# Family [64] Flat-LR Sweep Comparator

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lambda_orbit=1.0
- [63] baseline lr=0.01; [64] flat lr in [0.027, 0.03, 0.033]
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline (lr=0.01) | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat lr=0.027 | 0.055620 | -0.011839 +- 0.000191 | 0.092 | 2.912 | 17.341981 |
| [64] flat lr=0.03 | 0.054857 | -0.012602 +- 0.000192 | 0.237 | 3.617 | 17.698205 |
| [64] flat lr=0.033 | 0.054286 | -0.013173 +- 0.000192 | 0.415 | 4.410 | 17.876122 |