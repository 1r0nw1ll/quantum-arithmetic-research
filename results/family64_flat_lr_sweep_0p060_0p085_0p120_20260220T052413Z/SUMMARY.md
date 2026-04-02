# Family [64] Flat-LR Sweep Comparator

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lambda_orbit=1.0
- [63] baseline lr=0.01; [64] flat lr in [0.06, 0.085, 0.12]
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline (lr=0.01) | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat lr=0.06 | 0.053911 | -0.013549 +- 0.000160 | 2.889 | 8.553 | 15.797696 |
| [64] flat lr=0.085 | 0.056091 | -0.011368 +- 0.000176 | 3.666 | 8.053 | 13.097742 |
| [64] flat lr=0.12 | 0.059117 | -0.008343 +- 0.000222 | 4.922 | 6.963 | 10.254249 |