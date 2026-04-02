# Family [64] Flat-LR Sweep Comparator

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lambda_orbit=1.0
- [63] baseline lr=0.01; [64] flat lr in [0.035, 0.04, 0.045]
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline (lr=0.01) | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat lr=0.035 | 0.053983 | -0.013476 +- 0.000185 | 0.569 | 4.941 | 17.913715 |
| [64] flat lr=0.04 | 0.053521 | -0.013939 +- 0.000169 | 1.042 | 6.222 | 17.784444 |
| [64] flat lr=0.045 | 0.053341 | -0.014118 +- 0.000177 | 1.553 | 7.266 | 17.426782 |