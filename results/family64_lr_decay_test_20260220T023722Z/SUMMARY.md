# Family [64] LR-Decay Test (Matched)

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lr=0.01, lambda_orbit=1.0
- Decay schedule: epoch 1 -> 0.01, epoch 8 -> 0.001
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean±std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline | 0.067459 | 0.000000 ± 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat LR | 0.068050 | 0.000591 ± 0.000047 | -0.246 | 0.725 | 9.460326 |
| [64] LR decay | 0.093280 | 0.025821 ± 0.000220 | 0.630 | -0.246 | 2.525909 |

## Key Comparison

- Decay vs flat reconstruction shift: `0.025230 ± 0.000219` (negative means decay improves reconstruction).
- COSMOS z shift (decay - flat): `0.876`
- SAT z shift (decay - flat): `-0.971`