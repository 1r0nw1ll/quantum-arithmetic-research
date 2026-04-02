# Family [64] Schedule Probe: epoch30 0.01->0.008

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lr=0.01, lambda_orbit=1.0
- Step schedule: epoch 1 -> 0.01, epoch 30 -> 0.008
- Reconstruction delta convention: `recon(model) - recon([63])` (negative is better than [63])

| Model | Recon mean | Delta vs [63] mean+-std | COSMOS z mean | SAT z mean | reg_norm_final mean |
|---|---:|---:|---:|---:|---:|
| [63] baseline | 0.067459 | 0.000000 +- 0.000000 | -0.321 | 0.510 | n/a |
| [64] flat LR | 0.068050 | 0.000591 +- 0.000047 | -0.246 | 0.725 | 9.460326 |
| [64] step@30 | 0.070092 | 0.002633 +- 0.000115 | -0.242 | 0.655 | 8.388996 |

## Key Comparison

- Step vs flat reconstruction shift: `0.002042 +- 0.000076` (negative means step schedule improves reconstruction).
- COSMOS z shift (step - flat): `0.004`
- SAT z shift (step - flat): `-0.070`