# Family [63] vs [64] Matched LR=0.06

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lr=0.06, lambda_orbit=1.0
- Delta convention: `[64] - [63]` (negative recon delta means [64] better).

| Model | Recon mean | COSMOS z mean | SAT z mean | Status counts |
|---|---:|---:|---:|---|
| [63] baseline | 0.042441 | -0.636 | 0.932 | {'CONVERGED': 5} |
| [64] orbit-reg | 0.053911 | 2.889 | 8.553 | {'CONVERGED': 5} |

## Deltas ([64] - [63])

- Recon delta mean+-std: `0.011469 +- 0.000165`
- COSMOS z delta mean: `3.525`
- SAT z delta mean: `7.621`
- [64] reg_norm_final mean: `15.797696`