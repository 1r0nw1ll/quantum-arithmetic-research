# Family [63] vs [64] Matched LR=0.05

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, lr=0.05, lambda_orbit=1.0
- Delta convention: `[64] - [63]` (negative recon delta means [64] better).

| Model | Recon mean | COSMOS z mean | SAT z mean | Status counts |
|---|---:|---:|---:|---|
| [63] baseline | 0.044659 | -0.579 | 0.844 | {'CONVERGED': 5} |
| [64] orbit-reg | 0.053404 | 2.048 | 7.963 | {'STALLED': 4, 'CONVERGED': 1} |

## Deltas ([64] - [63])

- Recon delta mean+-std: `0.008745 +- 0.000132`
- COSMOS z delta mean: `2.627`
- SAT z delta mean: `7.119`
- [64] reg_norm_final mean: `16.934871`