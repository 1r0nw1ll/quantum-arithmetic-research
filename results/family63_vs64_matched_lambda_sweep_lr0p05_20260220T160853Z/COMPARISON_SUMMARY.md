# Matched [63] vs [64] Lambda Sweep (lr=0.05)

- Config: seeds [0, 1, 2, 3, 4], n_epochs=60, n_samples=1000, fixed lr=0.05
- Delta convention: `[64] - [63]` (negative recon delta means [64] better recon).

| lambda | Recon63 | Recon64 | Delta Recon | COSMOS Δz | SAT Δz | [64] reg_norm | status64 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1e-05 | 0.044659 | 0.044658 | -0.000000 +- 0.000002 | 0.000 | 0.000 | 32.793562 | {'CONVERGED': 5} |
| 0.001 | 0.044659 | 0.044670 | 0.000011 +- 0.000002 | 0.001 | 0.005 | 32.763247 | {'CONVERGED': 5} |
| 0.1 | 0.044659 | 0.045685 | 0.001026 +- 0.000038 | 0.121 | 0.599 | 29.982364 | {'CONVERGED': 5} |
| 1.0 | 0.044659 | 0.053404 | 0.008745 +- 0.000132 | 2.627 | 7.119 | 16.934871 | {'STALLED': 4, 'CONVERGED': 1} |
| 10.0 | 0.044659 | 0.074207 | 0.029548 +- 0.000994 | 10.246 | 0.792 | 1.200069 | {'STALLED': 5} |

## Picks

- Best [64] recon mean at lambda=1e-05 with recon=0.044658.
- Best [64] SAT z mean at lambda=1.0 with SAT z=7.963.