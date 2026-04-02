# Matched [63] vs [64] Tight Lambda Bracket (lr=0.05)

- Lambdas: [0.2, 0.4, 0.7]
| lambda | Recon63 | Recon64 | Delta Recon | COSMOS Δz | SAT Δz | [64] reg_norm | status64 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.2 | 0.044659 | 0.046680 | 0.002021 +- 0.000053 | 0.267 | 1.332 | 27.615909 | {'CONVERGED': 5} |
| 0.4 | 0.044659 | 0.048587 | 0.003928 +- 0.000075 | 0.604 | 3.090 | 23.861320 | {'CONVERGED': 5} |
| 0.7 | 0.044659 | 0.051167 | 0.006509 +- 0.000095 | 1.533 | 5.593 | 19.818758 | {'CONVERGED': 5} |

- Best recon in bracket: lambda=0.2 recon=0.046680
- Best SAT delta in bracket: lambda=0.7 SATΔ=5.593