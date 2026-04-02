# Family [63] vs [64] Matched Reconstruction Comparison

- Config: `n_epochs=20`, `n_samples=1000`, `lr=0.01`, seeds `[0, 1, 2, 3, 4]`
- Metric convention: `delta_recon = recon64 - recon63` (negative is better for [64])

| lambda | recon63 mean | recon64 mean | delta_recon mean±std | cosmos z63→z64 | sat z63→z64 | status64 |
|---|---:|---:|---:|---:|---:|---|
| 1e-05 | 0.088649 | 0.088649 | 0.000000 ± 0.000000 | 0.391 → 0.391 | -0.097 → -0.097 | STALLED:5 |
| 0.001 | 0.088649 | 0.088649 | 0.000000 ± 0.000000 | 0.391 → 0.392 | -0.097 → -0.097 | STALLED:5 |
| 0.1 | 0.088649 | 0.088655 | 0.000006 ± 0.000005 | 0.391 → 0.401 | -0.097 → -0.095 | STALLED:5 |
| 1 | 0.088649 | 0.088692 | 0.000043 ± 0.000009 | 0.391 → 0.485 | -0.097 → -0.085 | STALLED:5 |
| 10 | 0.088649 | 0.088958 | 0.000309 ± 0.000034 | 0.391 → 1.643 | -0.097 → 0.075 | STALLED:5 |

## Highlights

- Best reconstruction delta: `lambda=1e-05` with `delta_recon_mean=0.000000`.
- Positive delta means [64] reconstructed worse than [63] under this matched setup.
- z-score shifts are reported for coherence context, but this step optimizes for reconstruction parity/tradeoff.