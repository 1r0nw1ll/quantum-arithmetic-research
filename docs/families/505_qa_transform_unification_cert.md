# [505] QA Transform Unification Certificate

**Schema**: `QA_TRANSFORM_UNIFICATION_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_transform_unification_cert_v1/`
**Status**: PASS
**Added**: 2026-06-23
**Primary source**: Mühlbach P. et al. (2026) arXiv:2605.11589 — Peter-Weyl basis unifies DFT/DCT/WHT/KLT as matched-group transforms.

## Claim

QA's G-function G(b,e) = (b+e)² + e² provides a complete, non-degenerate matched-group channel bank across all three QA orbit classes, with three integer certificates:

**TU_1 — Matched-group bandwidth = 33**
Total independent frequency channels = 24 (Cosmos) + 8 (Satellite) + 1 (Singularity) = 33. For cyclic G = Z/n, the Peter-Weyl basis has n irreducible representations (each 1-dimensional for abelian groups). QA state space decomposes over G = Z/24Z × Z/8Z × Z/1Z; bandwidth = Σ|Gᵢ| per orbit class = 33.

**TU_2 — Three independent Cosmos sub-orbit G-channels**
Autocorrelation lag-0 values A[0] = Σᵢ G(σⁱ(b₀,e₀))² for the three Cosmos starting pairs:
- O1 = (1,1): A[0] = 744,693
- O2 = (1,3): A[0] = 658,293
- O3 = (1,4): A[0] = 574,269

All three are distinct integers → three independent, non-degenerate channel banks within the Cosmos class. In each sub-orbit, A[0] > max(A[1], …, A[23]) (lag-0 dominant; no self-similar sub-period).

**TU_3 — Cross-class energy isolation**
- Cosmos G-total (sum of G over all 72 Cosmos pairs) = 9,963
- Satellite G-total (sum over 8 Satellite pairs) = 1,377
- Singularity G(9,9) = 405

Strict integer ordering: 9,963 > 1,377 > 405. The three orbit classes occupy strictly disjoint energy levels: Cosmos high-band, Satellite mid-band, Singularity DC.

## Connection to arXiv:2605.11589

Mühlbach et al. prove that for any compact group G, every G-equivariant covariance matrix is block-diagonalized by the Peter-Weyl basis (irreducible representations of G). For cyclic G = Z/n, the Peter-Weyl basis is the DFT on Z/n — the matched-group transform is provably optimal (equals the KLT for G-stationary signals). The total bandwidth equals the sum of irrep counts across all component groups.

For QA mod-9:
- G = Z/24Z (Cosmos orbit class) × Z/8Z (Satellite orbit class) × Z/1Z (Singularity)
- Total bandwidth = 24 + 8 + 1 = 33 irreps
- Full parameter count: 24 × 3 + 8 + 1 = 81 = 9² (complete state space)

TU_2 establishes that the three Cosmos sub-orbits are genuinely independent (non-degenerate) channels — not compressed to a single 24-point DFT. TU_3 provides cross-class discrimination: the matched-group transform correctly assigns each orbit class to a disjoint energy band.

## Theorem NT Compliance

The DFT eigenvalues (24th and 8th roots of unity) are continuous observer projections over the discrete G-orbit structure. The channel structure is certified entirely via integer G-value arithmetic (sums, products, autocorrelations at integer lags). No float state; firewall crossed exactly twice.

## Companion Certs

- [500] QA Cosmos Chamber — G-sum arithmetic progression per sub-orbit
- [501] QA Algebraic Diversity Observer — G injective on each orbit; Z/24Z and Z/8Z minimal
- [503] QA Witt Tower tau-Monotone — empirical discrimination across 6 signal domains
- [504] QA Star-G Tensor — QA mod-9 as G-module; CRT Z/24Z ≅ Z/3Z × Z/8Z

## Fixtures

| File | Expected | Checks |
|------|----------|--------|
| `pass_transform_unification.json` | PASS | All three TU claims |
| `fail_wrong_bandwidth.json` | FAIL | TU_1: total_channel_count=32 (satellite=7 not 8) |
| `fail_degenerate_cosmos_channels.json` | FAIL | TU_2b: two Cosmos A[0] values equal (744693 twice) |
| `fail_energy_order_violated.json` | FAIL | TU_3b: cosmos_g_total=1377 < satellite_g_total=9963 (swapped) |
