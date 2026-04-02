# Family [180] QA_H_NULL_MODULARITY_CERT.v1

## One-line summary

H-null chromogeometric modularity: replacing the standard graph modularity null model with QA invariant H=C+F improves community detection on hub-dominated networks (Les Miserables ARI +0.050). Tier 2 — topology-specific, not universal.

## Mathematical content

### H-null model

Standard modularity uses null model P_ij = k_i * k_j / (2m) where k = degree. The QA H-null replaces this with H(k_i, k_j) = C + F where:
- C = 2 * k_j * (k_i + k_j) = green quadrance (Qg)
- F = k_i * (k_i + 2 * k_j) = red quadrance (Qr)
- H = C + F captures both symmetric AND asymmetric degree relationships

### Key identity

H / X = b/e + 4 + 2e/b grows linearly with degree asymmetry r = b/e. The standard null X = k_i * k_j is purely multiplicative; H adds the additive component through d = k_i + k_j.

### Chromogeometric connection

H = C + F = Qg + Qr (green + red quadrance from Wildberger chromogeometry). The identity C*C + F*F = G*G connects to the blue quadrance G = d*d + e*e.

### Benchmark results

| Graph | N | H-null ARI | Standard ARI | Delta |
|-------|---|-----------|-------------|-------|
| les_miserables | 77 | 0.638 | 0.588 | +0.050 |
| football | 115 | 0.824 | 0.824 | 0.000 |
| karate | 34 | — | — | -0.479 |

### Honest negatives

- Wins: 1/10 graphs (les_miserables)
- Ties: 3/10 (football, caveman, davis_women)
- Losses: 6/10 (karate, powerlaw variants, windmill)
- Hub locality correlation: r = -0.36 (NEGATIVE — higher locality correlates with H losing, driven by karate)
- Effect is topology-specific to hub-dominated networks with community-internal hubs

### Mechanism

Hubs (high k_i) get penalized more heavily by the asymmetric F term, so the modularity residual A_ij - H/norm is better calibrated for networks where hubs concentrate within communities.

## Axiom compliance

- A1: Degree values are positive integers
- A2: d = b + e, a = b + 2e derived correctly
- T2: ARI scores and community assignments are observer projections
- S1: No `**2` — uses d*d, e*e
- S2: All QA state is integer

## Scripts

- `qa_lab/qa_graph/` — analysis scripts for H-null model benchmarking
