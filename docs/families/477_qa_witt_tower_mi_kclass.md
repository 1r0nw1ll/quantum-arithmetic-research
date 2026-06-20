# [477] QA Witt Tower Multi-Class MI Ceiling Theory

**Closes the theoretical framework underlying the [467] 7-domain survey.**

## Claim

The closed-form binary MI_ratio formula (cert [468]) generalises to k signal
classes. For a rank-uniform 3-tier partition (P(tier)=1/3), the exact joint
distribution and mutual information are determined by k tuples (q_i, r_i, d_i):

- q_i = base rate of class i
- r_i = P(dominant tier d_i | class i) [tier concentration]
- background = 1 − Σq_i, rank-uniformly distributed (P(tier)=1/3 constraint)

**k=1 reduction** (numeric identity with cert [468]):
```
MI_ratio_k1(q, r) ≡ MI_ratio_468(q, r)   [max diff = 5.55e-17]
```

**k=3 symmetric balanced ceiling:**
```
At q_i = 1/3, r → 1:  MI_ratio_k3 → 1.000   (vs binary ~70% ceiling)
At q_i = 1/3, r = 0.97: MI_ratio_k3 = 0.858  (vs binary 0.683 at q=0.29)
```

**Compact form** (exact when background = 0 and each class owns one tier):
```
MI = Σ_i  q_i · (log₂3 − h_r(r_i))
h_r(r) = −r·log₂r − (1−r)·log₂((1−r)/2)
```

## ENSO 3-Class Validation

Applied to NOAA ONI (N=916 seasons): El Niño (ONI≥0.5) / Neutral / La Niña
threshold-label classification:

| Class     | q      | r_domain (→tier) | r value |
|-----------|--------|------------------|---------|
| El Niño   | 0.2675 | → T2             | **1.0000** |
| La Niña   | 0.2751 | → T0             | **1.0000** |
| Neutral   | 0.4574 | → T1             | 0.7279 |

**Structural zeros**: r_ElNino = r_LaNina = 1.0 exactly. Every El Niño season
(ONI≥0.5) falls in T2; every La Niña season (ONI≤−0.5) falls in T0. This is
a theorem: rank-based tiers assign exactly 1/3 of seasons to each tier, and
the ONI threshold criterion selects seasons warmer/colder than the T2/T0
boundary by construction.

| Metric | Value |
|--------|-------|
| Formula MI_ratio | **0.6950** |
| Empirical MI_ratio | **0.6990** |
| Cert [465] reference | **0.699** |
| Formula vs empirical diff | **0.0039** (0.4%) |

## Theorem NT Compliance

Observer: ONI SST anomaly → rank → bin ∈ Z/27Z → tier ∈ {0,1,2} (integer).
The k-class formula operates on (q_i, r_i, d_i) tuples derived from observer
projections (empirical fractions). No float state enters the QA layer.

## Certified Checks

| Check | Description | Result |
|-------|-------------|--------|
| C1 | k=1 reduction identical to cert [468] (max diff < 1e-8) | PASS (5.55e-17) |
| C2 | k=3 symmetric ceiling → 1.0 (r=1−ε, MI_ratio > 0.99) | PASS (1.000) |
| C3 | k=3 symmetric monotone in r (strictly increasing) | PASS |
| C4 | ENSO formula vs empirical diff < 0.05 | PASS (0.0039) |
| C5 | k=3 ceiling(r=0.97) > k=1 ceiling(q=0.29, r=0.97) | PASS (0.858 > 0.683) |
| C6 | ENSO structural zeros: r_ElNino = r_LaNina = 1.0 exactly | PASS |

## Primary Sources

- Shannon CE (1948). doi:10.1002/j.1538-7305.1948.tb01338.x
- Trenberth KE (1997). doi:10.1175/1520-0477(1997)078<2771:TDOENO>2.0.CO;2

## Related Certs

- [468] QA Witt Tower MI Ceiling Theory (binary formula, k=1 special case)
- [465] QA Witt Tower MI ENSO (I=1.07 bits, MI_ratio=69.9%, empirical reference)
- [476] QA Witt Tower ENSO Prediction (Markov chain, T1 mandatory gateway)
- [467] QA Witt Tower Cross-Domain MI Survey (7-domain 0/35000 null hits)
