<!-- PRIMARY-SOURCE-EXEMPT: CODATA 2018 (Mohr, Newell, Taylor, Tiesinga). DOI: 10.1103/RevModPhys.93.025010. Particle Data Group (2022). Review of Particle Physics. PTEP 2022, 083C01. DOI: 10.1093/ptep/ptac097. Briddell, D. (2024). Field Structure Theory. -->

# [290] QA Particle Mass Cosmos Orbit

**Cert family**: `qa_particle_mass_cosmos_cert_v1`
**Status**: PASS
**Depends on**: [279] QA Orbit Access Theorem

## Primary Sources

- Mohr, P.J., Newell, D.B., Taylor, B.N., & Tiesinga, E. (2021). CODATA recommended values of the fundamental physical constants: 2018. *Reviews of Modern Physics*, 93(2), 025010. DOI: 10.1103/RevModPhys.93.025010
- Particle Data Group (2022). Review of Particle Physics. *Progress of Theoretical and Experimental Physics*, 2022, 083C01. DOI: 10.1093/ptep/ptac097
- Briddell, D. (2024). Field Structure Theory. (FST constellation values)

## Claim

For every standard particle mass M in electron-mass units (integer-rounded, CODATA 2018) and every FST constellation value, every Pythagorean triple (d, e) satisfying d²−e²=M (F-interpretation) or 2de=M (C-interpretation) has orbit class **cosmos** under `qa_step` mod 24.

**Corollaries**:
- Masses ≡2 mod 4 have zero F-triples (structural impossibility: d²−e²=(d+e)(d−e) requires both factors same parity, excluding 2 mod 4) but ≥1 C-triple
- Proton (1836 mₑ) is **C-primary**: zero primitive F-triples, ≥1 primitive C-triples
- Singularity and satellite blockade is total: zero violations across 38 distinct mass values and 100+ triples

## Particle Masses Covered (CODATA 2018, electron-mass units)

| Particle | Mass (mₑ) | mod 4 | F-triples | C-triples |
|---|---|---|---|---|
| electron | 1 | 1 | 0 | 0 |
| muon | 207 | 3 | ≥1 | ≥1 |
| tau | 3477 | 1 | ≥1 | ≥1 |
| pion⁺ | 273 | 1 | ≥1 | ≥1 |
| pion⁰ | 264 | 0 | ≥1 | ≥1 |
| kaon⁺ | 966 | **2** | 0 | ≥1 |
| kaon⁰ | 974 | **2** | 0 | ≥1 |
| eta | 1072 | 0 | ≥1 | ≥1 |
| proton | 1836 | 0 | 0 (prim) | ≥1 (prim) |
| neutron | 1839 | 3 | ≥1 | ≥1 |
| alpha | 7294 | **2** | 0 | ≥1 |
| … | … | … | … | cosmos |

## FST Constellation Values Covered (Briddell 2024)

Powers of 3 (STF_1 through STF_8), structor=6, top_cluster=378, lambda_diff=351, x2_729=1458, x2_lambda=4374, x2_proton=3672, factor_17=17.

## Orbit Classifier (mod 24)

Using the divisor-shortcut theorem from cert [279]: let b̄ = d mod 24 (mapped to {1..24}), ē = e mod 24.

- **Singularity**: b̄=24 AND ē=24
- **Satellite**: 8|b̄ AND 8|ē (sat_div = 24//3 = 8)
- **Cosmos**: all other pairs

This is exact for m=24 since gcd(24,5)=1.

## Gates

| Gate | Description | Result |
|---|---|---|
| PMC_1 | Cosmos monopoly: all 38 values Cosmos-only in both F and C | PASS |
| PMC_2 | ≡2 mod 4 values: 0 F-triples, ≥1 C-triple | PASS |
| PMC_3 | Every value >1 has ≥1 triple in F or C | PASS |
| PMC_4 | Proton (1836): 0 primitive F-triples, ≥1 primitive C-triples | PASS |
| PMC_5 | Singularity and satellite blockade total | PASS |

## Theorem NT Compliance

Integer arithmetic throughout. Orbit class determined by exact integer divisibility (`b%8==0 and e%8==0`). No float feedback into QA layer. Factorization by trial division (primes ≤47 + remainder). Divisor enumeration by exact multiplicative reconstruction.

## Scope

- Covers integer-rounded masses only (NOT continuous mass values)
- Does NOT cover W/Z/Higgs bosons (masses >10⁵ mₑ, outside trial-division range)
- Does NOT certify neutrino masses (sub-electron-mass scale)
- Does NOT claim physical interpretation of the cosmos monopoly — this is a number-theoretic structural result
