# Family [194] QA_COGNITION_SPACE_MORPHOSPACE_CERT.v1

## One-line summary

Sole, Seoane et al.'s qualitative cognition-space morphospace (non-metric, clustered with voids) mapped to QA's exact, discrete, algebraically determined morphospace. Three clusters (basal/neural/human-AI) = three QA orbits. Voids are algebraically necessary, not contingent. Agency = |reachable set| / |total states|.

## Mathematical content

### Morphospace correspondence

| Sole et al. cluster | QA orbit | Size (mod-9) | Agency |
|---------------------|----------|-------------|--------|
| Basal cognition | Singularity | 1 | 1/81 |
| Neural cognition | Satellite | 8 | 8/81 |
| Human-AI / maximal | Cosmos | 72 | 72/81 |

### Algebraically necessary voids

No state exists between orbit types. The orbit membership is determined by v₃(f) where f = b²+be-e²:
- v₃ = 0 → Cosmos (exactly)
- v₃ = 2 → Satellite (exactly)
- v₃ = 4 → Singularity (exactly)

Voids are structural, not contingent on evolutionary history.

### Agency metric

QA_agency(b,e) = |{states reachable from (b,e) via QA dynamics}| / |S_m|

## Dependencies

- [191] QA_BATESON_LEARNING_LEVELS_CERT.v1

## Sources

- Sole, Seoane, Pla-Mauri, Bennett, Hochberg & Levin, "Cognition spaces" (arXiv:2601.12837, 2026)

## Verification Note (2026-07-04)

Independently checked the citation. **Sole, Seoane, Pla-Mauri, Bennett,
Hochberg & Levin, "Cognition spaces: natural, artificial, and hybrid"**
(arXiv:2601.12837, Jan 2026) confirmed real — all six author names match.
The paper's own abstract states it "introduces and examines three
cognition spaces—basal aneural, neural, and human–AI hybrid—and shows
that their occupation is highly uneven, with clusters of realized
systems separated by gaps," matching this cert's three-cluster/voids
claim almost verbatim (basal↔Singularity, neural↔Satellite, human-AI↔
Cosmos).

Validator already imports the shared QA primitives from
`tools/qa_kg/orbit_failure_enumeration.py` (cert [263]'s canonical
utility, also used by [193]) and performs genuine orbit-family
recomputation, not fixture-trusting. Spot-checked `MISSING_DIVISORS =
{2,3,4,6,12}`: correct — these are exactly the divisors of 24 excluding
the three realized orbit lengths {1,8,24}. No bugs found.
