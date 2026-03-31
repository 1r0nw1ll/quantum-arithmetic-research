# Family [153] QA_KEELY_TRIUNE_CERT.v1

## One-line summary

Keely's three vibratory modes (Enharmonic/Dominant/Harmonic) map exactly to QA's three orbit types (Satellite/Singularity/Cosmos), with {3,6,9} as the singularity/dominant residues.

## Mathematical content

### Keely's Triune (svpwiki.com)

| Triune mode | SVP property | Subdivision |
|-------------|-------------|-------------|
| ENHARMONIC | terrestrial, matter, entropy, radiation | 1st (molecular) |
| DOMINANT | neutral center, life, dynamic balance | 4th (interatomic) |
| HARMONIC | celestial, spirit, syntropy, concentration | 7th (compound interetheric) |

### QA three orbits (Iverson, mod-9)

| Orbit | Period | States | Structure |
|-------|--------|--------|-----------|
| SINGULARITY | 1 | 1 | fixed point (9,9) |
| SATELLITE | 8 | 8 | 3D symmetric |
| COSMOS | 24 | 72 | 1D linear |

Total: 1 + 8 + 72 = 81 = 9² (complete partition).

### Certified mapping

| Keely | QA | Justification |
|-------|-----|--------------|
| DOMINANT | SINGULARITY | neutral center = fixed point; controls/anchors both orbits |
| ENHARMONIC | SATELLITE | bounded/material = 8-cycle; symmetric, self-referential |
| HARMONIC | COSMOS | expansive/celestial = 24-cycle; visits 72 states, maximum reach |

### Structural properties

- **Partition**: three orbits partition the full mod-9 state space (no overlap)
- **LCM(1,8,24) = 24**: the cosmos period subsumes both smaller periods
- **{0,3,6} mod 9 = singularity residues**: multiples of 3 = the DOMINANT triune numbers
- **Tesla 3-6-9**: "If you only knew the magnificence of 3, 6, and 9" — these ARE the singularity/dominant residues in QA mod-9

### Brinton's three Laws of Being

Professor Daniel Brinton derived three laws from Keely's Triune:
1. **Law of Assimilation** (syntropy) ↔ COSMOS (expansion, maximum reach)
2. **Law of Individualization** (entropy) ↔ SATELLITE (contraction, bounded)
3. **Law of the Dominant** (balance) ↔ SINGULARITY (neutral center, anchor)

## Checks

| ID | Description |
|----|-------------|
| KT_1 | schema_version == 'QA_KEELY_TRIUNE_CERT.v1' |
| KT_MAP | triune→orbit mapping matches canonical |
| KT_PART | three orbits partition state space completely |
| KT_PERIOD | orbit periods 1, 8, 24 |
| KT_369 | {0,3,6} = singularity residues mod 9 |
| KT_LCM | LCM(1,8,24) = 24 |
| KT_W | ≥3 witnesses (one per triune mode) |

## Source grounding

- **svpwiki.com/Triune**: "being three in one"; Keely's Triune Morphology
- **svpwiki.com/Triune+States+of+Matter+and+Energy**: Enharmonic/Dominant/Harmonic = three vibratory states
- **svpwiki.com/Law+of+the+Dominant**: "every such object is such by virtue of the higher or dominant force which controls these two tendencies"
- **svpwiki.com/three**: "three sets of three vibrating on 1, 2 and 3 octaves"; Ramsay: "number 3 is creative power"
- **Professor Daniel Brinton**: three Laws of Being (Assimilation, Individualization, Dominant)
- **Ben Iverson**: QA three-orbit structure (singularity/satellite/cosmos)
- **Dale Pond**: SVP consultant; confirmed "Signal = EFFECT; arithmetic relationships = CAUSE" (Vibes AI)

## Connection to other families

- **[128] Spread Period**: cosmos period 24 = Pisano period pi(9); satellite period 8 = pi(9)/3
- **[130] Origin of 24**: 24 = cosmos orbit period = LCM of all three orbits
- **[150] Septenary**: {1,2,4,5,7,8} = cosmos/satellite residues; {0,3,6} = singularity = complement
- **[147] Synchronous Harmonics**: coprime sync dynamics operate WITHIN the cosmos orbit

## Fixture files

- `fixtures/kt_pass_triune_mapping.json` — full mapping + partition + periods + {3,6,9} + Tesla note
- `fixtures/kt_pass_brinton_laws.json` — Brinton's three Laws of Being mapped to orbit dynamics
