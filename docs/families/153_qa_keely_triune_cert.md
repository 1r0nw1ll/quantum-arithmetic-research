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
- **{3,6,9} mod 9 = singularity residues** (corrected 2026-07-06, A1 no-zero convention): multiples of 3 = the DOMINANT triune numbers
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
- **Audit note (2026-07-04)**: "Vibes" is Dale Pond's own AI tool (confirmed by Will), not an independent human co-reviewer — see `private/keely_40_laws_classification.md` provenance note. The triune↔orbit mapping's underlying QA arithmetic (partition, periods, LCM) is independently checkable and holds; the interpretive confirmation from Vibes should not be weighted as external peer review.

## Connection to other families

- **[128] Spread Period**: cosmos period 24 = Pisano period pi(9); satellite period 8 = pi(9)/3
- **[130] Origin of 24**: 24 = cosmos orbit period = LCM of all three orbits
- **[150] Septenary**: {1,2,4,5,7,8} = cosmos/satellite residues; {0,3,6} = singularity = complement
- **[147] Synchronous Harmonics**: coprime sync dynamics operate WITHIN the cosmos orbit

## Fixture files

- `fixtures/kt_pass_triune_mapping.json` — full mapping + partition + periods + {3,6,9} + Tesla note
- `fixtures/kt_pass_brinton_laws.json` — Brinton's three Laws of Being mapped to orbit dynamics

## Verification Note (2026-07-06)

Independently reconfirmed the partition (1+8+72=81=9²), orbit periods
(1,8,24), and LCM(1,8,24)=24 — all correct. The Keely-quote provenance
(svpwiki.com sources, the "Vibes" AI-tool caveat) was already audited
2026-07-04 (see the existing "Audit note" in Source grounding above);
not re-litigated here.

**Found and fixed the same A1 no-zero bug already found in [150]/[192]**:
both fixtures declared `singularity_residues_mod9: [0,3,6]`, and the
validator's own `SINGULARITY_RESIDUES` constant was `frozenset({0,3,6})`
— even though the *same validator's own docstring* already correctly
stated "{3,6,9} mod 9 = triune numbers," and one fixture's own
`tesla_369_note` already correctly wrote "{3,6,9≡0}". Fixed the constant
and both fixtures to `{3,6,9}`, matching this project's A1 no-zero
convention (QA states are {1,...,9}, never {0,...,8}) and every other
cert's singularity representation. Verified the hardened check rejects
a reintroduced `[0,3,6]`.

**Found and fixed a second, independent bug in `kt_pass_brinton_laws.json`**:
a witness claimed "three primes {2,3,5} always present in QA QN products
(all products divisible by 6=2×3)" — self-contradictory (6=2×3 has no
factor of 5!) and factually wrong: exhaustively checked 156 primitive
`(b,e)` pairs and found 58 with products *not* divisible by 5, including
the fundamental `(1,1,2,3)` itself (product=6, not divisible by 5). Only
2 and 3 are guaranteed factors (matching the correctly-established "QN
products divisible by 6" fact from [137]/[147]/[148], not "by 30"). This
claim wasn't checked by any validator field (free-text witness). Fixed
to state the correct, narrower claim.

`--self-test` passes on both fixtures after all fixes.
