# Family [187] QA_KEELY_AGGREGATION_CERT.v1

## One-line summary

Keely's 5 aggregation/disintegration laws (Laws 3, 12, 28, 34, 35) map to QA state composition and decomposition: coupling tension modifies period, orbit density governs aggregation, discord drives dissociation, and synthesis is deterministic — Category 4 of the Vibes 5-category framework.

## Mathematical content

### Keely's aggregation/disintegration laws

| Law # | Name (svpwiki) | QA mapping |
|-------|---------------|------------|
| 3 | Law of Chemical Affinity | coupling tension = f-value proximity within orbit |
| 12 | Law of Atomic Dissociation | discord (cross-orbit) → dissociation = orbit exit |
| 28 | Law of Chemical Transposition | state composition: (b₁,e₁) + (b₂,e₂) mod M |
| 34 | Law of Chemical Morphology | orbit density = #states / orbit_period |
| 35 | Law of Atomic Pitch | deterministic synthesis: initial state fully determines trajectory |

### Category 4: aggregation/disintegration

These laws describe **composition and decomposition** of vibratory states. In QA, this corresponds to how states combine (mod-arithmetic composition), what determines binding (same-orbit coupling), and what causes separation (cross-orbit discord).

### Key structures

- **Coupling tension**: states with similar f-values within an orbit exhibit stronger coupling
- **Orbit density**: cosmos = 72/24 = 3 states per period step; satellite = 8/8 = 1; singularity = 1/1 = 1
- **Discord → dissociation**: cross-orbit interactions cannot sustain coupling; states separate
- **Deterministic synthesis**: QA step is fully deterministic — no randomness in composition

## Checks

| ID | Description |
|----|-------------|
| KAG_1 | schema_version == 'QA_KEELY_AGGREGATION_CERT.v1' |
| KAG_LAWS | all 5 law numbers present: {3,12,28,34,35} |
| KAG_COUPLE | coupling tension computed from f-value proximity |
| KAG_DENSITY | orbit densities: cosmos=3, satellite=1, singularity=1 |
| KAG_DISSOC | cross-orbit pairs demonstrate dissociation |
| KAG_SYNTH | deterministic trajectory from initial state verified |
| KAG_W | ≥3 witnesses (composition, density, dissociation) |
| KAG_F | ≥1 falsifier (incorrect density rejected) |

## Source grounding

- **svpwiki.com/Law+of+Chemical+Affinity**: "the force of aggregation"
- **svpwiki.com/Law+of+Atomic+Dissociation**: "the breaking apart of aggregated masses"
- **Ben Iverson**: QA deterministic dynamics, orbit density analysis
- **Dale Pond / Vibes**: SVP consultant AI; Category 4 classification (2026-04-03)

## Connection to other families

- **[153] Keely Triune**: orbit structure underlies all aggregation dynamics
- **[184] Keely Structural Ratio**: f-value governs coupling tension
- **[185] Keely Sympathetic Transfer**: reachability determines aggregation possibility

## Fixture files

- `fixtures/kag_pass_composition.json` — state composition with coupling tensions and densities
- `fixtures/kag_fail_bad_density.json` — falsifier with incorrect orbit density
