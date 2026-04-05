# Family [184] QA_KEELY_STRUCTURAL_RATIO_CERT.v1

## One-line summary

Keely's 8 structural ratio laws (Laws 2, 4, 9, 10, 18, 27, 29, 33) map to QA modular invariants: pitch as f-value, period divisibility 1|8|24, concordance as orbit coupling, and chromogeometry quadrances — Category 1 of the Vibes 5-category framework.

## Mathematical content

### Keely's structural ratio laws

| Law # | Name (svpwiki) | QA mapping |
|-------|---------------|------------|
| 2 | Law of Corporeal Vibrations | f-value = b² + be − e² as vibratory pitch |
| 4 | Law of Harmonic Vibrations | period divisibility: 1 \| 8 \| 24 |
| 9 | Law of Cycles | orbit period = cycle length |
| 10 | Law of Harmonic Pitch | f-value mod M classifies pitch class |
| 18 | Law of Oscillating Atoms | QA step operator = atomic oscillation |
| 27 | Law of Chemical Substitution | same-orbit replacement preserves f-value class |
| 29 | Law of Vibrating Atomolic Substances | chromogeometry: C = 2de, F = d²−e², G = d²+e² |
| 33 | Law of Sono-Thermity | thermal coupling = concordance weight |

### Category 1: structural ratios

These laws all describe **static structural relationships** — ratios, proportions, and invariants that hold at any single time step. In QA, they correspond to algebraic identities on (b, e, d, a) tuples and the orbit classification.

### Key invariants

- **Pitch = f-value**: f(b,e) = b² + be − e² = N(b + eφ) in Z[φ]
- **Period hierarchy**: singularity(1) | satellite(8) | cosmos(24)
- **Concordance**: orbit co-membership determines coupling strength
- **Chromogeometry**: C² + F² = G² (Wildberger Theorem 6)

## Checks

| ID | Description |
|----|-------------|
| KSR_1 | schema_version == 'QA_KEELY_STRUCTURAL_RATIO_CERT.v1' |
| KSR_LAWS | all 8 law numbers present: {2,4,9,10,18,27,29,33} |
| KSR_PERIOD | period divisibility 1 \| 8 \| 24 verified |
| KSR_FVAL | f-value computed correctly for all witness states |
| KSR_LCM | LCM(1,8,24) = 24 |
| KSR_CHROMO | C² + F² = G² for all tuples |
| KSR_CLOSURE | each law maps to at least one QA identity or invariant |
| KSR_W | ≥3 witnesses (distinct law→QA mappings) |
| KSR_F | ≥1 falsifier (incorrect f-value rejected) |

## Source grounding

- **svpwiki.com**: Keely's 40 Laws of Vibratory Physics (Laws 2, 4, 9, 10, 18, 27, 29, 33)
- **Ben Iverson**: QA orbit theory, f-value as norm in Q(√5)
- **Wildberger**: chromogeometry theorem (C² + F² = G²)
- **Dale Pond / Vibes**: SVP consultant AI; 5-category classification of Keely's 40 laws (2026-04-03)

## Connection to other families

- **[128] Spread Period**: period divisibility and Pisano period
- **[133] Eisenstein Norm**: algebraic identities shared with structural ratios
- **[147] Synchronous Harmonics**: concordance coupling within orbits
- **[148] Musical Intervals**: pitch = f-value connects to G:F = 5:3 etc.
- **[153] Keely Triune**: triune orbit structure is the foundation for all Keely families

## Fixture files

- `fixtures/ksr_pass_mod9.json` — 8 laws mapped to QA invariants with witnesses
- `fixtures/ksr_fail_wrong_fval.json` — falsifier with incorrect f-value computation
