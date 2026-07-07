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
- **Audit note (2026-07-04)**: Keely quotes spot-checked against live svpwiki.com (byte-match); underlying QA arithmetic is pre-existing invariant machinery. The category *classification judgment* itself rests on Vibes' (Dale Pond's AI tool) interpretation, not an independent falsifiable check — see `private/keely_40_laws_classification.md` provenance note.

## Connection to other families

- **[128] Spread Period**: period divisibility and Pisano period
- **[133] Eisenstein Norm**: algebraic identities shared with structural ratios
- **[147] Synchronous Harmonics**: concordance coupling within orbits
- **[148] Musical Intervals**: pitch = f-value connects to G:F = 5:3 etc.
- **[153] Keely Triune**: triune orbit structure is the foundation for all Keely families

## Fixture files

- `fixtures/ksr_pass_mod9.json` — 8 laws mapped to QA invariants with witnesses
- `fixtures/ksr_fail_wrong_fval.json` — falsifier with incorrect f-value computation

## Verification Note (2026-07-06)

Independently confirmed the f-value formula is a genuine norm identity:
`f(b,e)=b²+be−e²=N(b+eφ)` in `Z[φ]` (φ=golden ratio), verified
numerically for 5 pairs against the literal complex-conjugate norm
computation. Independently reconfirmed all 4 numeric witnesses: (1,1)→
f=1, C=4,F=3,G=5 (4²+3²=5²); (9,9)→f=81 (SINGULARITY, the fixed point);
(3,3)→f=9 (SATELLITE), generator step (3,3)→(3,6) stays SATELLITE.
Keely-quote provenance already audited 2026-07-04 (Vibes-caveat note
above), not repeated here.

**Found and hardened two real fixture-trusting gaps**: `KSR_CHROMO`
only checked that declared `C,F,G` were mutually Pythagorean-consistent
(`C²+F²=G²`), never that they actually equal `2de, d²−e², d²+e²` for the
witness's own `(b,e)` — a witness could have declared *any* Pythagorean
triple regardless of its stated `(b,e)`. Similarly, `KSR_CLOSURE` only
checked that a witness's declared `orbit_family` and
`next_orbit_family` strings matched *each other*, never that either was
the *correct* classification for its `(b,e)` — a witness could
mislabel a genuine SATELLITE state as COSMOS in both fields and still
pass. Hardened both: `KSR_CHROMO` now recomputes `C,F,G` from raw
`d=b+e` (per project convention — raw, not qa_mod-wrapped) and compares
to every declared field; `KSR_CLOSURE` now genuinely classifies orbit
family from `(b,e,m)` via the standard `v_3`-based rule
(`SINGULARITY` iff both wrap to `m`; `SATELLITE` iff both divisible by 3;
else `COSMOS`). Verified both hardened checks reject planted wrong
values (a bad `C` and a mislabeled-but-internally-consistent orbit).
`--self-test` passes on both fixtures.
