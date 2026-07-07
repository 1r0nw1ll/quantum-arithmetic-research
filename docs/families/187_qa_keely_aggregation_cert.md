# Family [187] QA_KEELY_AGGREGATION_CERT.v1

## One-line summary

Keely's 5 aggregation/disintegration laws (Laws 3, 12, 28, 34, 35) map to QA state composition and decomposition: coupling tension modifies period, orbit density governs aggregation, discord drives dissociation, and synthesis is deterministic — Category 4 of the Vibes 5-category framework.

## Mathematical content

### Keely's aggregation/disintegration laws

**Corrected 2026-07-06** — the law-name table below previously did not
match the certified fixture data at all (all 5 law numbers had wrong
names/mappings attached, e.g. Law 3 was labeled "Law of Chemical
Affinity" here but the fixture's `keely_text`/`qa_mapping` fields
actually certify "Law of Corporeal Oscillations"). Corrected to match
`fixtures/kag_pass_composition.json`'s `laws` block, which is the
certified source of truth:

| Law # | Name (svpwiki) | QA mapping |
|-------|---------------|------------|
| 3 | Law of Corporeal Oscillations | coupling tension: non-isolated states modify each other's effective period |
| 12 | Law of Oscillating Atomic Substances | orbit density (states/period) determines effective pitch |
| 28 | Law of Chemical Dissociation | discord (cross-orbit f-value/pitch mismatch) causes orbit separation |
| 34 | Law of Atomic Dissociation | high-energy perturbation forces orbit reassignment |
| 35 | Law of Atomolic Synthesis of Chemical Elements | pitch (b,e) deterministically synthesizes the full tuple and orbit membership |

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
| KAG_COUPLE | isolated/coupled witness orbits genuinely reclassified from (b,e,modulus), not trusted as declared strings (fixed 2026-07-06 — was entirely unimplemented) |
| KAG_DENSITY | orbit density genuinely recomputed as count/period ratio (cosmos=72/24=3, satellite=8/8=1, singularity=1/1=1), plus 72+8+1==modulus² (hardened 2026-07-06 — was a bare count>count inequality) |
| KAG_DISSOC | concordant/discordant pair orbits genuinely reclassified and cross-orbit-ness recomputed (fixed 2026-07-06 — was entirely unimplemented) |
| KAG_SYNTH | d, a, f_value, orbit genuinely recomputed from declared (b,e,modulus) (hardened 2026-07-06 — was a bare trusted boolean flag) |
| KAG_W | ≥3 witnesses (composition, density, dissociation) |
| KAG_F | ≥1 falsifier (incorrect density rejected) |

## Source grounding

- **svpwiki.com/Law+of+Corporeal+Oscillations**, **Law+of+Oscillating+Atomic+Substances**, **Law+of+Chemical+Dissociation**, **Law+of+Atomic+Dissociation**, **Law+of+Atomolic+Synthesis+of+Chemical+Elements**: quoted verbatim in `fixtures/kag_pass_composition.json`'s `laws[*].keely_text` fields
- **Ben Iverson**: QA deterministic dynamics, orbit density analysis
- **Dale Pond / Vibes**: SVP consultant AI; Category 4 classification (2026-04-03)
- **Audit note (2026-07-04)**: Keely quotes spot-checked against live svpwiki.com (byte-match); underlying QA arithmetic is pre-existing invariant machinery. The category *classification judgment* itself rests on Vibes' (Dale Pond's AI tool) interpretation, not an independent falsifiable check — see `private/keely_40_laws_classification.md` provenance note.

## Connection to other families

- **[153] Keely Triune**: orbit structure underlies all aggregation dynamics
- **[184] Keely Structural Ratio**: f-value governs coupling tension
- **[185] Keely Sympathetic Transfer**: reachability determines aggregation possibility

## Fixture files

- `fixtures/kag_pass_composition.json` — state composition with coupling tensions and densities
- `fixtures/kag_fail_bad_density.json` — falsifier with incorrect orbit density

## Verification Note (2026-07-06)

Independently reconfirmed every numeric claim in `kag_pass_composition.json`
by hand: the Law-35 `derived_tuple` witness (b=2,e=1,m=9) gives
`d=qa_mod(3,9)=3`, `a=qa_mod(4,9)=4`, `f=b*b+b*e-e*e=5` — all match; the
Law-3 isolated/coupled `(1,1)` state and the Law-28/34 `concordant_pair`
`(1,1)&(2,3)` are both genuinely COSMOS; the `discordant_pair`
`(1,1)COSMOS` vs `(3,3)SATELLITE` are genuinely different orbits; the
81=72+8+1 state partition and 72/24=3, 8/8=1, 1/1=1 density ratios are
all correct. No fixture data was wrong.

**Found and fixed a real doc-only error** (did not affect the certified
fixture): the "Keely's aggregation/disintegration laws" table had all 5
law names/mappings wrong — none of them matched what the certified
fixture actually cites (e.g. Law 3 was labeled "Law of Chemical
Affinity" here vs. "Law of Corporeal Oscillations" in the fixture's
`keely_text`). Corrected the whole table plus the Source grounding
section to match the certified data.

**Found and implemented the two entirely-missing checks** flagged by
this cert's own docstring/checks-table (`KAG_COUPLE`, `KAG_DISSOC`) —
both were listed as implemented but had zero code in `validate()`. Also
hardened `KAG_DENSITY` (was a bare `cosmos_count <= satellite_count`
inequality, never used "period" at all despite the doc claiming a
count/period ratio) and `KAG_SYNTH` (was a bare trusted
`fully_determined` boolean, never recomputed d/a/f_value/orbit from the
declared (b,e)). This is the same "docstring lists a check, code never
implements it" pattern as [185]'s `KST_BLOCK`, but more severe: two full
checks were silently absent rather than one being vacuously true.
Verified all four hardened/new checks reject planted errors (wrong
isolated-state orbit, wrong discordant-pair orbits, wrong f_value, wrong
derived d, swapped density counts) while the real fixtures still pass.
