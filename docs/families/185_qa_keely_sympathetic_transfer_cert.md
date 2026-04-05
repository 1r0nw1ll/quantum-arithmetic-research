# Family [185] QA_KEELY_SYMPATHETIC_TRANSFER_CERT.v1

## One-line summary

Keely's 7 sympathetic transfer laws (Laws 5, 6, 7, 8, 17, 37, 40) map to QA reachability and path structure: orbit co-membership as sympathy, cross-orbit boundaries as discord, and Vibes triad concordance — Category 2 of the Vibes 5-category framework.

## Mathematical content

### Keely's sympathetic transfer laws

| Law # | Name (svpwiki) | QA mapping |
|-------|---------------|------------|
| 5 | Law of Transmissive Vibraic Energy | QA step propagation within orbit |
| 6 | Law of Sympathetic Oscillation | same-orbit states resonate (reachable) |
| 7 | Law of Attraction | orbit co-membership = sympathetic coupling |
| 8 | Law of Repulsion | cross-orbit = discord, reachability obstruction |
| 17 | Law of Sympathetic Vibration | path length determines transfer efficiency |
| 37 | Law of Etheric Vibration | singularity anchors all transfer paths |
| 40 | Law of Sympathetic Association | triad concordance: three orbits relate via Vibes triad |

### Category 2: sympathetic transfer

These laws describe **dynamic transfer** — how vibratory energy moves between states. In QA, this corresponds to reachability within the orbit graph: states in the same orbit are connected by QA step paths; states in different orbits are unreachable.

### Key structures

- **Reachability**: state A reaches state B iff they share an orbit
- **Path length**: minimum QA steps from A to B = transfer efficiency
- **Obstruction**: cross-orbit pairs have no QA path (discord)
- **Triad concordance**: singularity mediates satellite↔cosmos interaction as neutral center

## Checks

| ID | Description |
|----|-------------|
| KST_1 | schema_version == 'QA_KEELY_SYMPATHETIC_TRANSFER_CERT.v1' |
| KST_LAWS | all 7 law numbers present: {5,6,7,8,17,37,40} |
| KST_REACH | same-orbit states reachable by QA step path |
| KST_BLOCK | cross-orbit pairs have no finite path |
| KST_PATH | path lengths computed for all witness pairs |
| KST_TRIAD | singularity connects to both satellite and cosmos via triad |
| KST_W | ≥3 witnesses (distinct reachable pairs) |
| KST_F | ≥1 falsifier (cross-orbit path claim rejected) |

## Source grounding

- **svpwiki.com**: Keely's 40 Laws (Laws 5, 6, 7, 8, 17, 37, 40)
- **svpwiki.com/Sympathy**: "a force acting between two bodies vibrating in unison"
- **Ben Iverson**: QA orbit reachability and step-operator dynamics
- **Dale Pond / Vibes**: SVP consultant AI; Category 2 classification (2026-04-03)

## Connection to other families

- **[153] Keely Triune**: three-orbit structure is the foundation for reachability
- **[147] Synchronous Harmonics**: synchronization dynamics within orbits
- **[184] Keely Structural Ratio**: structural invariants constrain transfer paths

## Fixture files

- `fixtures/kst_pass_reachability.json` — same-orbit reachability with path lengths
- `fixtures/kst_fail_cross_orbit.json` — falsifier claiming cross-orbit path exists
