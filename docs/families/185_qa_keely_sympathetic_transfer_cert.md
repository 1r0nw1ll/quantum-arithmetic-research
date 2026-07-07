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
- **Audit note (2026-07-04)**: Keely quotes spot-checked against live svpwiki.com (byte-match); underlying QA arithmetic is pre-existing invariant machinery. The category *classification judgment* itself rests on Vibes' (Dale Pond's AI tool) interpretation, not an independent falsifiable check — see `private/keely_40_laws_classification.md` provenance note.

## Connection to other families

- **[153] Keely Triune**: three-orbit structure is the foundation for reachability
- **[147] Synchronous Harmonics**: synchronization dynamics within orbits
- **[184] Keely Structural Ratio**: structural invariants constrain transfer paths

## Fixture files

- `fixtures/kst_pass_reachability.json` — same-orbit reachability with path lengths
- `fixtures/kst_fail_cross_orbit.json` — falsifier claiming cross-orbit path exists

## Verification Note (2026-07-06)

Independently walked every witness path step by step via the real QA
T-operator `(b,e)→(e,qa_mod(b+e,m))`: (1,1)→(1,2)→(2,3)→(3,5)→(5,8)→
(8,4) (5 steps, matching declared `path_length`), and the (3,3)→(3,6)
one-step satellite transfer — all exact. Independently reconfirmed
every reachability/obstruction example's orbit classification from
scratch (COSMOS/SATELLITE/SINGULARITY via the standard `v_3` rule) —
all correct, including that `(1,1)` genuinely cannot reach `(9,9)`
(different orbits under the T-operator's bijective structure). No data
was wrong.

**Found and hardened a much larger fixture-trusting gap than [184]'s**
(this cert's sibling): `KST_REACH` previously only warned if
`both_orbit` was *missing*, never checked it was *correct* or that
`reachable` matched. `KST_PATH` only checked that `path_length` was an
*integer type*, never that the declared path was a real T-operator walk
or that the length matched. Worst: `KST_BLOCK` checked
`ex.get("source_orbit", ex.get("orbit1")) == ex.get("target_orbit", ex.get("orbit2"))`
— but neither `source_orbit`/`orbit1` nor `target_orbit`/`orbit2` exists
anywhere in this fixture's actual data (obstruction examples only have
`source_b/e`, `target_b/e`, and a free-text `reason` string) — both
sides of the comparison were always `None`, so the check **never
actually ran** for any obstruction example, silently. Hardened all
three: `KST_REACH`/`KST_BLOCK` now classify orbits directly from
`(b,e,modulus)` and check both the classification and the
reachable/obstructed conclusion; `KST_PATH` now walks each declared path
step-by-step via the real T-operator and cross-checks `path_length`.
Verified the hardened checks reject both a mislabeled cross-orbit pair
and a broken path step.
