# QA Arithmetic Orbit-Specificity Null

**Status:** controlled negative result  
**Date:** 2026-07-23  
**Scope:** QA arithmetic mining stages 30 and 31  
**Open Brain:** `bd826177-8e2e-4b6d-bafc-2854550c63d7`

## Claim

For the static and dynamic target families tested in Stages 30 and 31, the
mod-9/mod-24 QA orbit labels do not add predictive structure beyond the raw
coordinate residues and arithmetic baselines from which those orbit labels are
computed.

This is a negative result for this mining style. It is not a negative result
against every possible use of QA orbit structure.

## Evidence

Stage 30 tested static joint arithmetic targets with matched orbit and non-orbit
feature sets.

- Script: `experiments/qa_quantum_arithmetic_mining/orbit_specific_discovery_stage30.py`
- Commit: `cf7199e8`
- Train window: `b,e = 1..100`
- Test window: `b,e = 101..300`
- Targets: `14`
- Feature sets: `13`
- Shuffled-label nulls: `50`
- Result hash: `218fb2aa14ae5c4e8168df27676831162d7b4df2a20cdde502741fe62d508c5c`

Stage 30 verdict counts:

| Verdict | Count |
|---|---:|
| `EXACT_REDUCTION_AVAILABLE` | 4 |
| `NON_ORBIT_BASELINE_COMPETITIVE` | 9 |
| `ORBIT_NULL_COMPETITIVE` | 1 |
| `ORBIT_SPECIFIC_CANDIDATE` | 0 |

Stage 31 tested dynamic `T`-step transition targets and included
orbit-derived controls to prove the harness can detect circular orbit signal
when the label is itself orbit-defined.

- Script: `experiments/qa_quantum_arithmetic_mining/orbit_dynamic_transition_stage31.py`
- Commit: `0d29cda3`
- Train window: `b,e = 1..100`
- Test window: `b,e = 101..300`
- Targets: `14`
- Feature sets: `13`
- Shuffled-label nulls: `50`
- Result hash: `ee194c4c748326bf82a65243a32e434082075b6ff6a4935e2774f660f8afe658`

Stage 31 verdict counts:

| Verdict | Count |
|---|---:|
| `ORBIT_DERIVED_CONTROL` | 4 |
| `NON_ORBIT_BASELINE_COMPETITIVE` | 6 |
| `ORBIT_NULL_COMPETITIVE` | 3 |
| `DEGENERATE_OR_LOW_TRAIN_SUPPORT` | 1 |
| `ORBIT_DYNAMIC_CANDIDATE` | 0 |

## Interpretation

The orbit-derived controls in Stage 31 light up strongly, so the harness is not
blind to orbit information. However, those controls are intentionally circular:
their labels are defined from orbit membership.

For non-circular labels, both stages give the same answer:

- raw coordinate-pair controls such as `be_pair9` and `be_pair24` match or beat
  orbit features;
- component and factor-signature baselines beat orbit features on product and
  squarefree labels;
- exact reductions dominate the labels that are reducible to elementary
  arithmetic on `b,e,d,a`;
- shuffled-label null ceilings absorb the remaining weak orbit effects.

The fair conclusion is that this residue-label mining approach did not find an
orbit-specific arithmetic signal.

## Scorecard

The broader QA arithmetic mining arc still produced four useful closures:

| Family | Result Type |
|---|---|
| `[529] D_plus_F_square` | conic parametrization theorem |
| `[530] directrix_distance_integer` | structural divisibility reduction |
| `[531] G_square` | Pythagorean parametrization theorem |
| `[532] h_integer` | square-part structural reduction |

Those are real results about the QA generators and their derived conic
quantities. They should not be presented as evidence that the mod-9/mod-24 QA
orbit classification carries predictive structure beyond its source residues.

## Next Boundary

Do not continue with another Stage 32 variant of the same residue-label search
unless the target definition changes materially.

Future orbit-specific work should require at least one of:

- a target depending on global orbit trajectory rather than local residues;
- algebraic use of QA orbit structure before labels are formed;
- comparison to external observer data where `be_pair` is not a complete
  source representation;
- a theorem-level relationship between the orbit partition and an independent
  arithmetic or geometric invariant.

