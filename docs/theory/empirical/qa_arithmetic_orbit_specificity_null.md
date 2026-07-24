# QA Arithmetic Orbit-Specificity Null

**Status:** controlled negative result  
**Date:** 2026-07-23  
**Scope:** QA arithmetic mining stages 30, 31, and 32  
**Open Brain:** `bd826177-8e2e-4b6d-bafc-2854550c63d7`

## Claim

For the static factor-property, single-step dynamic-transition, and
multi-step ordered-path-invariant target families tested in Stages 30, 31,
and 32, the mod-9/mod-24 QA orbit labels do not add predictive structure
beyond the raw coordinate residues and arithmetic baselines from which those
orbit labels are computed.

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

Stage 32 tested multi-step ordered path invariants (whether the sequence of
states visited under repeated `qa_step`, not just the current cell, carries
information), with the same discipline of adding a harder control within the
same run rather than across stages.

- Script: `experiments/qa_quantum_arithmetic_mining/orbit_path_invariant_stage32.py`
- Commit: `b64332da`
- Train window: `b,e = 1..100`
- Test window: `b,e = 101..300`
- Feature sets: `13`+
- Result rows: `252`
- Result hash: `e07ecfafd8ae6416f23c13ecae080912d14ed2d54bd18417fef0b4e42496d248`

Stage 32 verdict counts:

| Verdict | Count |
|---|---:|
| `CONTROL_BASELINE_COMPETITIVE` | 9 |
| `DEGENERATE_OR_LOW_TRAIN_SUPPORT` | 3 |
| `ORBIT_PATH_CANDIDATE` | 0 |

One apparent candidate surfaced on the first pass
(`path_count_F_squarefree_ge_half`) and collapsed under a shuffled-path
control added within the same run: `ordered_path9_24`, `shuffled_path9_24`,
and `be_pair9_24` all produced bit-identical lift (`2.1206680417081905`,
matching to the last decimal) on that target, proving path *order*
contributed nothing beyond the raw coordinate pair.

## Interpretation

The orbit-derived controls in Stage 31 light up strongly, so the harness is not
blind to orbit information. However, those controls are intentionally circular:
their labels are defined from orbit membership.

For non-circular labels, all three stages give the same answer:

- raw coordinate-pair controls such as `be_pair9` and `be_pair24` match or beat
  orbit features — including beating a genuinely stronger prior read: an
  earlier pass (Stage 21) found `qa_orbit_family9` beating single-variable
  `b_only`/`e_only` baselines on `directrix_distance_integer` (lift 3.93 vs.
  2.61/2.43), which looked like real orbit-specific signal until Stage 30
  showed `be_pair9` beats that same orbit lift (4.44 vs. 3.93) at the same
  modulus — since orbit membership is a lossy function of the pair, this
  means the orbit compression was discarding signal, not adding it;
- component and factor-signature baselines beat orbit features on product and
  squarefree labels;
- exact reductions dominate the labels that are reducible to elementary
  arithmetic on `b,e,d,a`;
- shuffled-label and shuffled-path null ceilings absorb the remaining weak
  orbit effects, including one apparent path-order candidate in Stage 32 that
  turned out to be bit-identical to its shuffled-path control.

The fair conclusion is that this residue-label mining approach did not find an
orbit-specific arithmetic signal, across static, dynamic, and path-based
target families, each tested with a control built to be generous to orbit
features.

## Scorecard

The broader QA arithmetic mining arc still produced five useful closures:

| Family | Result Type |
|---|---|
| `[529] D_plus_F_square` | conic parametrization theorem |
| `[530] directrix_distance_integer` | structural divisibility reduction |
| `[531] G_square` | Pythagorean parametrization theorem |
| `[532] h_integer` | square-part structural reduction |
| `[533] QA Orbit Satellite Ramification` | ramification theorem for the orbit classifier itself |

The first four are real results about the QA generators and their derived
conic quantities. `[533]` is different in kind: it is a theorem about *why*
`qa_orbit_rules.py`'s own divisor-shortcut Satellite classifier under-counts
by exactly 32 states whenever `5 | m` (a discriminant-5 ramification
argument — `x²-x-1`'s repeated root mod 5 gives the Fibonacci matrix a
4-vector period-4 eigenspace there). It also closes an explicitly flagged
open item in the pre-existing cert `[291]`. None of these five results
should be presented as evidence that the mod-9/mod-24 QA orbit
classification carries predictive structure beyond its source residues on
external targets — `[533]` explains the classifier's own algebra, it does
not show orbit membership predicts anything about `X`, `F`, `D+F`, or any
other mined quantity.

## Next Boundary

Stages 30–32 close this line of inquiry (residue/path mining of orbit
membership against arithmetic-property targets) as an honest null. Do not
re-run another variant of the same search (a Stage 33 with yet another target
pack) unless the target definition changes materially in one of the ways
below.

`[533]` is a worked example of the kind of different question that *did* pay
off: instead of asking "does orbit membership correlate with target Y," it
asked "why does the orbit classifier's own algebra behave the way it does,"
and answered that with a genuine theorem (discriminant-5 ramification) rather
than another empirical sweep.

Future orbit-specific mining work should require at least one of:

- a target depending on global orbit trajectory rather than local residues;
- algebraic use of QA orbit structure before labels are formed (the `[533]`
  pattern: prove something about the classifier itself, or about how orbit
  structure composes with other exact QA machinery, rather than correlating
  a label against a target);
- comparison to external observer data where `be_pair` is not a complete
  source representation;
- a theorem-level relationship between the orbit partition and an independent
  arithmetic or geometric invariant (e.g. the E8/quaternion-order structure
  already established elsewhere in this project).

