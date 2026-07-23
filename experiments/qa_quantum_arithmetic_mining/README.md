# QA Quantum Arithmetic Mining

This experiment builds a small hybrid data-mining pipeline for QA arithmetic tables.
It is deliberately scoped as exploration: the scripts generate relational tables,
validate known identities, and export inspectable CSV cuts. They do not claim a new
prime theorem or a factorization shortcut.

The current orbit-specificity conclusion is documented in
`docs/theory/empirical/qa_arithmetic_orbit_specificity_null.md`: Stages 30 and
31 form controlled negative results for this residue-label mining style.

## Tables

`generate_dataset.py` writes a SQLite database with two linked tables:

- `core_matrix`: one row per `(b,e)` coordinate with the 21 generated variables.
- `validation_matrix`: one row per `core_matrix.config_id` with semiprime flags,
  slope from the selected origin, grid distance, square flags, area comparison,
  and the ellipse apex-to-focus check.

The current exploratory table keeps `L = C*F/12` because that was the working
formula in this mining setup. The validation table also records
`triangle_area_CF = C*F/2` and `L_times_6` so the distinction remains explicit.

## Formula Scope

The generator uses:

- `d = b + e`
- `a = e + d`
- `B = b*b`, `E = e*e`, `D = d*d`, `A = a*a`
- `X = e*d`, `C = e*d*2`, `F = a*b`, `G = D + E`
- `L = C*F/12`, `H = C + F`, `I = abs(C - F)`
- `J = d*b`, `K = d*a`, `W = d*(e+a)`
- `Y = A - D`, `Z = E + K`
- `h = sqrt(F)*d`

## Run

From the repository root:

```bash
python3 experiments/qa_quantum_arithmetic_mining/generate_dataset.py --b-max 100 --e-max 100
python3 experiments/qa_quantum_arithmetic_mining/analyze_patterns.py
python3 experiments/qa_quantum_arithmetic_mining/train_small_models.py
python3 experiments/qa_quantum_arithmetic_mining/symbolic_hebbian_probe.py
python3 experiments/qa_quantum_arithmetic_mining/null_controls.py
python3 experiments/qa_quantum_arithmetic_mining/scale_grid_stage1.py
python3 experiments/qa_quantum_arithmetic_mining/pattern_targets_stage2.py
python3 experiments/qa_quantum_arithmetic_mining/stronger_nulls_stage3.py
python3 experiments/qa_quantum_arithmetic_mining/generalization_metrics_stage4.py
python3 experiments/qa_quantum_arithmetic_mining/rule_family_mining_stage5.py
python3 experiments/qa_quantum_arithmetic_mining/sweep_targets.py
python3 experiments/qa_quantum_arithmetic_mining/cross_scale_confirmation_stage9.py
python3 experiments/qa_quantum_arithmetic_mining/identity_audit_stage16.py
python3 experiments/qa_quantum_arithmetic_mining/consolidated_ledger_stage18.py
python3 experiments/qa_quantum_arithmetic_mining/leak_orbit_ablation_stage19.py
python3 experiments/qa_quantum_arithmetic_mining/leak_orbit_ablation_stage19.py --stage-id qa_quantum_arithmetic_stage20_high_null_geometry_survivors --targets D_plus_F_square,G_square,h_integer,directrix_distance_integer --null-iterations 200 --summary-json qa_quantum_arithmetic_stage20_high_null_geometry_survivors.json --leaderboard-csv qa_quantum_arithmetic_stage20_high_null_geometry_survivors_leaderboard.csv
python3 experiments/qa_quantum_arithmetic_mining/leak_orbit_ablation_stage19.py --stage-id qa_quantum_arithmetic_stage21_directrix_e_alone_control --targets directrix_distance_integer --null-iterations 200 --summary-json qa_quantum_arithmetic_stage21_directrix_e_alone_control.json --leaderboard-csv qa_quantum_arithmetic_stage21_directrix_e_alone_control_leaderboard.csv
python3 experiments/qa_quantum_arithmetic_mining/dplusf_square_param_audit_stage22.py
python3 experiments/qa_quantum_arithmetic_mining/dplusf_square_theorem_stage23.py
python3 experiments/qa_quantum_arithmetic_mining/dplusf_square_proof_closure_stage24.py
python3 experiments/qa_quantum_arithmetic_mining/directrix_divisibility_closure_stage25.py
python3 experiments/qa_quantum_arithmetic_mining/reduction_triage_stage27.py
python3 experiments/qa_quantum_arithmetic_mining/g_square_proof_closure_stage28.py
python3 experiments/qa_quantum_arithmetic_mining/h_integer_reduction_closure_stage29.py
python3 experiments/qa_quantum_arithmetic_mining/orbit_specific_discovery_stage30.py
python3 experiments/qa_quantum_arithmetic_mining/orbit_dynamic_transition_stage31.py
python3 experiments/qa_quantum_arithmetic_mining/orbit_path_invariant_stage32.py
```

Default artifacts land in `results/qa_quantum_arithmetic_mining_001/`.

## Self-Test

```bash
python3 experiments/qa_quantum_arithmetic_mining/generate_dataset.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/analyze_patterns.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/train_small_models.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/symbolic_hebbian_probe.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/null_controls.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/scale_grid_stage1.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/pattern_targets_stage2.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/stronger_nulls_stage3.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/generalization_metrics_stage4.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/rule_family_mining_stage5.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/sweep_targets.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/cross_scale_confirmation_stage9.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/identity_audit_stage16.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/consolidated_ledger_stage18.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/leak_orbit_ablation_stage19.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/dplusf_square_param_audit_stage22.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/dplusf_square_theorem_stage23.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/dplusf_square_proof_closure_stage24.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/directrix_divisibility_closure_stage25.py --self-test
python3 experiments/qa_quantum_arithmetic_mining/orbit_path_invariant_stage32.py --self-test
```

Each self-test prints canonical JSON with `{"ok":true}` on success.

## First Questions To Ask

- Which slope labels contain the most `X` semiprime hits?
- How close are the nearest semiprime coordinates to the chosen origin?
- Which rows have integer `h`, square `G`, or both?
- Do readable models recover smooth generated variables more easily than
  discrete semiprime labels?
- Which modular coordinate predicates have held-out lift for `X` semiprime rows?
- Which residue features become strongest in the Hebbian semiprime prototype?
- Do coordinate-only residue associations persist on an out-of-window grid?
- Do observed Hebbian metrics beat shuffled-label null controls?
- Does the signal survive square windows, bands, prime-centered samples,
  Fibonacci/QA sequence samples, and sparse random samples up to `1e6`?
- Does `F` semiprime behave like `X` semiprime or form a different target class?
- Which expanded pattern targets persist above shuffled-label controls?
- Which targets are too low-support for responsible interpretation?
- Do persistent targets beat label, coordinate, residue-column, polynomial,
  and same-density random null controls?
- Does the signal survive modulus-family ablations and reversed-window tests?
- What are the average precision, top-k enrichment, and calibration curves for
  sparse targets?
- Which interpretable residue, gcd, parity, orbit, slope-ray, distance-band,
  and QA identity-derived rules persist out-of-window?
- Does each mining run leave a compact ledger row with target, train window,
  test window, feature set, model, observed lift, null max lift, verdict, and
  hash?
- In a first serious multi-target sweep, is `X_semiprime` special, or do
  coordinate residues carry broader arithmetic structure across semiprime,
  square, integer-height, squarefree, and omega-count targets?
- How quickly does multiplicative signal degrade from omega-2 to omega-3 to
  omega-4 across the QA product family `X/F/W/K/R/EA`?
- Do distinct-prime-count, Mobius, and Liouville targets preserve the same
  signal, or do they separate squarefree structure from Omega-depth structure?
- Do Fermat midpoint-gap residues add geometric factorization signal beyond
  coordinate residues, and are they useful alone?
- Which QA geometry predicates are residue-visible: gcd/primitive triangle
  conditions, `G_square`, integer height, or sparse intersections such as
  `G_square_and_primitive` and `h_integer_and_G_square`?
- Do the QA-native radius, semiperimeter, and excircle products
  `R=b*e`, `J=d*b`, `K=d*a`, and `EA=e*a` preserve semiprime and squarefree
  residue signal?
- Do B-smooth targets reveal small-prime dominated QA structure, and how does
  lift change as the smoothness bound increases?
- Do conic identity aliases for focus distance, apex/major distance, latus
  semirectum, and sparse conic intersections preserve the QA residue signal
  before adding Dandelin-specific features?
- Does the reduced eccentricity ratio `e/d` expose residue-visible numerator,
  denominator, smoothness, and gcd structure?
- Which exact directrix/focus-directrix divisibility targets are residue-visible
  when computed as integer arithmetic from `directrix_distance = d*d*d/e`?
- Do the strongest Stage 8 targets persist from train window `1..100` into
  `101..300`, sampled `301..1000`, sampled `1001..3000`, sampled
  `3001..10000`, and sparse random samples up to `1e6` while beating
  shuffled-label and same-density top-k null controls?
- Do latus rectum / focal chord labels derived from `semi_latus = F` and
  `full_latus = 2*F` preserve residue-visible squarefree, smoothness,
  distinct-omega, and omega-depth structure?
- Do conic shape ratios `minor^2/major^2 = F/D` and `eccentricity^2 = E/D`
  expose reduced numerator/denominator structure, and which divisibility
  predicates collapse to zero support under the QA definitions?
- Do confocal-family targets preserving `X=e*d` expose stable squarefree,
  smoothness, distinct-omega, and gcd-coupling structure, and does
  `gcd(X,D)` split cleanly into structural `d` versus extra shared factors?
- Do director-circle / tangent-geometry labels from
  `director_radius_sq = D*(D+F)` collapse on the full product because
  `D=d*d`, while the exposed `D+F` factor preserves residue-visible structure?
- Does the evolute / curvature identity `D-F = E = e*e` make square targets
  degenerate while preserving a prime-square semiprime signal when `e` is prime?
- Does a consolidated conic Tier 7B / polar pack preserve the strongest
  directrix, latus, director-factor, evolute, and confocal-gcd signals, and
  does `polar_scale_X_plus_F_semiprime` add useful texture beyond `D+F`?
- Which mined targets are closed by elementary QA algebra and should be moved
  from empirical low-support status into `PROVEN_STRUCTURAL_IDENTITY`,
  `PROVEN_EMPTY`, `PROVEN_ALWAYS_TRUE`, or `PROVEN_BOUNDARY_ONLY` buckets?
- Which non-closed Tier 7B conic/polar targets survive cross-scale testing
  after the Stage 16 structural identities and proven-empty labels are removed?
- Which empirical-open targets remain highest priority after consolidating all
  stage ledgers by support, lift, null margin, cross-scale persistence, and
  algebraic status?
- Which signals survive after dropping raw `b,e` coordinate residues, comparing
  against canonical QA orbit-family/id features, and subtracting trivial
  component-Omega baselines for product semiprime targets?
- Which geometry survivors remain after stabilizing orbit-feature null ceilings
  and comparing `D_plus_F_square` against the trivial `b_even` obstruction
  baseline?
- Since `directrix_distance_integer` reduces to `e | b*b*b`, does mod-9 QA
  orbit family/id outperform `e`-only and `b`-only residue controls?
- Does the Pell-style parametrization for `D_plus_F_square` cover every
  positive `(b,e)` solution, and can the bounded zero-miss audit be promoted to
  an exact proof?
- Can the `D_plus_F_square` parametrization be moved from bounded audit to a
  proof cert by closing the remaining coprime-factor casework in the Stage 23
  theorem ledger?
- Does the rational-conic proof closure for `D_plus_F_square` satisfy the
  project bar for a formal QA theorem cert?
- Should `directrix_distance_integer` be retired from empirical-open status now
  that it reduces exactly to `e | b*b*b`, equivalently `kernel3(e) | b`?
- Should Will convert the closed `D_plus_F_square` and
  `directrix_distance_integer` results into one combined theorem cert or two
  separate cert families?
- Do ordered global orbit-path features predict orbit-integrated arithmetic
  labels beyond shuffled-path, unordered-path, current-cell, static-orbit, and
  factor-signature controls?
