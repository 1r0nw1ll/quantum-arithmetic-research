# QA-ML Pepe Replication Manifest

Status: draft, 2026-05-15.

Primary source: Pepe (2025), *Machine Learning with Geometric Algebra:
Multivectors for Modelling, Understanding and Computing*, local companion
file `/Users/player3/Downloads/2025-pepe.pdf`.

Purpose: inventory the thesis datasets, metrics, tables, figures, and
future-work branches before running large reproductions. This manifest
separates "reproduce Pepe" from "test a QA analog."

Ordered reading companion:
`docs/specs/QA_ML_PEPE_THESIS_READING_NOTES.md`.

Corrective claim audit:
`docs/specs/QA_ML_PEPE_CLAIM_AUDIT.md`.

## Claim Discipline

- Do not invent datasets, metrics, or claims.
- Record thesis page numbers for every artifact.
- Reproduce the GA/baseline result first; test QA analog second.
- Treat all continuous tensors as observer-side unless a QA cert explicitly
  owns the integer substrate.
- No cert promotion from this manifest alone.
- Strong claims are allowed when backed. Until then, each artifact must state
  its current claim boundary. A rational direction cert can support exact
  substrate claims; it does not by itself establish a Maxwell, EM,
  Whittaker-kernel, quantum-PDE, or physical-reconstruction claim. Those
  claims require their own theorem, cert, or empirical protocol.
- The goal of the Whittaker/PDE branch is to **earn** stronger claims by
  moving through explicit stages: exact substrate -> approximation/residual
  theorem -> solver/observer experiment -> external validation.

## Thesis Artifact Inventory

| ID | Thesis locus | Page(s) | Reproduce Pepe artifact | Dataset / domain | Metric / output | QA analog to test | Status |
|---|---|---:|---|---|---|---|---|
| ROT-1 | Ch. 2.2.3 rotation representation sanity check | 59-60 | Tables 2.1 and 2.2 matrix/rotor-to-representation geodesic error | Random rotation matrices / rotors with train-test split | Max/mean/std geodesic error by representation and loss | QA generator-state representation sanity check: compare canonical 4-tuple, residue/phase packet, and undercomplete packets | GA smoke PASS; QA analog PASS: canonical phase fixes [277], full v3 fixes [277]+[278] |
| ROT-2 | Ch. 2.2.4 pose estimation from 3D point clouds | 60-63 | Table 2.3 and Fig. 2.2 / 2.3 rotation-prediction error and noise robustness | corpus/modelnet40/airplane/ (real, 626 train + 100 test, 3000 pts/mesh) | Geodesic error distribution; noise-vs-error curves | QA paired-action recovery under partial observation | Pepe-side real-data PARTIAL: 6D matches Pepe (9.94 vs 8.36 deg), bivector gap architectural (34.34 vs 8.41 deg — needs PointNet, not cross-cov+MLP); QA analog PASS: full4_delta masked RF 0.970 vs be_pair 0.454 |
| ROT-3 | Ch. 2.2.5 inverse kinematics | 64-66 | Tables 2.4 and 2.5 plus Figs. 2.5-2.7 | CMU MoCap, 10000 frames, 67-33 train-test split | Geodesic error and Euclidean distance by rotation representation | QA analog: recover generator-chain state from observed orbit coordinates | PUBLIC RECONSTRUCTION COMPLETE; EXACT AUTHOR ARRAY STILL MISSING. Public source checks found no downloadable `positions-new.npy`, `rotations-new.npy`, `MOTIONS.zip`, or `ROTATIONS-NEW.zip` in the repo/releases/tags. Built a full-size deterministic archive-order ASF/AMC reconstruction at Pepe's declared intermediate size: positions `[1194159,31,3]`, rotations `[1194159,31,3,3]`, 926 files with `137_19` truncated to 400 frames, clean SO(3) checks. Corrected GA heads to rotor log/Cayley (not matrix Cayley). Three-seed thesis-scale 10k rerun: geodesic ranking `zhou_6d`, `quaternion`, `bivector_cayley`, `axis_angle`, `bivector_log`, `euler`, `matrix`; endpoint-Euclidean ranking `matrix`, `zhou_6d`, `quaternion`, `bivector_cayley`, `axis_angle`, `bivector_log`, `euler`. Metric split reproduced; exact Pepe numeric reproduction not claimed because the unseeded 875/760-file author subset remains absent. |
| MOL-1 | Ch. 2.3 molecular geometry optimisation | 67-75 | Figs. 2.9-2.12 and Table 2.6 rotor-guided differential evolution | Gly-Gly and Gly-Phe dipeptides; xTB energy objective | Convergence iterations, DoF, final conformation visual | QA analog: discrete rotor/generator proposal search over QA energy/cost packets | QA HARD-START ANALOG PASS; PEPE-SIDE STILL NOT EXACT. Pepe-side: `32` recovers protocol; `33` proves xTB/RDKit/OpenBabel; `34`/`35` run/repeat all three parameterization families; `36` high-energy public-start attempt is a negative control. QA-side: `37` replaces the old alphabet-size toy with independently selected hard starts; `38` adds dynamic family-preserving constrained proposals; `39` proves m=75 target is reachable by `sigma^5,nu^2`; `40` verifies exact path representation. `41_pepe_ch2_qa_mol1_generalized_path_benchmark.py` computes shortest constrained path per modulus and uses that as `chain_len`: m=24 dynamic/path-aware constrained converge 5/5 with valid_rate 1.0; m=75 dynamic/path-aware constrained converge 5/5, while full unconstrained unseeded converges only 1/5 with valid_rate 0.0039. QA-side now cleanly reproduces the constrained-vs-unconstrained validity/search lesson. No Pepe Table 2.6 reproduction claim until author-like molecule starts/validity exist. |
| PSP-1 | Ch. 3.7 PSP with GA features | 97-98 | Tables 3.2 and 3.3 GDT_TS/GDT_HA comparisons for orientational features | PDNET-style five-dataset PSP split named D1-D5 in thesis tables | GDT_TS, GDT_HA max/median/min | QA canonical/phase packet as an orientational sidecar; compare to no-orientation and GA orientation features | SOURCE RECOVERED; SOURCE VISUALS EXTRACTED; EXACT PEPE REPRODUCTION BLOCKED; REPAIRED QA ANALOG PASS WITH QA VISUALS. `42` recovers the thesis protocol: 1000 DEEPCOV train proteins, 150 PSICOV test proteins, Graph Transformer + 3D projector, SVD alignment for Tables 3.2/3.3, GDT_TS/GDT_HA max/median/min over D1-D5, feature cases (a)-(i). `55` renders the Chapter 3 source pages for Figs. 3.1-3.28 and crops Figs. 3.17-3.28 plus Tables 3.2-3.6 into `experiments/qa_ml/ch3_source_visuals/`. Exact reproduction requires PDNET/PDB splits and feature-generation code. Old QA analog `22` is INVALID/SATURATED: 295/300 majority class and macro-F1 drops 0.519 -> 0.496. Repaired QA analog `43` with balanced labels and endpoint coordinates withheld: best non-integrating QA sidecar is per-step phase (`c_step_phase_sidecar`), macro-F1 0.933 vs no-orientation baseline 0.860 over five seeds; balanced accuracy delta +0.073. `56` adds QA-generated visuals in `experiments/qa_ml/ch3_qa_psp_visuals/`: metric bars, baseline-vs-phase confusion matrices, representative path-orientation heatmaps, and phase-sidecar feature importance. The 4-tuple transition sidecar is reported only as an upper-bound because start plus deltas can be integrated toward the endpoint. This validates a narrow QA-side orientation-sidecar lesson, not the PDNET/GDT thesis table. |
| PSP-2 | Ch. 6.3 future work | 211 | Larger PSP pipeline exploration, explicitly including AlphaFold as future direction | PSP at realistic scale | GDT_TS/GDT_HA or pipeline-native structure metrics | AlphaFold-adjacent sidecar first; AlphaFold internal integration only after sidecar gains are stable | Deferred |
| POSE-1 | Ch. 4.2 CGAPoseNet + GCAN | 124 | Table 4.2 median translation/rotation errors | Cambridge Landmarks and 7 Scenes pose estimation | Median translation error, median rotation error | QA canonical-equivariant routing as projector/downsampler over pose proposals, after baseline reproduction | SOURCE RECOVERED; SOURCE FIGURES EXTRACTED; REAL 7-SCENES SUBSET ACQUIRED; PUBLIC MOTOR-TARGET RECONSTRUCTED; ACTUAL GCAN SANDWICH LAYER IMPLEMENTED; CALIBRATED QA RETRIEVAL-FED GCAN WINS BOTH METRICS; FULL PEPE MODEL STILL PENDING. `44` recovers the protocol text. `54` extracts the visual source artifacts and corrects the figure mapping: Fig. 4.1 is the CGAPoseNet+GCAN architecture graphic; Fig. 4.2 is the original CGAPoseNet pipeline graphic, not the GCAN architecture. Protocol: InceptionV3 2048 output reshaped to 256 x 8 motor proposals, GCAN sandwich-product dense layers 128-64-1, motor labels in 1D-Up CGA/G(4,0), Table 4.1 params, Table 4.2 median pose errors, Fig. 4.10 average-pose traces. `45` verifies official Microsoft 7-Scenes Heads (`956,332,240` bytes), 2,000 RGB/depth/pose frames, official seq-02 train / seq-01 test. `49` reconstructs an 8-coefficient public CGA motor target from Eq. 4.7 with explicit SO(3) projection and round-trip errors below 1e-9. `52` implements the actual even-blade `G(4,0)` sandwich dense layer. `53` fixes proposal source and QA metric calibration: scaling QA before `StandardScaler` was a bug because it canceled alpha; now plain/QA blocks are standardized separately and QA alpha is applied afterward. Validation-tail selection chose `qa_alpha=0.5, K=32`. Test result: plain weighted 0.443m / 20.96deg; QA weighted 0.439m / 21.21deg; plain retrieval-fed GCAN 0.446m / 20.50deg; QA retrieval-fed GCAN **0.435m / 19.77deg**. Current honest status: real sandwich reducer adds value, and calibrated QA wins both translation and rotation on the public Heads subset. Exact Table 4.2 still requires the InceptionV3 proposal backbone, author lambda/curvature conventions, all 13 scenes, and TensorFlow-GA training harness. |
| POSE-2 | Ch. 4.2 visual/ablation | 130 | Fig. 4.10 GCAN input/output pose visual plus Table 4.3 backbone ablation | Pose estimation test images | Pose plots; backbone error table | QA proposal/refine visualization mirroring E3 DRA traces | PUBLIC QA MOTOR-PROPOSAL VISUAL HARNESS BUILT; TABLE 4.3 STILL PENDING. `51` builds Fig. 4.10-style traces on official 7-Scenes Heads using reconstructed 8D motor targets: source RGB frame plus decoded staged motor-proposal averages for plain vs QA retrieval (`k=63,31,9,3`) and ground truth. Representative frame `000344`: QA `k=9` gives 0.359m / 30.07deg vs plain `k=3` 1.988m / 47.79deg. Claim boundary: these are staged weighted proposal averages, not trained CGAPoseNet+GCAN layer outputs. |
| LINE-1 | Ch. 4.3 Define/Refine/Align | 142-143 | Tables 4.5 and 4.6 line-registration performance and relative promotion/demotion | Structured3D and Semantic3D line alignment | Rotation and translation errors by quartile/mean | QA DRA: Define fallback proposals, Refine canonical head, Align route | Manifest only |
| LINE-2 | Ch. 4.3 line-alignment examples | 145 | Fig. 4.17 alignment examples across curvature settings | Semantic3D test line bundles | Source/target/estimated line bundle visuals | QA DRA visual trace; show proposal stream before and after canonical alignment | Tier 1 visual |
| PDE-1 | Ch. 5.3 GA-ReLU | 153-158 | Figs. 5.1-5.5 and Navier-Stokes GA-ReLU vs ReLU comparison | 2D incompressible Navier-Stokes generated via PhiFlow | Scalar, vector, one-step MSE; residual/error field visuals | QA-GA-ReLU: scalar ReLU plus phase/cardioid gate on quantized vector phase packets | Source mapped; solver replica pending |
| PDE-2 | Ch. 5.4 Fengbo | 159-178 | Tables 5.1-5.3, 5.7, 5.8 and Figs. 5.6-5.18 | ShapeNet Car and Ahmed Body irregular-geometry CFD benchmarks voxelized into `G(3,0,0)` multivector volumes | Relative L2 pressure/velocity errors; grid-size, parameter/model-size, and alpha/beta loss-weight ablations | QA-Fengbo: exact QA/CGA grid packets for mask + coordinate vector + normal-dual bivector + optional inlet trivector | **Mapped; not parked. Missing QA-Fengbo primitive.** |
| PDE-3 | Ch. 5.5 STAResNet | 184-202 | Figs. 5.21-5.37 including Fig. 5.30 seen/unseen Maxwell geometry error vs trainable parameters | Maxwell/FDTD 2D and 3D field benchmarks with Faraday bivector regression | MSE, correlation/SSIM, seen/unseen obstacle MSE, rollout MSE, parameter-count ablation | QA Faraday/STA packet plus QA residual block and rollout ledger | Source mapped; field-packet solver replica pending |
| PDE-4 | Ch. 6.3 future work | 211 | Validate Ch. 5.3 activation and Ch. 5.5 algebra-choice claims on other PDEs, including quantum-physics PDEs | Quantum/PDE toy problems to be selected after source inventory | PDE residual, conservation residual, and prediction error | QA-Whittaker branch: exact rational S1/S2 direction nets, angular-kernel sampling, phase-packet algebra | Proposed |
| BEYOND-1 | Ch. 6.4 beyond | 211-212 | Broader geometry-aware architecture claim for robotics, embodied AI, molecular modeling, physical simulation | No single dataset in thesis section | Source-faithful scoping only | Defer until pose, line alignment, PSP, and PDE reproductions are stable | Deferred |

## PDE And QA-Whittaker Track

Pepe's PDE future-work paragraph creates two separate tests:

1. **Activation test:** does a structure-aware activation like GA-ReLU
   remain helpful outside 2D Navier-Stokes?
2. **Algebra-choice test:** does choosing the algebra that matches the PDE
   structure remain helpful outside Maxwell/STAResNet?

The QA-Whittaker work is relevant to this branch because the repo already has
an exact, cert-gated substrate for Whittaker-motivated angular and phase
packets:

| Existing QA artifact | Current claim support | Next claim it could support with more evidence |
|---|---|---|
| `[266]` QA Whittaker Rational Direction S1 | exact rational direction net on the unit circle | density/convergence or approximation theorem over S1 |
| `[273]` QA Whittaker Rational Direction S2 | exact rational direction set on S2 via inverse stereographic chart | S2 sampling/convergence theorem suitable for wave-kernel approximation |
| `[274]` QA Whittaker Scalar Angular-Kernel Sampling | exact finite scalar samples over `[273]` packets | quantified Whittaker-kernel or spherical-quadrature approximation |
| `qa_whittaker_phase_packet_algebra_cert_v1` | exact symbolic phase-packet algebra candidate | numerical phase-packet approximation with audited trig/float boundary |
| `experiments/whittaker_em_qa_observer_null_test.py` | measured EM observer/null pattern over phase-packet coordinates | external-validation claim if replicated with registered data/null protocol |
| `experiments/whittaker_em_direction_sweep.py` | direction-sweep control for selected Whittaker/QA directions | direction-selection claim if statistically stable across controls |

### Proposed PDE Sequence

1. **PDE manifest extraction:** finish page-indexed extraction for all Ch. 5
   PDE tables and figures before running anything heavy.
2. **Toy PDE activation probe:** choose one low-cost PDE beyond
   Navier-Stokes, implement ReLU vs QA-ReLU/phase-gated observer comparison,
   and report residual metrics.
3. **Whittaker phase-packet observer:** test whether exact phase-packet
   features reduce PDE residual or improve sample efficiency on a bounded
   wave/field dataset.
4. **Maxwell-adjacent comparison:** only after the observer probe works,
   compare QA-Whittaker packets against the STAResNet-style Maxwell task.
5. **Quantum-physics PDE branch:** defer equation choice until source
   inventory. Candidate families must be simple enough to make residuals and
   boundary conditions auditable.

### First Toy PDE Probe

Implemented `experiments/qa_ml/10_pde_whittaker_toy_probe.py` as a bounded
observer experiment for Pepe §6.3, using analytic free-Schrodinger plane
waves:

```text
psi(x,t) = exp(i * (k*x - k*k*t))
```

Training uses k = {1,2,3,4,5}; testing holds out k = {6,7}. The model predicts
the next complex state from current real/imaginary state, k, x, t, and an
integer QA phase packet `(b,e,m=24)`. QA-ReLU gates the first hidden layer by
the orbit family and satellite phase of that packet.

| Model | next-step MSE | discrete PDE residual-error MSE | unit-circle MSE |
|---|---:|---:|---:|
| ReLU | 0.111996 | 182.850987 | 0.001663 |
| ReLU + output projection | 0.107784 | 175.973346 | **0.000000** |
| ReLU + penalty + projection | 0.117244 | 191.418563 | **0.000000** |
| QA phase-gated ReLU | **0.104473** | **170.568764** | 0.003131 |
| QA phase-gated ReLU + norm penalty | 0.118267 | 193.088337 | 0.001699 |
| QA phase-gated ReLU + output projection | **0.102567** | **167.456427** | **0.000000** |
| QA phase-gated ReLU + penalty + projection | 0.115971 | 189.340947 | **0.000000** |

Verdict: **narrow positive.** The QA phase gate improves prediction and
residual-error metrics on this analytic quantum-PDE toy. The initial
unconstrained variant worsened norm preservation, but output projection fixes
that and becomes the best model on all three reported metrics. The control
comparison matters: ReLU + projection improves over raw ReLU, but QA +
projection still beats ReLU + projection on next-step MSE and residual-error
MSE while both preserve norm. A training-time norm penalty alone is too blunt
here. The next claim can be promoted one step: phase-gated QA observer
features plus an explicit conservation projection improve held-out plane-wave
prediction and stencil residual matching over a conservation-projected ReLU
observer on this bounded analytic test.

### Robustness Sweep

Implemented `experiments/qa_ml/11_pde_whittaker_toy_sweep.py` to compare
QA + projection against ReLU + projection over 54 regimes:

```text
seeds = {0,1,2}
m_phase = {12,24,48}
dt = {0.020, 0.035, 0.050}
test_k = near {6,7} or far {8,9,10}
```

Overall sweep result:

| Delta: QA projected - ReLU projected | Mean | Std | QA win rate |
|---|---:|---:|---:|
| next-step MSE | +0.002241 | 0.021224 | 30/54 = 0.556 |
| residual-error MSE | -5.062264 | 50.830225 | 30/54 = 0.556 |
| unit-circle MSE | ~0 | ~0 | tied in practice |

Verdict: **mixed / conditional.** The single-probe claim survives as a real
regime, but does **not** generalize uniformly across the sweep. QA +
projection improves the residual-error mean and wins 30/54 regimes, but the
next-step MSE mean is slightly worse because several regimes fail strongly.
Best grouped regimes include `m_phase=24, dt=0.020, far test k` and
`m_phase=48, dt=0.020, far test k`; worst regimes include coarse
`m_phase=12` at larger dt on near tests.

Updated claim boundary: **QA phase gating plus conservation projection is a
conditional inductive bias for analytic wave observers, not yet a universal
PDE activation result.** The next evidence step is to identify the phase
resolution / timestep stability law, or to replace coarse phase bins with
Whittaker `[273]/[274]` direction/kernel packet features.

### Stability-Law Diagnostic

Implemented `experiments/qa_ml/12_pde_whittaker_stability_analysis.py` over
the saved sweep results. Candidate predictor:

```text
bin_advance = (k_max*k_max * dt / (2*pi)) * m_phase
```

Key correlations with QA - ReLU deltas:

| Predictor | corr(next-step delta) | corr(residual-error delta) |
|---|---:|---:|
| `radians_per_bin` | +0.303 | +0.333 |
| `phase_advance_rad` | +0.047 | +0.133 |
| `bin_advance` | -0.043 | -0.027 |
| `distance_to_integer_bin` | -0.070 | -0.141 |

Interpretation: coarse phase resolution (`radians_per_bin` larger) is the
clearest weak predictor of worse QA performance. This matches the failure
pattern around `m_phase=12`, but the law is not strong enough to explain all
regimes. Banding by `bin_advance` is also only partial: the `1_to_2_bins`
band wins 3/3, `2_to_4_bins` wins 4/9, and `over_4_bins` wins 23/42.

Diagnostic verdict: **weak stability law, useful but not explanatory enough.**
The next meaningful step is not more bin sweeps; it is to replace the coarse
phase-bin packet with the existing Whittaker `[273]/[274]` direction/kernel
packet and test whether a richer exact phase substrate removes the unstable
regimes.

### Whittaker Packet Ablation

Implemented:

- `tools/qa_ml/qa_whittaker_features.py`
- `experiments/qa_ml/13_pde_whittaker_packet_ablation.py`

This first ablation was deliberately conservative: keep the projected
ReLU-vs-QA comparison and add nearest `[273]` S2 direction/profile observer
features for the current complex phase `(cos theta, sin theta, 0)`.

Focused regimes: the clearest coarse-bin failures plus two strong coarse-bin
successes. Result:

| Feature packet | next-step delta mean | residual-error delta mean | QA win rate |
|---|---:|---:|---:|
| coarse phase only | +0.007228 | -15.246850 | 7/15 = 0.467 |
| coarse + nearest `[273]` S2 features | +0.019661 | +33.999383 | 5/15 = 0.333 |

Verdict: **negative for naive nearest-direction Whittaker features.** The
nearest `[273]` S2 packet is not a drop-in replacement for phase-bin
stability; it worsened both mean deltas and win rate on the targeted regimes.
This does not reject the Whittaker branch. It rejects this simplistic
observer map. The next Whittaker test should use the phase-packet algebra
directly: rational `omega dot x - v*t` packet features and/or `[274]`
profile averages aligned to the wave phase, not nearest static S2 directions.

### Final Phase-Packet Ablation Before Returning To Pepe

Implemented `experiments/qa_ml/14_pde_whittaker_phase_packet_ablation.py`.
This test added phase-aligned observer features:

```text
phase_frac = theta/(2*pi) mod 1
next_phase_frac
advance_frac = k*k*dt/(2*pi) mod 1
advance_cos, advance_sin
bin_advance = m_phase * k*k*dt/(2*pi)
```

Focused regimes match the nearest-S2 ablation. Result:

| Feature packet | next-step delta mean | residual-error delta mean | QA win rate |
|---|---:|---:|---:|
| coarse phase only | +0.007228 | -15.246850 | 7/15 = 0.467 |
| coarse + direct phase-packet features | +0.000491 | -28.067144 | 6/15 = 0.400 |

Verdict: **mixed / unstable.** Direct phase-packet features improve the mean
next-step and residual-error deltas relative to coarse phase only, but they
increase variance and reduce win rate. Some regimes become much stronger
(`m_phase=48, dt=0.020, far test k`), while others flip badly
(`m_phase=24, dt=0.020, far test k`). This is the right stopping point for
the current PDE branch: QA/Whittaker phase structure has real signal, but
the stable architecture is not yet identified. Return to Pepe replication
with this branch recorded as **conditional and unresolved**, not failed.

## Immediate Next Extraction Work

TASK: extract Pepe thesis replication rows for Ch. 5 PDE figures/tables

CONTEXT: The user explicitly identified Pepe Section 6.3 PDE future work as
related to QA-Whittaker. The next concrete step is a source-faithful
extraction of Ch. 5 PDE artifacts before implementing any solver.

READ:
- `/Users/player3/Downloads/2025-pepe.pdf`
- `docs/families/266_qa_whittaker_rational_direction_s1_cert.md`
- `docs/families/273_qa_whittaker_rational_direction_s2_cert.md`
- `docs/families/274_qa_whittaker_scalar_angular_kernel_sampling_cert.md`
- `qa_alphageometry_ptolemy/qa_whittaker_phase_packet_algebra_cert_v1/README.md`

WRITE:
- `experiments/qa_ml/pepe_pde_manifest.json`
- update this manifest with exact extracted captions/metrics

CONSTRAINTS:
- no network downloads in the manifest step
- no unbacked physics claims; backed claims should be promoted explicitly
- distinguish exact QA substrate from observer-side PDE tensors
- state Whittaker claim boundaries and escalation conditions explicitly

VERIFICATION:
- every Ch. 5 PDE table/figure row has page number, metric, baseline,
  dataset/source, and candidate QA analog
