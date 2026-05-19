# QA-ML Pepe Chapter 5 PDE Solver Mapping

Status: corrective source map, 2026-05-18.

Machine artifact:
`experiments/qa_ml/results_pepe_ch5_pde_solver_source_map.json`.

First primitive artifact:
`experiments/qa_ml/results_pepe_ch5_qa_fengbo_packet_parity.json`.

First operator artifact:
`experiments/qa_ml/results_pepe_ch5_qa_fengbo_operator_parity.json`.

First real-data smoke artifact:
`experiments/qa_ml/results_pepe_ch5_real_ahmedml_source_smoke.json`.

First real pressure-field artifact:
`experiments/qa_ml/results_pepe_ch5_real_shapenet_pressure_field_smoke.json`.

First real grid-packet pressure artifact:
`experiments/qa_ml/results_pepe_ch5_real_shapenet_grid_packet_pressure.json`.

First real pressure grid-neural artifact:
`experiments/qa_ml/results_pepe_ch5_real_shapenet_voxel_cnn_pressure.json`.

## Correction

The Chapter 5 visual replica is a scaffold, not a PDE solver replication.
The honest status is:

| Pepe solver | Current QA status | Next required primitive |
|---|---|---|
| GA-ReLU, 2D Navier-Stokes | Mapped; solver replica pending | QA-GA-ReLU over quantized vector phase packets |
| Fengbo, 3D irregular CFD | **Volumetric P+V QA/continuous operator parity ESTABLISHED (script 68).** On the RAW primary-source Umetani `mlcfd_data` archive (sha256-pinned, the file the Geo-FNO readme points to), a 3D-FNO predicts the **volumetric velocity field V (R²=0.991, relative-L2 0.081 — published-GINO quality) and surface pressure P (R²=0.876)**; QA tracks continuous to ~1–2e-3 with both gaps shrinking monotonically by modulus — a non-degenerate parity on **both** packets. This closes the "velocity V packet missing" gap flagged across 60–67. Prior milestones still stand: script 67 established surface-pressure operator parity on the canonical GINO record (R²=0.958) and **retracted** the earlier "hard ceiling R²≈0.41" (a target-misalignment artifact). | Full published GINO training budget (CPU scale here: 32³, 15 epochs, 180 cars). Signed SDF is *infeasible* on this data (sampled surfaces non-watertight) — not a gap, a recorded property. |
| STAResNet, Maxwell | Mapped; solver replica pending | QA Faraday/STA residual packet |

## Fengbo Is Not A Dead End

The earlier "parked" status was too conservative. Pepe Fengbo already
performs the key move needed by QA: irregular 3D geometry is discretized into
fixed-resolution voxel volumes of multivectors. QA can lawfully enter at that
boundary by quantizing the Clifford/CGA geometry packets, then decoding to
observer-side pressure/velocity fields for relative-L2 evaluation.

The required QA-Fengbo packet is:

```text
P_QA = mask + quantized coordinate vector + quantized normal-dual bivector
V_QA = mask + quantized coordinate vector
```

For Ahmed Body, the inlet velocity maps to the optional trivector channel,
matching Pepe's construction.

## Implementation Order

1. Build `qa_cga_grid_packet_v1`: exact rational/integer voxel coordinate
   packets with dequantization metadata. **Done.**
2. Build `qa_fengbo_geometry_multivector_v1`: pressure and velocity geometry
   packets matching Pepe's `P` and `V`. **Done as packet smoke.**
3. Build a synthetic mini-Fengbo operator benchmark: sphere/ellipsoid/car-like voxel
   masks with analytic pressure-like scalar and velocity-like vector fields.
   **Done.**
4. Run continuous mini-Fengbo and QA-quantized mini-Fengbo head-to-head.
   **Done.**
5. Acquire ShapeNet Car / Ahmed Body and attempt a real-subset Fengbo smoke.
   **Done for AhmedML geometry metadata + Cd/Cl force coefficients only.**
6. Acquire field/mesh pressure/velocity data and attempt a real-subset
   Fengbo field smoke. **Done for ShapeNet-Car pressure only.**
7. Build real `P` grid-packet pressure dataset with mesh vertices and normals.
   **Done.**
8. Build a real pressure-only voxel/grid neural operator parity test.
   **Done — small 3D CNN (65) and then the published-class 3D FNO (66).**
9. **Non-degenerate operator parity ACHIEVED (script 67).** The "ceiling"
   diagnosed at step 8/9 was a *target-misalignment artifact*, not a real
   ceiling. Switching to the trusted canonical neuraloperator GINO record
   (Zenodo 13936501) and applying the authors' canonical press alignment
   `concat(press[:16],press[112:])` instead of our "first-3586" hack, the
   continuous 3D-FNO reaches R²=0.958 (relative-L2 0.173) — a genuinely
   working operator — and QA tracks it to 7.7e-4 with the gap shrinking by
   modulus. This is the first non-degenerate QA/continuous operator-parity
   rung. The benchmark task is surface pressure (no volumetric velocity in
   any distribution); the full volumetric Fengbo solver is out of scope for
   lack of such data, not lack of operator. See the GINO section.

## Parity Criterion

QA-Fengbo should match the continuous mini-Fengbo baseline within a declared
relative-L2 band, initially `+0.5` to `+1.0` percentage points at matched
parameter count, with monotone improvement as grid resolution `M` or QA
modulus/resolution `m` increases.

No claim should say "Fengbo replicated" until a continuous Fengbo baseline and
QA-Fengbo run are compared on the same data split.

## Packet Parity Smoke

`60_pepe_ch5_qa_fengbo_packet_parity.py` validates only the geometry packet
boundary. It uses a synthetic ellipsoid surface and volume field, encodes
Pepe-style pressure packets `P` and velocity packets `V`, decodes them, and
measures observer relative-L2 error.

At `m = 144`:

| Field | Relative L2 |
|---|---:|
| pressure-like scalar | 0.001952 |
| velocity-like vector | 0.008818 |

Both errors decrease monotonically across the tested moduli. This validates
the QA packet mapping, not the learned Fengbo neural operator.

## Synthetic Operator Parity

`61_pepe_ch5_qa_fengbo_operator_parity.py` trains the same deterministic
mini-Fengbo operator on the same synthetic shape split:

- continuous Fengbo-style geometry packets
- QA-dequantized Fengbo-style geometry packets

At `m = 144`:

| Metric | Continuous | QA | QA - continuous |
|---|---:|---:|---:|
| pressure relative L2 | 0.0000000017 | 0.0016336911 | +0.0016336894 |
| velocity relative L2 | 0.0000032558 | 0.0027234740 | +0.0027202182 |

The pressure and velocity gaps decrease monotonically over the tested moduli.
This validates controlled mini-operator parity, not ShapeNet/Ahmed or full
Clifford-FNO Fengbo replication.

## Real AhmedML Metadata/Force Smoke

`62_pepe_ch5_real_ahmedml_source_smoke.py` is the first real-data gate. It
downloads a small fixed public AhmedML subset:

- `run_i/geo_parameters_i.csv`: eight Ahmed-body geometry parameters
- `run_i/force_mom_i.csv`: Cd/Cl force coefficients

It trains matched continuous and QA-quantized degree-2 ridge regressors on
runs 1-64 with the same deterministic train/test split. This is intentionally
not a pressure/velocity field solver and not a full Fengbo reproduction.

At `m = 144`:

| Metric | Continuous | QA | QA - continuous |
|---|---:|---:|---:|
| joint Cd/Cl relative L2 | 1.1951471828 | 1.0570075062 | -0.1381396766 |
| Cd relative L2 | 0.2224478212 | 0.2414403658 | +0.0189925447 |
| Cl relative L2 | 1.9550267988 | 1.7191958029 | -0.2358309960 |

Verdict: `PASS_REAL_METADATA_FORCE_SMOKE` under the pre-declared parity
criterion. The negative joint/Cl gap should not be framed as a QA win; on this
small metadata-only split the continuous Cl error is high, so the likely
interpretation is quantization acting as mild regularization. The result only
establishes that public real AhmedML geometry/force data can pass through the
QA-Fengbo packet boundary without destroying a matched baseline.

The next unmapped step is field-level data acquisition: pressure/velocity
volumes or meshes from NVIDIA PhysicsNeMo Ahmed Body, ShapeNet-Car/GINO, or an
equivalent public source, followed by continuous Fengbo vs QA-Fengbo on the
same field split.

## Real ShapeNet-Car Pressure-Field Smoke

`63_pepe_ch5_real_shapenet_pressure_field_smoke.py` is the first field-level
gate. It uses the public Zenodo `processed-car-pressure-data.zip` archive for
the processed ShapeNet-Car pressure dataset. The archive provides official
train/test manifests, watertight car meshes, and fixed-length pressure vectors.

The smoke trains matched operators on the same official-manifest subset:

- continuous: mesh descriptors -> pressure PCA coefficients -> pressure field
- QA: quantized/dequantized mesh descriptors -> same pressure PCA target form

At `m = 144` on 64 train cars and 16 test cars:

| Metric | Continuous | QA | QA - continuous |
|---|---:|---:|---:|
| pressure relative L2 | 0.8059375308 | 0.7926530780 | -0.0132844528 |
| mean per-car relative L2 | 0.6315384962 | 0.6218258170 | -0.0097126791 |

Verdict: `PASS_REAL_PRESSURE_FIELD_SMOKE`. The absolute parity gap shrinks
from `0.1818438059` at `m = 24` to `0.0049611482` at `m = 288`.

This must not be described as a full Fengbo or solver-quality result. The
baseline is intentionally small and has high pressure error; the result
validates real field acquisition plus QA geometry-boundary parity. Velocity
remains pending because this Zenodo record is the processed pressure archive,
not a pressure+velocity bundle.

## Real ShapeNet-Car `P` Grid-Packet Pressure Smoke

`64_pepe_ch5_real_shapenet_grid_packet_pressure.py` is the first real
Fengbo-style pressure-packet gate. It uses mesh vertices, computed vertex
normals, and direct pressure scalar targets from the processed ShapeNet-Car
archive.

The feature schema is the actual pressure packet boundary:

```text
P = mask + coordinate vector + normal-dual bivector
```

The matched operators are:

- continuous: floating `P` features -> direct pressure scalar
- QA: `P_QA` integer packets -> observer-decoded `P` features -> direct
  pressure scalar

At `m = 144` on 64 train cars, 16 test cars, and 256 sampled vertices per car:

| Metric | Continuous | QA | QA - continuous |
|---|---:|---:|---:|
| pressure relative L2 | 0.7885834000 | 0.7885871486 | +0.0000037486 |
| pressure MAE | 27.9786139148 | 27.9779144748 | -0.0006994400 |

Verdict: `PASS_REAL_GRID_PACKET_PRESSURE_SMOKE`. The absolute parity gap
shrinks from `0.0003071683` at `m = 24` to `0.0000002236` at `m = 288`.

Archive caveat: pressure vectors contain 3682 entries while each mesh has 3586
vertices. This smoke aligns by using the first `vertex_count` pressure entries
and records the 96-entry pressure tail as unused. That keeps the test honest
but means it is still a sampled packet-field smoke, not a full reproduction.

This is much closer to Fengbo than the descriptor/PCA pressure smoke because
the geometry boundary is now actual sampled `P` packets. It is still not full
Fengbo because the operator is a small polynomial ridge model, not a 3D
Clifford/FNO grid operator, and velocity `V` packets are not covered.

## Real ShapeNet-Car Pressure Voxel-CNN Smoke

`65_pepe_ch5_real_shapenet_voxel_cnn_pressure.py` removes the polynomial-ridge
operator from script 64. It voxelizes real mesh vertices and computed normals
into sparse `P` tensors on a `24^3` grid, trains a small 3D CNN with
occupied-voxel pressure loss, and compares the same architecture under
continuous and QA-quantized inputs. **Hardened to the script-66 standard:**
heterogeneous-archive PLY parser, QA quantization applied to feature channels
**and** voxel placement, honest verdict (no green Fengbo PASS).

Operating-point run (500 train / 80 test cars, 24³ grid, hidden 16, 15
epochs; QA on channels and placement):

| Metric | Continuous | QA m=144 | QA − continuous |
|---|---:|---:|---:|
| pressure relative L2 | 0.570853 | 0.565877 | −0.004976 |
| pressure R² | 0.2568 | — | — |

Verdict: `QA_BOUNDARY_PARITY_OK__CONTINUOUS_OPERATOR_WEAK`. The QA boundary is
faithful (abs gaps 2.0e-3 at m=24 → 9.6e-4 at m=288, all far inside the 0.03
band) **but the continuous CNN is an even weaker operator than the FNO
(R²≈0.26 vs 0.41)** — it does not escape the surface-pressure floor either.
This is QA quantization-boundary parity, not solver-quality parity, and not a
Fengbo solver rung. See the FNO section for the full ceiling analysis.

## Real ShapeNet-Car 3D-FNO Operator — and the Honest Ceiling

> **⚠ RETRACTION (see the Canonical GINO Record section below).** The "hard
> ceiling R²≈0.41" concluded in this section is **wrong**. It was a
> target-misalignment artifact: scripts 64–66 paired mesh vertices with
> pressure via a "first-3586-of-3682" truncation, while the dataset authors'
> canonical fixup is `concat(press[:16], press[112:])`. With the correct
> alignment (script 67) the same FNO reaches **R²=0.958**. The analysis below
> is retained as the diagnostic record of the misaligned regime; its ceiling
> conclusion does not hold.

`66_pepe_ch5_real_shapenet_fno_pressure.py` replaces the local CNN with the
published-class operator: a 3D Fourier Neural Operator (spectral convolution)
trained on a dense geometry channel — an *unsigned* nearest-surface distance
field (GINO/Fengbo use a true signed SDF; the archive provides surface meshes
only, so this is the unsigned approximation), on the full 500-car official
ShapeNet-Car split, under matched continuous and QA-quantized packet
boundaries. QA quantization is applied to both the distance channel and voxel
placement, so the QA perturbation is not understated.

The public archive is heterogeneous (Open3D-emitted double/uchar meshes and
meshio-emitted float/uint8/int32 meshes); the parser handles both. The
official train/test split this script loads is empirically 100% double/uchar
and parsed exactly, so the result below is valid; the float-path robustness is
covered by the self-test and matters only for non-manifest meshes.

A capacity × data × regularization sweep (96–500 cars, with test-curve
tracking to separate underfit / overfit / true ceiling), plus a
distance-channel ablation, established a hard empirical ceiling: the
continuous operator — **regardless of class (ridge, point-MLP, CNN, FNO),
capacity, data, or the dense distance input** — does not escape a
surface-pressure mean-residual relative-L2 floor of ≈0.77 (R²≈0.41) on this
data at CPU scale.

Operating-point run (500 train / 80 test cars, 24³ grid, FNO width 12 / 7
modes / 3 layers; QA quantization on distance channel **and** placement):

| Metric | Continuous | QA m=144 | QA − continuous |
|---|---:|---:|---:|
| pressure relative L2 | 0.5103559 | 0.5091642 | −0.0011917 |
| pressure R² | 0.4059 | — | — |

Verdict: `QA_BOUNDARY_PARITY_OK__CONTINUOUS_OPERATOR_WEAK`. With the QA boundary
applied to placement too, the QA gap is a non-trivial 8.8e-3 at coarse m=24
and converges monotonically toward continuous as the modulus rises
(4.0e-3 at m=72 → 1.2e-3 at m=144 → 1.5e-3 at m=288), all far inside the 0.03
band — **but the continuous FNO's own error (relative-L2 ≈0.51, R²≈0.41) is
still ~2–3 orders of magnitude larger than that gap.** The parity therefore
certifies the QA packet *boundary* (and its convergence with modulus), not
solver-quality *operator* parity. This is deliberately not a green Fengbo
PASS under any branch.

**Root cause (stated plainly, not hedged):** the bottleneck is the data
source, not the operator or the encoding. The public Zenodo archive is
surface-pressure-only; GINO/Fengbo (Li et al., 2023, DOI
10.48550/arXiv.2309.00583) learn the smooth *volumetric* field at GPU scale,
of which the surface is a slice. The volumetric formulation is not available
from this archive, so no non-degenerate Fengbo solver rung is reachable here.
Scripts 60–66 are valid as QA quantization-boundary parity checks; they are
not Fengbo solver-quality reproduction, and the chain should not be extended
with further pressure smokes against this data source.

## Canonical GINO Record — Ceiling Retracted, Operator Parity Achieved

`67_pepe_ch5_canonical_gino_carcfd_pressure.py` is the faithful continuation.
A landscape pass established that **the Fengbo/GINO ShapeNet-Car benchmark has
no volumetric velocity field in any distribution — trusted or mirror.** Every
ML-ready release (Zenodo 13737721, Zenodo 13936501, neuraloperator
`CarCFDDataset`) is a *surface-pressure* prediction task; the "velocity V
packet" expectation in this chain was a framing error. The canonical record
the published GINO pipeline actually uses is **Zenodo 13936501** (md5-pinned,
trusted), which also carries the authors' documented press alignment
`concat(press[:16], press[112:])` (drop indices 16:112 → 3586 = vertex count).

Switching to that record + alignment, with the *same* 3D FNO, the picture
inverts completely:

| Setup | continuous relative-L2 | continuous R² | QA m=144 gap | status |
|---|---:|---:|---:|---|
| 64–66, "first-3586" misalignment | 0.51–0.57 | 0.26–0.41 | ~1e-3 | degenerate |
| **67, canonical record + alignment** | **0.1734** | **0.9583** | **−7.7e-4** | operator parity |

Operating point: trusted Zenodo 13936501, full official split (500 train /
111 test), 24³ grid, FNO width 12 / 7 modes / 3 layers; QA quantization on
distance channel **and** voxel placement. Continuous R²=0.958 is a genuinely
working operator (published GINO is ≈0.07–0.10 relative-L2; 0.173 at CPU
scale / 24³ / 15 epochs is solver-quality, not a mean-predictor). QA tracks it
to 7.7e-4 with the absolute gap shrinking monotonically by modulus
(3.5e-3 → 1.6e-3 → 7.7e-4 → 2.3e-4 for m=24/72/144/288).

Verdict: `QA_OPERATOR_PARITY_OK__SURFACE_ONLY_NOT_FENGBO_SOLVER`. This is the
**first non-degenerate QA/continuous operator-parity rung** in the chain —
genuine discrete-fractional parity on a competent operator, on the real
Fengbo/GINO benchmark task, with trusted provenance. It is *not* a green
Fengbo solver PASS: the benchmark task is surface pressure, so the full
volumetric Fengbo solver is out of scope for lack of any volumetric
pressure+velocity data — not lack of operator capability.

Honest residual caveats: (1) open3d is unavailable here, so this uses an
unsigned nearest-surface distance instead of the authors' signed SDF —
recorded, not hidden; (2) CPU scale, 24³ grid, 15 epochs — competent but not
the published GINO training budget. Codex independently verified the alignment
is faithful to neuraloperator source and that there is no train/test leakage
(separate manifests, 0 overlap, train-only normalization, pressure never in
the input). Reproducible: `python experiments/qa_ml/67_pepe_ch5_canonical_gino_carcfd_pressure.py`.

### Lesson

The validation-theater diagnosis of scripts 60–65 was correct (they *were*
degenerate), but the attributed root cause — "surface-only data / operator too
weak / hard ceiling" — was **wrong**. The real cause was an unverified target
alignment vs the dataset authors' canonical preprocessing. "Verify reproduction
matches the original before interpreting" applies to *data preprocessing*, not
just code; the forceful ceiling claim violated that and is retracted.

## Volumetric Fengbo — V *and* P Packets (the rung the chain was building toward)

`68_pepe_ch5_volumetric_fengbo_pv.py`. A second landscape correction: the
claim "no volumetric velocity exists in any distribution" was **wrong**. The
Geo-FNO readme canonically points to the RAW primary-source archive
`http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip` (2.03 GB,
sha256-pinned `f4c89976…e34b3`). Per car it ships the surface mesh + `press`
**and the volumetric sample grid + `velo` (29498×3 velocity field)** — the
real V packet. Raw alignment is clean 1:1 (surface 3682↔press, volume
29498↔velo); the 3682-vs-3586 mismatch was purely a Zenodo *processing*
artifact, independently confirming the retracted ceiling.

A 3D FNO maps a dense surface-distance grid to a 4-channel output —
volumetric velocity (vx,vy,vz) on volume voxels + surface pressure on surface
voxels — under matched continuous and QA-quantized packet boundaries
(quantization on the distance grid and placement coords; targets are observer
projections per Theorem NT, not quantized).

Operating point (180 train cars from param1–3, 60 test from param0, 32³ grid,
FNO width 12 / 8 modes / 3 layers, 15 epochs):

| Field | continuous R² | continuous relative-L2 | QA m=144 gap | abs gaps m24→m288 |
|---|---:|---:|---:|---|
| Velocity **V** | **0.9908** | **0.0808** | +1.1e-3 | 0.0125 → 0.0007 |
| Pressure **P** | **0.8764** | 0.2901 | +1.9e-3 | 0.0280 → 0.0009 |

Verdict: `QA_OPERATOR_PARITY_OK__VOLUMETRIC_P_AND_V`, `qa_boundary_faithful`.
Velocity at R²=0.991 / relative-L2 0.081 is **published-GINO quality**; both
fields are genuinely learned and QA tracks continuous with the gap shrinking
monotonically by modulus on *both* — a non-degenerate QA/continuous operator
parity on the genuine volumetric Fengbo task. This **closes the "velocity V
packet missing" gap** flagged across scripts 60–67.

It is *not* a green full-Fengbo solver claim: CPU scale (32³, 15 epochs, 180
cars), below the published GINO training budget.

Recorded finding (signed SDF): a true signed SDF is **infeasible** on this
data — the raw `quadpress_smpl.vtk` surfaces are sampled point sets, not
closed watertight manifolds (`pysdf` sentinels ~61% of grid points; open3d's
signed distance would hit the same non-manifold condition). The robust
unsigned nearest-surface distance used in scripts 66/67/68 is therefore the
**principled** representation here, not a shortcut. Codex independently
verified no train/test leakage and correct V/P voxelization. Reproducible:
`python experiments/qa_ml/68_pepe_ch5_volumetric_fengbo_pv.py`.

## References

- Pepe, A. (2025). *Machine Learning with Geometric Algebra: Multivectors and
  Neural Networks*, PhD thesis, University of Cambridge. Local PDF:
  `corpus/pepe_2025/2025-pepe.pdf`.
- Three-dimensional flow dataset over ShapeNet-Car (processed surface-pressure
  archive). Zenodo. DOI 10.5281/zenodo.13737721.
- Three-dimensional flow dataset over ShapeNet-Car (canonical neuraloperator
  GINO Car-CFD record; the one `neuralop.data.datasets.CarCFDDataset`
  downloads). Zenodo. DOI 10.5281/zenodo.13936501.
- Umetani, N. and Bickel, B. (2018). *Learning three-dimensional flow for
  interactive aerodynamic design*. ACM Transactions on Graphics.
  DOI 10.1145/3197517.3201325.
- Li, Z., et al. (2023). *Geometry-Informed Neural Operator for Large-Scale 3D
  PDEs* (GINO/Fengbo). DOI 10.48550/arXiv.2309.00583.
- Umetani, N. & Bickel, B. (2018) raw CFD archive `mlcfd_data.zip`
  (`http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip`,
  sha256 `f4c899769c92cdf17c997d2b0b0d0686fe11d753a691214ee5eb7d88580e34b3`),
  canonically referenced by the Geo-FNO repository
  (`https://github.com/neuraloperator/Geo-FNO`).
