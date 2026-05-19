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

## Correction

The Chapter 5 visual replica is a scaffold, not a PDE solver replication.
The honest status is:

| Pepe solver | Current QA status | Next required primitive |
|---|---|---|
| GA-ReLU, 2D Navier-Stokes | Mapped; solver replica pending | QA-GA-ReLU over quantized vector phase packets |
| Fengbo, 3D irregular CFD | Mapped; packet parity PASS; synthetic operator parity PASS; real AhmedML metadata/force smoke PASS | Field/mesh acquisition + pressure/velocity Fengbo smoke |
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
   Fengbo field smoke.
7. Only after real-subset field parity, attempt the
   full Pepe Fengbo reproduction.

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
