# QA-ML Pepe Chapter 5 PDE Solver Mapping

Status: corrective source map, 2026-05-18.

Machine artifact:
`experiments/qa_ml/results_pepe_ch5_pde_solver_source_map.json`.

First primitive artifact:
`experiments/qa_ml/results_pepe_ch5_qa_fengbo_packet_parity.json`.

## Correction

The Chapter 5 visual replica is a scaffold, not a PDE solver replication.
The honest status is:

| Pepe solver | Current QA status | Next required primitive |
|---|---|---|
| GA-ReLU, 2D Navier-Stokes | Mapped; solver replica pending | QA-GA-ReLU over quantized vector phase packets |
| Fengbo, 3D irregular CFD | Mapped; packet parity smoke PASS; neural operator pending | QA-Fengbo operator over quantized geometry packets |
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
4. Run continuous mini-Fengbo and QA-quantized mini-Fengbo head-to-head.
5. Only after mini parity, acquire ShapeNet Car / Ahmed Body and attempt the
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
