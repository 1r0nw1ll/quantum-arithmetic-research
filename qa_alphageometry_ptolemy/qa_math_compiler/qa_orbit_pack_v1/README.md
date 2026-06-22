# qa_orbit_pack_v1

Machine-checked Lean 4 proofs of core QA structural claims.

QAOrbits.lean (5 theorems): QA Pythagorean identity, singularity fixed-point,
satellite period 8, cosmos period 24, universal period-24 bound.

QAOrbitPartition.lean (6 theorems): three-orbit partition (81 = 72+8+1),
cosmos cardinality 72, exact cosmos period 24, exact satellite period 8,
singularity uniqueness, exact Pisano period π(9) = 24.

Each example is a standalone extract with a single-theorem proof file.

## Cert References

- `qa_orbit01_cfgpythag` → cert [496] ESC_PYTH
- `qa_orbit02_singularity_fixed` → cert [153] DOMINANT=SINGULARITY
- `qa_orbit03_satellite_period_8` → cert [126] orbit-structure
- `qa_orbit04_cosmos_period_24` → cert [128] SP3
- `qa_orbit05_t_period_divides_24` → cert [128] SP2
- `qa_orbit06_cosmos_card` → cert [126] / [191]
- `qa_orbit07_orbit_partition` → cert [126] / [191] (three-orbit partition)
- `qa_orbit08_cosmos_period_exact` → cert [128] SP3 (exact period)
- `qa_orbit09_satellite_period_exact` → cert [126] (exact period)
- `qa_orbit10_singularity_unique` → cert [153] (unique fixed point)
- `qa_orbit11_pisano_9_exact` → cert [128] SP2 (π(9) = 24 exact)
- `qa_orbit12_cosmos_invariant` → cert [126] (Cosmos T-invariant)
- `qa_orbit13_satellite_invariant` → cert [126] (Satellite T-invariant)
- `qa_orbit14_cosmos_orbit1_card` → cert [126] / [128] (sub-orbit size 24)
- `qa_orbit15_cosmos_orbit12_disjoint` → cert [126] (sub-orbits disjoint)
- `qa_orbit16_cosmos_suborbit_union` → cert [126] / [191] (sub-orbit decomp)
- `qa_orbit17_cosmos_step_injective` → cert [126] (T bijective on Cosmos)
- `qa_orbit18_cosmos_reps_distinct` → cert [126] / [128] (three distinct orbits)
