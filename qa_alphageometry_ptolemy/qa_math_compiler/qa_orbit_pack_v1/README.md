# qa_orbit_pack_v1

Machine-checked Lean 4 proofs of core QA structural claims.

These are the first QA-native formal theorems: the QA Pythagorean identity,
singularity fixed-point, satellite orbit period 8, cosmos orbit period 24,
and the universal period-24 bound (Pisano period π(9) divides 24).

All proofs are in `QAOrbits.lean` in the mathlib_ingest project.
Each example here is a standalone extract with a single-theorem proof file.

## Cert References

- `qa_orbit01_cfgpythag` → cert [496] ESC_PYTH
- `qa_orbit02_singularity_fixed` → cert [153] DOMINANT=SINGULARITY
- `qa_orbit03_satellite_period_8` → cert [126] orbit-structure
- `qa_orbit04_cosmos_period_24` → cert [128] SP3
- `qa_orbit05_t_period_divides_24` → cert [128] SP2
