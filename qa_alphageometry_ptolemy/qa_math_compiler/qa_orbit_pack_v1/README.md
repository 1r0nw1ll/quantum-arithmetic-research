# qa_orbit_pack_v1

Machine-checked Lean 4 proofs of core QA structural claims.

QAOrbits.lean (5 theorems): QA Pythagorean identity, singularity fixed-point,
satellite period 8, cosmos period 24, universal period-24 bound.

QAOrbitPartition.lean (11 theorems): three-orbit partition (81 = 72+8+1),
cosmos cardinality 72, satellite cardinality 8, singularity cardinality 1,
pairwise disjointness, exact cosmos period 24, exact satellite period 8,
singularity uniqueness, exact Pisano period π(9) = 24.

QAOrbitInvariance.lean (7 theorems): orbit T-invariance, sub-orbit decomposition,
T injectivity on Cosmos, three sub-orbits pairwise disjoint.

QAFibMatrix.lean (7 theorems): F^24=I and exact order 24 in M₂(ZMod 9),
det(F)=8≠0, T-step = matrix action, iteration = matrix power.

QAFibMatrixGroup.lean (7 theorems): F lifted to unit group GL₂(ZMod 9),
orderOf(F)=24 via orderOf_eq_iff, |⟨F⟩|=24, ⟨F⟩ is cyclic.

QAFibMatrixGroupIso.lean (6 theorems): explicit isomorphism ⟨F⟩ ≅ ℤ/24ℤ via
zmodMulEquivOfGenerator, generator mapping, zpow mapping, inverse mappings.

QAFibNatPeriodicity.lean (4 theorems): Fibonacci sequence periodicity mod 9,
matrix recurrence, iterated action, fib_vec periodic, π(9) = 24 for Nat.fib.

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
- `qa_orbit19_fib_mat_pow_24` → cert [128] SP2 (F^24 = I in M₂(ZMod 9))
- `qa_orbit20_fib_mat_order_exact` → cert [128] SP2 (ord(F) = 24 in GL₂(ZMod 9))
- `qa_orbit21_fib_mat_det` → cert [153] (det(F) = 8 = -1 mod 9)
- `qa_orbit22_fib_mat_det_ne_zero` → cert [153] (F ∈ GL₂(ZMod 9))
- `qa_orbit23_fib_mat_action` → cert [126] (T-step = matrix action)
- `qa_orbit24_fib_mat_iter` → cert [126] / [128] (iterate = matrix power)
- `qa_orbit25_fib_mat_pisano_9` → cert [128] SP2 (π(9) = 24, matrix form)
- `qa_orbit26_fib_mat_unit_pow_24` → cert [128] SP2 (F^24=1 in unit group)
- `qa_orbit27_fib_mat_unit_order_exact` → cert [128] SP2 (exact order in unit group)
- `qa_orbit28_fib_mat_unit_pow_12_ne_one` → cert [128] SP2 (F^12≠1, rules out p=2)
- `qa_orbit29_fib_mat_unit_pow_8_ne_one` → cert [128] SP2 (F^8≠1, rules out p=3)
- `qa_orbit30_fib_mat_unit_orderOf` → cert [128] SP2 (orderOf(F)=24 in GL₂)
- `qa_orbit31_fib_mat_zpowers_card` → cert [128] / [126] (|⟨F⟩|=24)
- `qa_orbit32_fib_mat_zpowers_isCyclic` → cert [126] (⟨F⟩ is cyclic)
- `qa_orbit33_fib_mat_zpowers_nat_card` → cert [128] / [126] (Nat.card ⟨F⟩ = 24)
- `qa_orbit34_fib_mat_iso_ZMod24` → cert [128] SP2 (explicit iso ⟨F⟩ ≅ ℤ/24ℤ)
- `qa_orbit35_fib_mat_iso_maps_generator` → cert [128] SP2 (iso sends 1 → F)
- `qa_orbit36_fib_mat_iso_zpow` → cert [128] SP2 (iso sends k → F^k)
- `qa_orbit37_fib_mat_iso_symm_generator` → cert [128] SP2 (inverse sends F → 1)
- `qa_orbit38_fib_mat_iso_symm_zpow` → cert [128] SP2 (inverse sends F^k → k mod 24)
- `qa_orbit39_fib_mat_mul_fib_vec` → cert [128] (fib_mat *ᵥ fib_vec n = fib_vec (n+1))
- `qa_orbit40_fib_mat_pow_fib_vec` → cert [128] (fib_mat^n *ᵥ fib_vec m = fib_vec (n+m))
- `qa_orbit41_fib_vec_periodic` → cert [128] (fib_vec (n+24) = fib_vec n)
- `qa_orbit42_fib_nat_mod9_periodic` → cert [128] (Fibonacci sequence periodic mod 9 with period 24)
- `qa_orbit43_fib_mat_pow_eq_one_iff` → cert [128] (matrix power equals identity iff 24 divides exponent)
- `qa_orbit44_fib_vec_period_iff` → cert [128] (fib_vec returns to initial state iff 24 divides exponent)
- `qa_orbit45_pisano_period_9_exact` → cert [128] (complete π(9)=24 characterization)
- `qa_orbit46_satellite_card` → cert [126] / [191] (Satellite orbit cardinality 8)
- `qa_orbit47_singularity_card` → cert [153] / [191] (Singularity orbit cardinality 1)
- `qa_orbit48_cosmos_satellite_disjoint` → cert [126] / [191] (Cosmos ∩ Satellite = ∅)
- `qa_orbit49_cosmos_singularity_disjoint` → cert [126] / [153] / [191] (Cosmos ∩ Singularity = ∅)
- `qa_orbit50_satellite_singularity_disjoint` → cert [126] / [153] / [191] (Satellite ∩ Singularity = ∅)
