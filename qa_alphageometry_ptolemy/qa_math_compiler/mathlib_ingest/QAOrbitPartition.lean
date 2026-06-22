import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic

/-!
# QA Three-Orbit Partition Theorems

Formal foundations for the partition of QA mod-9 state space into three orbits
under the T-step `(b, e) ↦ (b+e, b)`.

## The three orbits

- **Cosmos**: 72 states, three sub-orbits of size 24 each (reps: (1,0), (2,0), (4,0)), period 24
- **Satellite**: 8 states, single orbit of rep (6,3), period 8
- **Singularity**: 1 state, the fixed point (0,0)

These three sets are pairwise disjoint, exhaustive, and account for all 81 states.

## Orbit structure

The T-step acts on ZMod 9 × ZMod 9 by the Fibonacci matrix F = [[1,1],[1,0]].
Since ZMod 9 has nontrivial zero-divisors (3·3=0), the orbits do not form a
single cycle. The 81 states decompose as:
- 3 orbits of size 24 = 72 cosmos states (reps: (1,0), (2,0), (4,0))
- 1 orbit of size 8  =  8 satellite states (rep: (6,3))
- 1 orbit of size 1  =  1 singularity state (rep: (0,0))

## Cert references

- `[126]` orbit structure (Cosmos, Satellite, Singularity)
- `[128]` Pisano period π(9) = 24
- `[153]` singularity dominance: (0,0) is the unique fixed point
- `[191]` reachability tier counts (72, 8, 1)
- `[211]` Cayley-Bateson filtration: connected components match tier counts
-/

/-- QA T-step in ZMod 9. -/
def qa_step (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

-- ============================================================================
-- ORBIT FINSETS
-- ============================================================================

/-- The Cosmos orbit: union of the three 24-element orbits with reps (1,0), (2,0), (4,0). -/
def qa_cosmos : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 24).image (fun k => (qa_step^[k]) (1, 0)) ∪
  (Finset.range 24).image (fun k => (qa_step^[k]) (2, 0)) ∪
  (Finset.range 24).image (fun k => (qa_step^[k]) (4, 0))

/-- The Satellite orbit: the single 8-element orbit with rep (6,3). -/
def qa_satellite : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 8).image (fun k => (qa_step^[k]) (6, 3))

/-- The Singularity: the fixed point {(0, 0)}. -/
def qa_singularity : Finset (ZMod 9 × ZMod 9) := {(0, 0)}

-- ============================================================================
-- CARDINALITIES
-- ============================================================================

/-- The Cosmos orbit has exactly 72 states (three sub-orbits of 24 each). -/
theorem qa_cosmos_card : qa_cosmos.card = 72 := by native_decide

/-- The Satellite orbit has exactly 8 states. -/
theorem qa_satellite_card : qa_satellite.card = 8 := by native_decide

/-- The Singularity is a single state. -/
theorem qa_singularity_card : qa_singularity.card = 1 := by decide

-- ============================================================================
-- PARTITION THEOREM (cert [126], [191])
-- ============================================================================

/-- The three orbits are pairwise disjoint: Cosmos ∩ Satellite = ∅. -/
theorem qa_cosmos_satellite_disjoint : Disjoint qa_cosmos qa_satellite := by native_decide

/-- The three orbits are pairwise disjoint: Cosmos ∩ Singularity = ∅. -/
theorem qa_cosmos_singularity_disjoint : Disjoint qa_cosmos qa_singularity := by native_decide

/-- The three orbits are pairwise disjoint: Satellite ∩ Singularity = ∅. -/
theorem qa_satellite_singularity_disjoint : Disjoint qa_satellite qa_singularity := by native_decide

/-- **Three-orbit partition theorem** (cert [126] / [191]).

    The 81 states of ZMod 9 × ZMod 9 are partitioned exactly into:
    Cosmos (72 states), Satellite (8 states), Singularity (1 state).

    Every QA mod-9 state belongs to exactly one of these three orbits.
    Equivalently, the Fibonacci matrix F has exactly three orbit types on ZMod 9²:
    period-24 (×3 orbits), period-8 (×1 orbit), period-1 (×1 orbit). -/
theorem qa_orbit_partition :
    qa_cosmos ∪ qa_satellite ∪ qa_singularity = Finset.univ := by native_decide

-- ============================================================================
-- EXACT PERIODS (cert [128] SP2/SP3)
-- ============================================================================

/-- **Cosmos period is exactly 24** (cert [128] SP3).

    For every k ∈ {1,...,23}, (1,0) is NOT returned by k T-steps.
    Combined with `qa_orbit04_cosmos_period_24` (QAOrbits.lean), the minimal period is 24. -/
theorem qa_cosmos_period_exact :
    ∀ k : Fin 24, k.val ≠ 0 → (qa_step^[k.val]) (1, 0) ≠ (1, 0) := by native_decide

/-- **Satellite period is exactly 8** (cert [126]).

    For every k ∈ {1,...,7}, (6,3) is NOT returned by k T-steps.
    Combined with `qa_orbit03_satellite_period_8` (QAOrbits.lean), the minimal period is 8. -/
theorem qa_satellite_period_exact :
    ∀ k : Fin 8, k.val ≠ 0 → (qa_step^[k.val]) (6, 3) ≠ (6, 3) := by native_decide

-- ============================================================================
-- SINGULARITY UNIQUENESS (cert [153])
-- ============================================================================

/-- **The singularity (0,0) is the unique fixed point** (cert [153] DOMINANT=SINGULARITY).

    A state s is a fixed point of the T-step if and only if s = (0,0).
    No Cosmos or Satellite state is a fixed point. -/
theorem qa_singularity_unique :
    ∀ s : ZMod 9 × ZMod 9, qa_step s = s ↔ s = (0, 0) := by native_decide

-- ============================================================================
-- EXACT PISANO PERIOD π(9) = 24 (cert [128] SP2)
-- ============================================================================

/-- **Exact Pisano period: π(9) = 24** (cert [128] SP2).

    For every k ∈ {1,...,23}, the map T^k is NOT the identity on ZMod 9 × ZMod 9.
    Equivalently, the Fibonacci matrix F = [[1,1],[1,0]] has exact order 24
    as a map on ZMod 9 × ZMod 9 (not just divisor of 24).

    Combined with `qa_t_period_divides_24` (QAOrbits.lean: T^24 = id), this
    establishes the exact order: the smallest k > 0 with T^k = id is k = 24. -/
theorem qa_pisano_9_exact :
    ∀ k : Fin 24, k.val ≠ 0 →
      ∃ s : ZMod 9 × ZMod 9, (qa_step^[k.val]) s ≠ s := by native_decide

/-- **Explicit witness for Pisano non-identity** (cert [128]).

    (1,0) is a witness: for every k ∈ {1,...,23}, T^k (1,0) ≠ (1,0).
    This is the concrete "non-trivial" form of `qa_pisano_9_exact`. -/
theorem qa_pisano_9_witness :
    ∀ k : Fin 24, k.val ≠ 0 → (qa_step^[k.val]) (1, 0) ≠ (1, 0) :=
  qa_cosmos_period_exact

-- ============================================================================
-- COROLLARIES
-- ============================================================================

/-- The three orbit sizes sum to 81 = |ZMod 9 × ZMod 9|. -/
theorem qa_orbit_size_total :
    qa_cosmos.card + qa_satellite.card + qa_singularity.card = Fintype.card (ZMod 9 × ZMod 9) := by
  native_decide

/-- Every state belongs to at least one of the three orbits. -/
theorem qa_orbit_membership (s : ZMod 9 × ZMod 9) :
    s ∈ qa_cosmos ∨ s ∈ qa_satellite ∨ s ∈ qa_singularity := by
  have h : s ∈ qa_cosmos ∪ qa_satellite ∪ qa_singularity := by
    rw [qa_orbit_partition]; exact Finset.mem_univ s
  simp [Finset.mem_union] at h
  tauto

/-- The three orbit reps (1,0), (6,3), (0,0) are in their respective orbits. -/
theorem qa_orbit_reps :
    (1, 0) ∈ qa_cosmos ∧ (6, 3) ∈ qa_satellite ∧ (0, 0) ∈ qa_singularity := by
  native_decide
