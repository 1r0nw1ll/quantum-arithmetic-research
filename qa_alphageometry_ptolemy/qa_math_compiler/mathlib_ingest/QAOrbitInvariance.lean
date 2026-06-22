import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Tactic

/-!
# QA Orbit Invariance and Sub-orbit Decomposition

Formal foundations for the invariance and internal structure of the three QA
mod-9 orbits under the T-step `(b, e) ↦ (b+e, b)`.

## Main theorems

- **Orbit invariance**: T maps each orbit to itself (Cosmos, Satellite, Singularity
  are all T-invariant sets).
- **Sub-orbit decomposition**: The Cosmos splits into exactly 3 pairwise-disjoint
  T-orbits of size 24 each (reps: (1,0), (2,0), (4,0)).
- **T is a bijection on each orbit**: T restricted to any orbit is a bijection.

## Cert references

- `[126]` orbit structure (Cosmos, Satellite, Singularity)
- `[128]` Pisano period π(9) = 24
- `[191]` reachability tier counts (72, 8, 1)
-/

/-- QA T-step in ZMod 9. -/
def qa_step' (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

-- ============================================================================
-- ORBIT FINSETS (copied inline for self-containment)
-- ============================================================================

def qa_cosmos' : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 24).image (fun k => (qa_step'^[k]) (1, 0)) ∪
  (Finset.range 24).image (fun k => (qa_step'^[k]) (2, 0)) ∪
  (Finset.range 24).image (fun k => (qa_step'^[k]) (4, 0))

def qa_satellite' : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 8).image (fun k => (qa_step'^[k]) (6, 3))

def qa_singularity' : Finset (ZMod 9 × ZMod 9) := {(0, 0)}

-- ============================================================================
-- ORBIT INVARIANCE
-- ============================================================================

/-- **Cosmos is T-invariant** (cert [126]).

    Every state in the Cosmos maps to another Cosmos state under the T-step.
    The Cosmos is a union of T-orbits and hence closed under T. -/
theorem qa_cosmos_invariant :
    ∀ s ∈ qa_cosmos', qa_step' s ∈ qa_cosmos' := by native_decide

/-- **Satellite is T-invariant** (cert [126]).

    Every state in the Satellite maps to another Satellite state under the T-step. -/
theorem qa_satellite_invariant :
    ∀ s ∈ qa_satellite', qa_step' s ∈ qa_satellite' := by native_decide

/-- **Singularity is T-invariant** (cert [153]).

    The fixed point (0,0) maps to itself. Trivially closed. -/
theorem qa_singularity_invariant :
    ∀ s ∈ qa_singularity', qa_step' s ∈ qa_singularity' := by native_decide

-- ============================================================================
-- THREE COSMOS SUB-ORBITS
-- ============================================================================

/-- The three Cosmos sub-orbit representatives: (1,0), (2,0), (4,0). -/
def qa_cosmos_orbit1 : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 24).image (fun k => (qa_step'^[k]) (1, 0))

def qa_cosmos_orbit2 : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 24).image (fun k => (qa_step'^[k]) (2, 0))

def qa_cosmos_orbit3 : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 24).image (fun k => (qa_step'^[k]) (4, 0))

/-- Each cosmos sub-orbit has exactly 24 elements. -/
theorem qa_cosmos_orbit1_card : qa_cosmos_orbit1.card = 24 := by native_decide
theorem qa_cosmos_orbit2_card : qa_cosmos_orbit2.card = 24 := by native_decide
theorem qa_cosmos_orbit3_card : qa_cosmos_orbit3.card = 24 := by native_decide

/-- The three cosmos sub-orbits are pairwise disjoint (cert [126]).

    The zero-divisors of ZMod 9 cause the Fibonacci map to generate three
    distinct orbits from the representatives (1,0), (2,0), (4,0). -/
theorem qa_cosmos_orbit12_disjoint :
    Disjoint qa_cosmos_orbit1 qa_cosmos_orbit2 := by native_decide

theorem qa_cosmos_orbit13_disjoint :
    Disjoint qa_cosmos_orbit1 qa_cosmos_orbit3 := by native_decide

theorem qa_cosmos_orbit23_disjoint :
    Disjoint qa_cosmos_orbit2 qa_cosmos_orbit3 := by native_decide

/-- The three cosmos sub-orbits exhaust the Cosmos (cert [126]).

    Cosmos = orbit(1,0) ∪ orbit(2,0) ∪ orbit(4,0), with all three disjoint
    and of size 24 each, totaling 72 states. -/
theorem qa_cosmos_suborbit_union :
    qa_cosmos_orbit1 ∪ qa_cosmos_orbit2 ∪ qa_cosmos_orbit3 = qa_cosmos' := by
  native_decide

-- ============================================================================
-- T IS A BIJECTION ON EACH ORBIT
-- ============================================================================

/-- T is injective on the Cosmos: distinct states map to distinct images. -/
theorem qa_cosmos_step_injective :
    ∀ s ∈ qa_cosmos', ∀ t ∈ qa_cosmos', qa_step' s = qa_step' t → s = t := by native_decide

/-- T is injective on the Satellite. -/
theorem qa_satellite_step_injective :
    ∀ s ∈ qa_satellite', ∀ t ∈ qa_satellite', qa_step' s = qa_step' t → s = t := by native_decide

-- ============================================================================
-- ORBIT DISTINCTNESS
-- ============================================================================

/-- No cosmos sub-orbit rep is reachable from another (cert [128]).

    The three T-orbits within the Cosmos are genuinely distinct: you cannot
    reach the rep of orbit 2 or 3 from the rep of orbit 1 by any number of
    T-steps (and vice versa). -/
theorem qa_cosmos_reps_distinct :
    ∀ k : ℕ, k < 24 → (qa_step'^[k]) (1, 0) ≠ (2, 0) ∧
                        (qa_step'^[k]) (1, 0) ≠ (4, 0) ∧
                        (qa_step'^[k]) (2, 0) ≠ (4, 0) := by native_decide
