import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.GroupTheory.SpecificGroups.Cyclic
import Mathlib.Tactic
import QAFibMatrixGroup

/-!
# Explicit isomorphism ⟨F⟩ ≅ ℤ/24ℤ via Mathlib's IsCyclic API

Constructs a concrete group isomorphism from `Multiplicative (ZMod 24)` to the
cyclic subgroup `⟨F⟩ = Subgroup.zpowers fib_mat_unit` in `(M₂(ZMod 9))ˣ`,
using Mathlib's `zmodMulEquivOfGenerator` with the explicit generator F:

  `Multiplicative.ofAdd k  ↦  F^k`

This is the full group-theoretic statement of the Pisano period:
the matrix F generates a cyclic subgroup of order 24, isomorphic to ℤ/24ℤ.

## Cert references

- `[128]` Pisano period π(9) = 24 — the order statement lifts to the isomorphism.
- `[126]` orbit structure — ⟨F⟩ acts as the three QA orbit structure.
-/

-- ============================================================================
-- GENERATOR OF ⟨F⟩ IN THE SUBGROUP TYPE
-- ============================================================================

/-- The generator of ⟨F⟩: `fib_mat_unit` viewed as an element of the subgroup. -/
private def fib_gen : ↥(Subgroup.zpowers fib_mat_unit) :=
  ⟨fib_mat_unit, Subgroup.mem_zpowers fib_mat_unit⟩

/-- Every element of `⟨F⟩` is a power of `fib_gen`.

    Each `⟨x, hx⟩ ∈ Subgroup.zpowers fib_mat_unit` satisfies
    `x = fib_mat_unit^k` for some `k : ℤ`, so `⟨x, hx⟩ = fib_gen^k ∈ zpowers fib_gen`. -/
private theorem fib_gen_generates :
    ∀ x : ↥(Subgroup.zpowers fib_mat_unit), x ∈ Subgroup.zpowers fib_gen := by
  intro ⟨x, hx⟩
  obtain ⟨k, rfl⟩ := Subgroup.mem_zpowers_iff.mp hx
  exact Subgroup.mem_zpowers_iff.mpr ⟨k, Subtype.ext (by simp [fib_gen])⟩

-- ============================================================================
-- Nat.card ⟨F⟩ = 24
-- ============================================================================

/-- The `Nat.card` of `⟨F⟩` is 24 (from `Fintype.card_zpowers` and `orderOf = 24`). -/
theorem fib_mat_zpowers_nat_card :
    Nat.card ↥(Subgroup.zpowers fib_mat_unit) = 24 := by
  rw [Nat.card_eq_fintype_card]
  exact fib_mat_zpowers_card

-- ============================================================================
-- THE EXPLICIT ISOMORPHISM
-- ============================================================================

/-- **Explicit isomorphism ⟨F⟩ ≅ ℤ/24ℤ** via `zmodMulEquivOfGenerator`.

    This is Mathlib's `zmodMulEquivOfGenerator` instantiated at the Fibonacci
    matrix unit, with explicit generator `fib_gen = F` and cardinality 24.
    The isomorphism sends `Multiplicative.ofAdd k ↦ fib_gen^k = F^k`. -/
noncomputable def fib_mat_iso_ZMod24 :
    Multiplicative (ZMod 24) ≃* ↥(Subgroup.zpowers fib_mat_unit) :=
  zmodMulEquivOfGenerator fib_gen_generates fib_mat_zpowers_nat_card

-- ============================================================================
-- PROPERTIES OF THE ISOMORPHISM
-- ============================================================================

/-- The isomorphism sends `Multiplicative.ofAdd 1` to the generator F. -/
theorem fib_mat_iso_maps_generator :
    fib_mat_iso_ZMod24 (Multiplicative.ofAdd 1) = fib_gen :=
  zmodMulEquivOfGenerator_apply_ofAdd_one fib_gen_generates fib_mat_zpowers_nat_card

/-- The isomorphism sends `Multiplicative.ofAdd k` to `F^k` for any `k : ℤ`. -/
theorem fib_mat_iso_zpow (k : ℤ) :
    fib_mat_iso_ZMod24 (Multiplicative.ofAdd k) = fib_gen ^ k :=
  zmodMulEquivOfGenerator_apply_ofAdd_intCast fib_gen_generates fib_mat_zpowers_nat_card k

/-- The inverse isomorphism sends F to `Multiplicative.ofAdd 1`. -/
theorem fib_mat_iso_symm_generator :
    fib_mat_iso_ZMod24.symm fib_gen = Multiplicative.ofAdd 1 :=
  zmodMulEquivOfGenerator_symm_apply_generator fib_gen_generates fib_mat_zpowers_nat_card

/-- The inverse isomorphism sends `F^k` to `Multiplicative.ofAdd k`. -/
theorem fib_mat_iso_symm_zpow (k : ℤ) :
    fib_mat_iso_ZMod24.symm (fib_gen ^ k) = Multiplicative.ofAdd (k : ZMod 24) :=
  zmodMulEquivOfGenerator_symm_apply_zpow fib_gen_generates fib_mat_zpowers_nat_card k
