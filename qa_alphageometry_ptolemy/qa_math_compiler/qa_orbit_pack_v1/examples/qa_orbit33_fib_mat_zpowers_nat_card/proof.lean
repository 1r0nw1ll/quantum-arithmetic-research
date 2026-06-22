import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.GroupTheory.SpecificGroups.Cyclic.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]
private def fib_mat_inv_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![0, 1; 1, 8]
def fib_mat_unit : (Matrix (Fin 2) (Fin 2) (ZMod 9))ˣ :=
  ⟨fib_mat, fib_mat_inv_mat, by native_decide, by native_decide⟩
theorem fib_mat_zpowers_card : Fintype.card ↕(Subgroup.zpowers fib_mat_unit) = 24 := by
  rw [Fintype.card_zpowers]
  rw [orderOf_eq_iff (by norm_num : 0 < 24)]
  exact ⟨by apply Units.val_injective; simp [fib_mat_unit]; native_decide,
          fun m hm24 hm1 heq => absurd heq (by native_decide)⟩
theorem fib_mat_zpowers_nat_card :
    Nat.card ↕(Subgroup.zpowers fib_mat_unit) = 24 := by
  rw [Nat.card_eq_fintype_card]
  exact fib_mat_zpowers_card
