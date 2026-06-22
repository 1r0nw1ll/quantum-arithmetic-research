import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]
private def fib_mat_inv_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![0, 1; 1, 8]

def fib_mat_unit : (Matrix (Fin 2) (Fin 2) (ZMod 9))ˣ :=
  ⟨fib_mat, fib_mat_inv_mat, by native_decide, by native_decide⟩

theorem fib_mat_unit_pow_24 : fib_mat_unit ^ 24 = 1 := by
  apply Units.val_injective
  simp only [Units.val_pow_eq_pow_val, Units.val_one,
             show (fib_mat_unit : Matrix (Fin 2) (Fin 2) (ZMod 9)) = fib_mat from rfl]
  native_decide

theorem fib_mat_unit_order_exact :
    ∀ k : Fin 24, k.val ≠ 0 → fib_mat_unit ^ k.val ≠ 1 := by
  intro k hk heq
  have hmat : fib_mat ^ k.val = 1 := by
    have h := congr_arg Units.val heq
    simp only [Units.val_pow_eq_pow_val, Units.val_one,
               show (fib_mat_unit : Matrix (Fin 2) (Fin 2) (ZMod 9)) = fib_mat from rfl] at h
    exact h
  have key : ∀ j : Fin 24, j.val ≠ 0 → fib_mat ^ j.val ≠ 1 := by native_decide
  exact key k hk hmat

theorem fib_mat_unit_orderOf : orderOf fib_mat_unit = 24 := by
  rw [orderOf_eq_iff (by norm_num : 0 < 24)]
  refine ⟨fib_mat_unit_pow_24, fun m hm24 hm1 heq => ?_⟩
  exact fib_mat_unit_order_exact ⟨m, hm24⟩ (Nat.pos_iff_ne_zero.mp hm1) heq

theorem fib_mat_zpowers_card :
    Fintype.card ↥(Subgroup.zpowers fib_mat_unit) = 24 := by
  rw [Fintype.card_zpowers]
  exact fib_mat_unit_orderOf
