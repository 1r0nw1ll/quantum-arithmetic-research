import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]
private def fib_mat_inv_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![0, 1; 1, 8]

def fib_mat_unit : (Matrix (Fin 2) (Fin 2) (ZMod 9))ˣ :=
  ⟨fib_mat, fib_mat_inv_mat, by native_decide, by native_decide⟩

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
