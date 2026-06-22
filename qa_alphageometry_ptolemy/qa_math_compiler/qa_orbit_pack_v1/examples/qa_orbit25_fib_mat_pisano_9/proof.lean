import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]

theorem fib_mat_order_exact :
    ∀ k : Fin 24, k.val ≠ 0 → fib_mat ^ k.val ≠ 1 := by native_decide

theorem fib_mat_pow_24 : fib_mat ^ 24 = 1 := by native_decide

theorem fib_mat_pisano_9 :
    (∀ k : Fin 24, k.val ≠ 0 → fib_mat ^ k.val ≠ 1) ∧ fib_mat ^ 24 = 1 :=
  ⟨fib_mat_order_exact, fib_mat_pow_24⟩
