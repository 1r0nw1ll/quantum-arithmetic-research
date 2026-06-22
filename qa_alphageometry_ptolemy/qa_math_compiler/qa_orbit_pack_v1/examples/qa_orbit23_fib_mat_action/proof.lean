import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]

theorem fib_mat_action :
    ∀ b e : ZMod 9,
    Matrix.mulVec fib_mat ![b, e] = ![b + e, b] := by native_decide
