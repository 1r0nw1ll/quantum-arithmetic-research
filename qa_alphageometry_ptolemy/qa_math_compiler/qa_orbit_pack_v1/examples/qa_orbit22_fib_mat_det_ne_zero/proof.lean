import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]

theorem fib_mat_det_ne_zero : Matrix.det fib_mat ≠ 0 := by native_decide
