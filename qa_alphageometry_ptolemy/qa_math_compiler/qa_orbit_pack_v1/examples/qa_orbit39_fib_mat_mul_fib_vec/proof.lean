import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean and QAFibNatPeriodicity.lean)
theorem fib_mat_mul_fib_vec (n : ℕ) :
    fib_mat *ᵥ fib_vec n = fib_vec (n + 1) := by
  funext i
  fin_cases i
  · simp [fib_mat, fib_vec, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    push_cast [Nat.fib_add_two]; ring
  · simp [fib_mat, fib_vec, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
