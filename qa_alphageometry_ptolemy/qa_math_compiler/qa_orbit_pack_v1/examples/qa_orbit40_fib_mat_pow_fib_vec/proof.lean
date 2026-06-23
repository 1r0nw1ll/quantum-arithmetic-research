import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean and QAFibNatPeriodicity.lean)
theorem fib_mat_pow_fib_vec (n m : ℕ) :
    (fib_mat ^ n) *ᵥ fib_vec m = fib_vec (n + m) := by
  induction n generalizing m with
  | zero => simp [fib_vec]
  | succ n ih =>
    rw [pow_succ', ← Matrix.mulVec_mulVec, ih, fib_mat_mul_fib_vec]
    congr 1; omega
