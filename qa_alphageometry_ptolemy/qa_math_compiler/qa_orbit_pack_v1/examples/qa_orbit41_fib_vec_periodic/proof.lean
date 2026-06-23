import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean and QAFibNatPeriodicity.lean)
theorem fib_vec_periodic (n : ℕ) : fib_vec (n + 24) = fib_vec n := by
  have key : fib_vec (24 + n) = fib_vec n :=
    calc fib_vec (24 + n)
        = (fib_mat ^ 24) *ᵥ fib_vec n := (fib_mat_pow_fib_vec 24 n).symm
      _ = (1 : Matrix (Fin 2) (Fin 2) (ZMod 9)) *ᵥ fib_vec n := by rw [fib_mat_pow_24]
      _ = fib_vec n := Matrix.one_mulVec _
  rwa [Nat.add_comm] at key
