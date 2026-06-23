import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean and QAFibNatPeriodicity.lean)
theorem fib_nat_mod9_periodic (n : ℕ) :
    (Nat.fib (n + 24) : ZMod 9) = Nat.fib n := by
  have h := congr_fun (fib_vec_periodic n) 1
  simpa [fib_vec] using h
