import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean, QAFibNatPeriodicity.lean,
--  and QAFibNatMinimalPeriod.lean)
theorem pisano_period_9_exact (m : ℕ) :
    (∀ n : ℕ, (Nat.fib (n + m) : ZMod 9) = Nat.fib n) ↔ 24 ∣ m := by
  rw [← fib_vec_period_iff]
  constructor
  · intro h
    have hm : (Nat.fib m : ZMod 9) = 0 := by
      have := h 0; simp only [Nat.zero_add] at this; simpa using this
    have hm1 : (Nat.fib (m + 1) : ZMod 9) = 1 := by
      have := h 1
      rwa [show 1 + m = m + 1 from by omega] at this
    ext i; fin_cases i
    · simp [fib_vec, hm1]
    · simp [fib_vec, hm]
  · intro heq n
    have key : fib_vec (n + m) = fib_vec n :=
      calc fib_vec (n + m)
          = (fib_mat ^ n) *ᵥ fib_vec m := (fib_mat_pow_fib_vec n m).symm
        _ = (fib_mat ^ n) *ᵥ fib_vec 0 := by rw [heq]
        _ = fib_vec (n + 0) := fib_mat_pow_fib_vec n 0
        _ = fib_vec n := by simp
    simpa [fib_vec] using congr_fun key 1
