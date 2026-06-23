import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean, QAFibNatPeriodicity.lean,
--  and QAFibNatMinimalPeriod.lean private helpers)
theorem fib_vec_period_iff (m : ℕ) : fib_vec m = fib_vec 0 ↔ 24 ∣ m := by
  constructor
  · intro h
    -- reduce mod 24; fib_vec_ne_fib_vec_zero rules out 1..23
    have hmod : fib_vec (m % 24) = fib_vec 0 := by
      rwa [← fib_vec_mod24]
    have hlt : m % 24 < 24 := Nat.mod_lt _ (by norm_num)
    rcases Nat.eq_zero_or_pos (m % 24) with h0 | hpos
    · exact Nat.dvd_of_mod_eq_zero h0
    · exact absurd hmod (fib_vec_ne_fib_vec_zero ⟨m % 24, hlt⟩ (by omega))
  · rintro ⟨k, rfl⟩
    rw [fib_vec_mod24]
    norm_num [Nat.mul_mod_right]
