import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Nat.Fib.Basic
import Mathlib.Tactic

open scoped Matrix

-- (prerequisite defs in QAFibMatrix.lean and QAFibNatPeriodicity.lean)
theorem fib_mat_pow_eq_one_iff (m : ℕ) : fib_mat ^ m = 1 ↔ 24 ∣ m := by
  constructor
  · intro h
    have hpow : fib_mat ^ (m % 24) = 1 := by
      have : fib_mat ^ m = fib_mat ^ (m % 24) := by
        conv_lhs => rw [show m = 24 * (m / 24) + m % 24 from by omega]
        rw [pow_add, pow_mul, fib_mat_pow_24, one_pow, one_mul]
      rw [← this]; exact h
    have hlt : m % 24 < 24 := Nat.mod_lt _ (by norm_num)
    rcases Nat.eq_zero_or_pos (m % 24) with h0 | hpos
    · exact Nat.dvd_of_mod_eq_zero h0
    · exact absurd hpow (fib_mat_order_exact ⟨m % 24, hlt⟩ (by omega))
  · rintro ⟨k, rfl⟩
    rw [pow_mul, fib_mat_pow_24, one_pow]
