import Mathlib.Data.Nat.Factorial.Basic

theorem qa_mathlib24_factorial_pos (n : ℕ) : 0 < Nat.factorial n := by
  exact Nat.factorial_pos n
