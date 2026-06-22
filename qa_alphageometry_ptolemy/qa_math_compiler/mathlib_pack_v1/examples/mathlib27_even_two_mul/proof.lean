import Mathlib.Algebra.Ring.Parity

theorem qa_mathlib27_even_two_mul (n : ℕ) : Even (2 * n) := by
  exact ⟨n, by omega⟩
