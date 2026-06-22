import Mathlib.Algebra.Ring.Parity

theorem qa_mathlib26_even_add_even (n m : ℕ) (hn : Even n) (hm : Even m) : Even (n + m) := by
  obtain ⟨k, rfl⟩ := hn
  obtain ⟨l, rfl⟩ := hm
  exact ⟨k + l, by omega⟩
