import Mathlib.Data.Finset.Card

theorem qa_mathlib29_finset_card_range (n : ℕ) : (Finset.range n).card = n := by
  simp [Finset.card_range]
