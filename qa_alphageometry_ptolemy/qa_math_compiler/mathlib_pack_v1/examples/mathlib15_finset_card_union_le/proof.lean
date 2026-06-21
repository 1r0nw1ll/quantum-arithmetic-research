import Mathlib.Data.Finset.Card

theorem qa_mathlib15_finset_card_union_le {α : Type u} [DecidableEq α] (s t : Finset α) : (s ∪ t).card ≤ s.card + t.card := by
  exact Finset.card_union_le s t
