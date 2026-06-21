import Mathlib.Data.Finset.Card

theorem qa_mathlib14_finset_card_inter_union {α : Type u} [DecidableEq α] (s t : Finset α) : (s ∩ t).card + (s ∪ t).card = s.card + t.card := by
  exact Finset.card_inter_add_card_union s t
