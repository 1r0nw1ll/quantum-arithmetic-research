import Mathlib.Data.Finset.Card

theorem qa_mathlib13_finset_card_union_inter {α : Type u} [DecidableEq α] (s t : Finset α) : (s ∪ t).card + (s ∩ t).card = s.card + t.card := by
  exact Finset.card_union_add_card_inter s t
