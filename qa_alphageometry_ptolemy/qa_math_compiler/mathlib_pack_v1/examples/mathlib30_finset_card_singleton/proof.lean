import Mathlib.Data.Finset.Card

theorem qa_mathlib30_finset_card_singleton {α : Type u} (a : α) : ({a} : Finset α).card = 1 := by
  simp
