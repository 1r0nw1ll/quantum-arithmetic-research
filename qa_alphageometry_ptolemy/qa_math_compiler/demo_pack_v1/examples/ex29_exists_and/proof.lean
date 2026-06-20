theorem ex29_exists_and {α : Sort u} (p : Prop) (q : α → Prop) : p ∧ (∃ x, q x) ↔ ∃ x, p ∧ q x :=
  by
    constructor
    · intro h
      cases h.right with
      | intro x hx => exact ⟨x, h.left, hx⟩
    · intro h
      cases h with
      | intro x hx => exact ⟨hx.left, ⟨x, hx.right⟩⟩
