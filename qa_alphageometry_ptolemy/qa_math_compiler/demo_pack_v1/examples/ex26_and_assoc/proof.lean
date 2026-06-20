theorem ex26_and_assoc (p q r : Prop) : (p ∧ q) ∧ r ↔ p ∧ (q ∧ r) :=
  by
    constructor
    · intro h
      exact ⟨h.left.left, h.left.right, h.right⟩
    · intro h
      exact ⟨⟨h.left, h.right.left⟩, h.right.right⟩
