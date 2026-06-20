theorem ex28_and_or_distrib (p q r : Prop) : p ∧ (q ∨ r) ↔ (p ∧ q) ∨ (p ∧ r) :=
  by
    constructor
    · intro h
      cases h.right with
      | inl hq => exact Or.inl ⟨h.left, hq⟩
      | inr hr => exact Or.inr ⟨h.left, hr⟩
    · intro h
      cases h with
      | inl hpq => exact ⟨hpq.left, Or.inl hpq.right⟩
      | inr hpr => exact ⟨hpr.left, Or.inr hpr.right⟩
