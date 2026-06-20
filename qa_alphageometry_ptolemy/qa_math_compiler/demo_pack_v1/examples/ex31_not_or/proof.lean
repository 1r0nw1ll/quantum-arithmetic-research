theorem ex31_not_or (p q : Prop) : ¬ (p ∨ q) ↔ ¬ p ∧ ¬ q :=
  by
    constructor
    · intro h
      exact ⟨fun hp => h (Or.inl hp), fun hq => h (Or.inr hq)⟩
    · intro h hpq
      cases hpq with
      | inl hp => exact h.left hp
      | inr hq => exact h.right hq
