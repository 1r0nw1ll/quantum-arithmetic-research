theorem ex27_or_assoc (p q r : Prop) : (p ∨ q) ∨ r ↔ p ∨ (q ∨ r) :=
  by
    constructor
    · intro h
      cases h with
      | inl hpq =>
          cases hpq with
          | inl hp => exact Or.inl hp
          | inr hq => exact Or.inr (Or.inl hq)
      | inr hr => exact Or.inr (Or.inr hr)
    · intro h
      cases h with
      | inl hp => exact Or.inl (Or.inl hp)
      | inr hqr =>
          cases hqr with
          | inl hq => exact Or.inl (Or.inr hq)
          | inr hr => exact Or.inr hr
