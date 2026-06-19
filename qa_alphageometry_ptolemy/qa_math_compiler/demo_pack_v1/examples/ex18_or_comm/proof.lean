theorem ex18_or_comm (p q : Prop) : p ∨ q ↔ q ∨ p :=
  by
    constructor <;> intro h
    · cases h with
      | inl hp => exact Or.inr hp
      | inr hq => exact Or.inl hq
    · cases h with
      | inl hq => exact Or.inr hq
      | inr hp => exact Or.inl hp
