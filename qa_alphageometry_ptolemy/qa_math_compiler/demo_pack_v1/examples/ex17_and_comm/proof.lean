theorem ex17_and_comm (p q : Prop) : p ∧ q ↔ q ∧ p :=
  by
    constructor
    · intro h
      exact ⟨h.right, h.left⟩
    · intro h
      exact ⟨h.right, h.left⟩
