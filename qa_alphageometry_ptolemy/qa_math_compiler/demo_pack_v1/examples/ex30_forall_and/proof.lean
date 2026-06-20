theorem ex30_forall_and {α : Sort u} (p q : α → Prop) : (∀ x, p x ∧ q x) ↔ (∀ x, p x) ∧ (∀ x, q x) :=
  by
    constructor
    · intro h
      exact ⟨fun x => (h x).left, fun x => (h x).right⟩
    · intro h x
      exact ⟨h.left x, h.right x⟩
