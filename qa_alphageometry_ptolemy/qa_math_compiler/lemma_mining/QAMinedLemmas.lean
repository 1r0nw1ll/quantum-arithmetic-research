def qaCompose {α β γ : Sort u} : (β → γ) → (α → β) → α → γ :=
  fun h₂ h₁ x => h₂ (h₁ x)

theorem qaListInduction {α : Type u} {P : List α → Prop}
    (base : P []) (step : ∀ x xs, P xs → P (x :: xs)) : ∀ xs, P xs :=
  List.rec base step
