theorem ex35_comp_assoc {α β γ δ : Sort u} (f : γ → δ) (g : β → γ) (h : α → β) (x : α) : f (g (h x)) = (fun y => f (g y)) (h x) :=
  rfl
