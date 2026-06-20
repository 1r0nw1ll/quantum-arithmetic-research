theorem ex33_contraposition (p q : Prop) : (p → q) → (¬ q → ¬ p) :=
  by
    intro hpq hnq hp
    exact hnq (hpq hp)
