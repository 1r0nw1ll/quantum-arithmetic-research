theorem ex19_imp_trans (p q r : Prop) : (p → q) → (q → r) → p → r :=
  by
    intro hpq hqr hp
    exact hqr (hpq hp)
