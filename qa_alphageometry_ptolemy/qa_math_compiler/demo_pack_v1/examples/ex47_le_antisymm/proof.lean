theorem ex47_le_antisymm (a b : Nat) : a ≤ b → b ≤ a → a = b :=
  by
    intro hab hba
    exact Nat.le_antisymm hab hba
