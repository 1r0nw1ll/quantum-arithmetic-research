theorem ex46_le_trans (a b c : Nat) : a ≤ b → b ≤ c → a ≤ c :=
  by
    intro hab hbc
    exact Nat.le_trans hab hbc
