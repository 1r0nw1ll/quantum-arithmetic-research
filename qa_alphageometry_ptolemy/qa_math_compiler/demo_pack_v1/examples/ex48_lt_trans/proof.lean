theorem ex48_lt_trans (a b c : Nat) : a < b → b < c → a < c :=
  by
    intro hab hbc
    exact Nat.lt_trans hab hbc
