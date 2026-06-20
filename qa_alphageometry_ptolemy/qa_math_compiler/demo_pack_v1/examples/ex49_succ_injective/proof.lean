theorem ex49_succ_injective (a b : Nat) : Nat.succ a = Nat.succ b → a = b :=
  by
    intro h
    exact Nat.succ.inj h
