theorem ex42_add_left_cancel (a b c : Nat) : a + b = a + c → b = c :=
  by
    intro h
    exact Nat.add_left_cancel h
