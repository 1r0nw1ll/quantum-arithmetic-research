theorem ex43_add_right_cancel (a b c : Nat) : a + c = b + c → a = b :=
  by
    intro h
    exact Nat.add_right_cancel h
