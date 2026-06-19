theorem ex05_induction_add_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n _ =>
      rfl
