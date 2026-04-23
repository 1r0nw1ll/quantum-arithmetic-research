theorem add_zero_right (n : Nat) : n + 0 = n := by
  induction n with
  | zero =>
      simp
  | succ n ih =>
      simp [ih]

