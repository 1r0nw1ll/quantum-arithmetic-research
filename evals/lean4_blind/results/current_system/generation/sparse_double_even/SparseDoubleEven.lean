theorem double_is_even (n : Nat) : ∃ k : Nat, n + n = 2 * k := by
  refine ⟨n, ?_⟩
  simp [two_mul, Nat.add_comm]
