theorem ex50_even_double (n : Nat) : ∃ k, n + n = 2 * k :=
  by
    exact ⟨n, (Nat.two_mul n).symm⟩
