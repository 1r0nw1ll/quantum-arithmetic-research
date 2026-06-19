def sumFirst : Nat → Nat
  | 0 => 0
  | n + 1 => sumFirst n + (n + 1)

theorem ex03_sum_first_n (n : Nat) :
    2 * sumFirst n = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ n ih =>
      simp only [
        sumFirst,
        Nat.mul_add,
        Nat.add_mul,
        Nat.one_mul,
        Nat.mul_one,
        ih
      ]
      omega
