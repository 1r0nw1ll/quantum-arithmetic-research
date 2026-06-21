import Mathlib.Data.Nat.Nth

theorem qa_mathlib07_nth_true (n : Nat) : Nat.nth (fun _ => True) n = n := by
  exact Nat.nth_true n
