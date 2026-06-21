import Mathlib.Data.Nat.Count

theorem qa_mathlib09_count_true (n : Nat) : Nat.count (fun _ => True) n = n := by
  exact Nat.count_true n
