import Mathlib.Data.Nat.Count

theorem qa_mathlib10_count_false (n : Nat) : Nat.count (fun _ => False) n = 0 := by
  exact Nat.count_false n
