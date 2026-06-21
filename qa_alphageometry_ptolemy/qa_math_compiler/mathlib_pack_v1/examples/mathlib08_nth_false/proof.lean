import Mathlib.Data.Nat.Nth

theorem qa_mathlib08_nth_false (n : Nat) : Nat.nth (fun _ => False) n = 0 := by
  exact Nat.nth_false n
