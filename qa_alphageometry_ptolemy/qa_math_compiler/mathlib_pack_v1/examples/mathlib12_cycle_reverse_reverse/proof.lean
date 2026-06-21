import Mathlib.Data.List.Cycle

theorem qa_mathlib12_cycle_reverse_reverse {α : Type u} (s : Cycle α) : s.reverse.reverse = s := by
  exact Cycle.reverse_reverse s
