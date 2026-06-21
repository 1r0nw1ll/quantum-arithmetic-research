import Mathlib.Data.List.Cycle

theorem qa_mathlib11_cycle_mem_reverse {α : Type u} {a : α} {s : Cycle α} : a ∈ s.reverse ↔ a ∈ s := by
  exact Cycle.mem_reverse_iff
