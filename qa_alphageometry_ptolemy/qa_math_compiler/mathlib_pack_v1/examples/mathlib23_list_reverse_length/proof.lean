import Mathlib.Data.List.Basic

theorem qa_mathlib23_list_reverse_length {α : Type u} (l : List α) : l.reverse.length = l.length := by
  simp
