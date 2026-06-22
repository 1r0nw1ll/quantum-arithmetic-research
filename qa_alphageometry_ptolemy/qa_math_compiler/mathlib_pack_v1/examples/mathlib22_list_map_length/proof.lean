import Mathlib.Data.List.Basic

theorem qa_mathlib22_list_map_length {α β : Type u} (f : α → β) (l : List α) : (l.map f).length = l.length := by
  simp
