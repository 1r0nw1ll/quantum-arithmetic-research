import Mathlib.Data.List.Basic

theorem qa_mathlib20_list_length_cons {α : Type u} (a : α) (l : List α) : (a :: l).length = l.length + 1 := by
  simp
