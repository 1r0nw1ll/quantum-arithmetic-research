import Mathlib.Data.List.Basic

theorem qa_mathlib21_list_length_append {α : Type u} (l₁ l₂ : List α) : (l₁ ++ l₂).length = l₁.length + l₂.length := by
  simp
