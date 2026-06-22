import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

def qa_step' (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_cosmos_reps_distinct :
    ∀ k : ℕ, k < 24 → (qa_step'^[k]) (1, 0) ≠ (2, 0) ∧
                        (qa_step'^[k]) (1, 0) ≠ (4, 0) ∧
                        (qa_step'^[k]) (2, 0) ≠ (4, 0) := by native_decide
