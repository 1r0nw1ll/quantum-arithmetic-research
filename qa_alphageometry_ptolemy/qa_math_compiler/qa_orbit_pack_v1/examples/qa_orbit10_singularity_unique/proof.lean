import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

def qa_step (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_singularity_unique :
    ∀ s : ZMod 9 × ZMod 9, qa_step s = s ↔ s = (0, 0) := by native_decide
