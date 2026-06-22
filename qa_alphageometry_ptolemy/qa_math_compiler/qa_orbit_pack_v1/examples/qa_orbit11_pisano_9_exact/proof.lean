import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

def qa_step (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_pisano_9_exact :
    ∀ k : Fin 24, k.val ≠ 0 →
      ∃ s : ZMod 9 × ZMod 9, (qa_step^[k.val]) s ≠ s := by native_decide
