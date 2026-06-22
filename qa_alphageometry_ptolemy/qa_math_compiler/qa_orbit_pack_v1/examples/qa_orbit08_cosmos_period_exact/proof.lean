import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

def qa_step (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_cosmos_period_exact :
    ∀ k : Fin 24, k.val ≠ 0 → (qa_step^[k.val]) (1, 0) ≠ (1, 0) := by native_decide
