import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

def qa_step (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_satellite_period_exact :
    ∀ k : Fin 8, k.val ≠ 0 → (qa_step^[k.val]) (6, 3) ≠ (6, 3) := by native_decide
