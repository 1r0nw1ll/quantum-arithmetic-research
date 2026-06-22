import Mathlib.Data.ZMod.Basic

def qa_t_step9 (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_t_period_divides_24 : ∀ s : ZMod 9 × ZMod 9, (qa_t_step9^[24]) s = s := by
  native_decide
