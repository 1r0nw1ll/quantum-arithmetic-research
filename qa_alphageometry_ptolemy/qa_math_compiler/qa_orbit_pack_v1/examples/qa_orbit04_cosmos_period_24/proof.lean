import Mathlib.Data.ZMod.Basic

def qa_t_step9 (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_cosmos_period_24 : (qa_t_step9^[24]) (1, 0) = (1, 0) := by decide
