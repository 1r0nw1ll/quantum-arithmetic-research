import Mathlib.Data.ZMod.Basic

def qa_t_step9 (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_satellite_period_8 : (qa_t_step9^[8]) (6, 3) = (6, 3) := by decide
