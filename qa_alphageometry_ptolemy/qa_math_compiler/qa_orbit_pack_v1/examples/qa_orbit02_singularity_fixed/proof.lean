import Mathlib.Data.ZMod.Basic

def qa_t_step9 (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

theorem qa_singularity_fixed : qa_t_step9 (0, 0) = (0, 0) := by rfl
