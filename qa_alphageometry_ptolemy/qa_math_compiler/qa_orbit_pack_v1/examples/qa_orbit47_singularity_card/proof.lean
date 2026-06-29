import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

def qa_singularity : Finset (ZMod 9 × ZMod 9) := {(0, 0)}

theorem qa_singularity_card : qa_singularity.card = 1 := by decide
