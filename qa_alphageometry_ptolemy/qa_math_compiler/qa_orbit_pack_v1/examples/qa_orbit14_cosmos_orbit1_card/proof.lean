import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

def qa_step' (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

def qa_cosmos_orbit1 : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 24).image (fun k => (qa_step'^[k]) (1, 0))

theorem qa_cosmos_orbit1_card : qa_cosmos_orbit1.card = 24 := by native_decide
