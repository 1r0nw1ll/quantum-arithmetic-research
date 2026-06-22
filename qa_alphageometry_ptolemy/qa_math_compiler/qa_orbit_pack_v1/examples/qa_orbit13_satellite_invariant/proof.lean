import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

def qa_step' (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

def qa_satellite' : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 8).image (fun k => (qa_step'^[k]) (6, 3))

theorem qa_satellite_invariant :
    ∀ s ∈ qa_satellite', qa_step' s ∈ qa_satellite' := by native_decide
