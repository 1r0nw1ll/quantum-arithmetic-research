import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic

def qa_step (s : ZMod 9 × ZMod 9) : ZMod 9 × ZMod 9 := (s.1 + s.2, s.1)

def qa_satellite : Finset (ZMod 9 × ZMod 9) :=
  (Finset.range 8).image (fun k => (qa_step^[k]) (6, 3))

def qa_singularity : Finset (ZMod 9 × ZMod 9) := {(0, 0)}

theorem qa_satellite_singularity_disjoint : Disjoint qa_satellite qa_singularity := by native_decide
