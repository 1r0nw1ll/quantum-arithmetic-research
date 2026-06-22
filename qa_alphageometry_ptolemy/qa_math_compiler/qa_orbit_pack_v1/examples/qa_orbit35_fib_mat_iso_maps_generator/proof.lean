import Mathlib.GroupTheory.SpecificGroups.Cyclic
import Mathlib.Tactic

-- (prerequisite defs in QAFibMatrixGroupIso.lean)
theorem fib_mat_iso_maps_generator :
    fib_mat_iso_ZMod24 (Multiplicative.ofAdd 1) = fib_gen :=
  zmodMulEquivOfGenerator_apply_ofAdd_one fib_gen_generates fib_mat_zpowers_nat_card
