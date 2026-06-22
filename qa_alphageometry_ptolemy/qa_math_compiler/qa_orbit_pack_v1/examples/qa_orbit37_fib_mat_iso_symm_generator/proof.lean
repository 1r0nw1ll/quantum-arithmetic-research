import Mathlib.GroupTheory.SpecificGroups.Cyclic
import Mathlib.Tactic

-- (prerequisite defs in QAFibMatrixGroupIso.lean)
theorem fib_mat_iso_symm_generator :
    fib_mat_iso_ZMod24.symm fib_gen = Multiplicative.ofAdd 1 :=
  zmodMulEquivOfGenerator_symm_apply_generator fib_gen_generates fib_mat_zpowers_nat_card
