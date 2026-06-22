import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.GroupTheory.SpecificGroups.Cyclic.Basic
import Mathlib.Tactic

def fib_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![1, 1; 1, 0]
private def fib_mat_inv_mat : Matrix (Fin 2) (Fin 2) (ZMod 9) := !![0, 1; 1, 8]

def fib_mat_unit : (Matrix (Fin 2) (Fin 2) (ZMod 9))ˣ :=
  ⟨fib_mat, fib_mat_inv_mat, by native_decide, by native_decide⟩

theorem fib_mat_zpowers_isCyclic : IsCyclic ↥(Subgroup.zpowers fib_mat_unit) :=
  Subgroup.isCyclic_zpowers fib_mat_unit
