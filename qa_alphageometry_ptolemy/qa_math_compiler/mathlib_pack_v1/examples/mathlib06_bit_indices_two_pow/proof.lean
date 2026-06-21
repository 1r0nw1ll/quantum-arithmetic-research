import Mathlib.Data.Nat.BitIndices

theorem qa_mathlib06_bit_indices_two_pow (k : Nat) : (2 ^ k).bitIndices = [k] := by
  exact Nat.bitIndices_two_pow k
