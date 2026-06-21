import Mathlib.Data.Nat.BitIndices

theorem qa_mathlib04_bit_indices_even (n : Nat) : (2 * n).bitIndices = List.map (fun x => x + 1) n.bitIndices := by
  exact Nat.bitIndices_two_mul n
