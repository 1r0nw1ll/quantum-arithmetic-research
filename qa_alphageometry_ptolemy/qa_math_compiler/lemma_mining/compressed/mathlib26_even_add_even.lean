import QAMathlibMinedLemmas

theorem compressed_mathlib26_even_add_even (n m : ℕ) (hn : Even n) (hm : Even m) : Even (n + m) := by
  exact qaEvenAdd hn hm
