import QAMinedLemmas

theorem compressed_ex19_imp_trans (p q r : Prop) :
    (p → q) → (q → r) → p → r :=
  fun hpq hqr => qaCompose hqr hpq
