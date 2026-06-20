import QAMinedLemmas

theorem compressed_ex33_contraposition (p q : Prop) :
    (p → q) → (¬ q → ¬ p) :=
  fun hpq hnq => qaCompose hnq hpq
