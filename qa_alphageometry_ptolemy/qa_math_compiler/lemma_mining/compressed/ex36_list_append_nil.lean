import QAMinedLemmas

theorem compressed_ex36_list_append_nil {α : Type u} (xs : List α) :
    xs ++ [] = xs :=
  qaListInduction (P := fun xs => xs ++ [] = xs)
    rfl (fun x _xs ih => congrArg (List.cons x) ih) xs
