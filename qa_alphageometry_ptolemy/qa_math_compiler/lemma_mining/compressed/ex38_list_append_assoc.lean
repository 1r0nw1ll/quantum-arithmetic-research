import QAMinedLemmas

theorem compressed_ex38_list_append_assoc {α : Type u}
    (xs ys zs : List α) : (xs ++ ys) ++ zs = xs ++ (ys ++ zs) :=
  qaListInduction (P := fun xs => (xs ++ ys) ++ zs = xs ++ (ys ++ zs))
    rfl (fun x _xs ih => congrArg (List.cons x) ih) xs
