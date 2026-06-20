import QAMinedLemmas

theorem compressed_ex40_list_map_id {α : Type u} (xs : List α) :
    List.map (fun x => x) xs = xs :=
  qaListInduction (P := fun xs => List.map (fun x => x) xs = xs)
    rfl (fun x _xs ih => congrArg (List.cons x) ih) xs
