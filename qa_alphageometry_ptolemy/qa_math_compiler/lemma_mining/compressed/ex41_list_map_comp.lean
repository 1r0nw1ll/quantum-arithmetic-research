import QAMinedLemmas

theorem compressed_ex41_list_map_comp {α β γ : Type u}
    (f : β → γ) (g : α → β) (xs : List α) :
    List.map f (List.map g xs) = List.map (fun x => f (g x)) xs :=
  qaListInduction
    (P := fun xs =>
      List.map f (List.map g xs) = List.map (fun x => f (g x)) xs)
    rfl (fun x _xs ih => congrArg (List.cons (f (g x))) ih) xs
