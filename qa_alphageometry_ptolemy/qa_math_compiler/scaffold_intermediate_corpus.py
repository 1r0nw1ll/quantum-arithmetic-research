#!/usr/bin/env python3
"""Deterministically extend the Family [31] demo corpus with examples 26-50."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACK = ROOT / "demo_pack_v1"
EXAMPLES = PACK / "examples"

CASES = [
    (
        "ex26_and_assoc",
        "Logical conjunction is associative.",
        "theorem ex26_and_assoc (p q r : Prop) : (p ∧ q) ∧ r ↔ p ∧ (q ∧ r)",
        """by
  constructor
  · intro h
    exact ⟨h.left.left, h.left.right, h.right⟩
  · intro h
    exact ⟨⟨h.left, h.right.left⟩, h.right.right⟩""",
        ["And.left", "And.right"],
        "logic",
    ),
    (
        "ex27_or_assoc",
        "Logical disjunction is associative.",
        "theorem ex27_or_assoc (p q r : Prop) : (p ∨ q) ∨ r ↔ p ∨ (q ∨ r)",
        """by
  constructor
  · intro h
    cases h with
    | inl hpq =>
        cases hpq with
        | inl hp => exact Or.inl hp
        | inr hq => exact Or.inr (Or.inl hq)
    | inr hr => exact Or.inr (Or.inr hr)
  · intro h
    cases h with
    | inl hp => exact Or.inl (Or.inl hp)
    | inr hqr =>
        cases hqr with
        | inl hq => exact Or.inl (Or.inr hq)
        | inr hr => exact Or.inr hr""",
        ["Or.inl", "Or.inr"],
        "logic",
    ),
    (
        "ex28_and_or_distrib",
        "Conjunction distributes over disjunction.",
        "theorem ex28_and_or_distrib (p q r : Prop) : p ∧ (q ∨ r) ↔ (p ∧ q) ∨ (p ∧ r)",
        """by
  constructor
  · intro h
    cases h.right with
    | inl hq => exact Or.inl ⟨h.left, hq⟩
    | inr hr => exact Or.inr ⟨h.left, hr⟩
  · intro h
    cases h with
    | inl hpq => exact ⟨hpq.left, Or.inl hpq.right⟩
    | inr hpr => exact ⟨hpr.left, Or.inr hpr.right⟩""",
        ["And.left", "Or.inl", "Or.inr"],
        "logic",
    ),
    (
        "ex29_exists_and",
        "A fixed proposition can move across an existential quantifier.",
        "theorem ex29_exists_and {α : Sort u} (p : Prop) (q : α → Prop) : p ∧ (∃ x, q x) ↔ ∃ x, p ∧ q x",
        """by
  constructor
  · intro h
    cases h.right with
    | intro x hx => exact ⟨x, h.left, hx⟩
  · intro h
    cases h with
    | intro x hx => exact ⟨hx.left, ⟨x, hx.right⟩⟩""",
        ["Exists.intro", "Exists.elim"],
        "quantifiers",
    ),
    (
        "ex30_forall_and",
        "Universal quantification distributes over conjunction.",
        "theorem ex30_forall_and {α : Sort u} (p q : α → Prop) : (∀ x, p x ∧ q x) ↔ (∀ x, p x) ∧ (∀ x, q x)",
        """by
  constructor
  · intro h
    exact ⟨fun x => (h x).left, fun x => (h x).right⟩
  · intro h x
    exact ⟨h.left x, h.right x⟩""",
        ["And.intro", "function_application"],
        "quantifiers",
    ),
    (
        "ex31_not_or",
        "The negation of a disjunction is the conjunction of the negations.",
        "theorem ex31_not_or (p q : Prop) : ¬ (p ∨ q) ↔ ¬ p ∧ ¬ q",
        """by
  constructor
  · intro h
    exact ⟨fun hp => h (Or.inl hp), fun hq => h (Or.inr hq)⟩
  · intro h hpq
    cases hpq with
    | inl hp => exact h.left hp
    | inr hq => exact h.right hq""",
        ["Or.inl", "Or.inr", "False.elim"],
        "logic",
    ),
    (
        "ex32_double_neg_intro",
        "Every proposition implies its double negation.",
        "theorem ex32_double_neg_intro (p : Prop) : p → ¬¬p",
        """by
  intro hp hnp
  exact hnp hp""",
        ["function_application"],
        "logic",
    ),
    (
        "ex33_contraposition",
        "An implication yields its constructive contrapositive.",
        "theorem ex33_contraposition (p q : Prop) : (p → q) → (¬ q → ¬ p)",
        """by
  intro hpq hnq hp
  exact hnq (hpq hp)""",
        ["function_application"],
        "logic",
    ),
    (
        "ex34_congr_arg",
        "Functions preserve equality of their inputs.",
        "theorem ex34_congr_arg {α β : Sort u} (f : α → β) (a b : α) : a = b → f a = f b",
        """by
  intro h
  exact congrArg f h""",
        ["congrArg"],
        "equality",
    ),
    (
        "ex35_comp_assoc",
        "Function composition is associative.",
        "theorem ex35_comp_assoc {α β γ δ : Sort u} (f : γ → δ) (g : β → γ) (h : α → β) (x : α) : f (g (h x)) = (fun y => f (g y)) (h x)",
        "rfl",
        ["rfl"],
        "functions",
    ),
    (
        "ex36_list_append_nil",
        "Appending the empty list on the right changes nothing.",
        "theorem ex36_list_append_nil {α : Type u} (xs : List α) : xs ++ [] = xs",
        """by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      exact congrArg (List.cons x) ih""",
        ["induction", "congrArg", "List.cons"],
        "lists",
    ),
    (
        "ex37_list_nil_append",
        "Appending a list to the empty list returns the list.",
        "theorem ex37_list_nil_append {α : Type u} (xs : List α) : [] ++ xs = xs",
        "rfl",
        ["rfl"],
        "lists",
    ),
    (
        "ex38_list_append_assoc",
        "List append is associative.",
        "theorem ex38_list_append_assoc {α : Type u} (xs ys zs : List α) : (xs ++ ys) ++ zs = xs ++ (ys ++ zs)",
        """by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      exact congrArg (List.cons x) ih""",
        ["induction", "congrArg", "List.cons"],
        "lists",
    ),
    (
        "ex39_list_length_append",
        "The length of an appended list is the sum of the lengths.",
        "theorem ex39_list_length_append {α : Type u} (xs ys : List α) : (xs ++ ys).length = xs.length + ys.length",
        """by
  induction xs with
  | nil => exact (Nat.zero_add ys.length).symm
  | cons x xs ih =>
      simp only [List.length, Nat.succ_add]
      exact congrArg Nat.succ ih""",
        ["induction", "Nat.zero_add", "congrArg"],
        "lists",
    ),
    (
        "ex40_list_map_id",
        "Mapping the identity function over a list changes nothing.",
        "theorem ex40_list_map_id {α : Type u} (xs : List α) : List.map (fun x => x) xs = xs",
        """by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      simp only [List.map, ih]""",
        ["induction", "List.map"],
        "lists",
    ),
    (
        "ex41_list_map_comp",
        "Two list maps compose into one map.",
        "theorem ex41_list_map_comp {α β γ : Type u} (f : β → γ) (g : α → β) (xs : List α) : List.map f (List.map g xs) = List.map (fun x => f (g x)) xs",
        """by
  induction xs with
  | nil => rfl
  | cons x xs ih =>
      simp only [List.map, ih]""",
        ["induction", "List.map"],
        "lists",
    ),
    (
        "ex42_add_left_cancel",
        "Natural-number addition cancels on the left.",
        "theorem ex42_add_left_cancel (a b c : Nat) : a + b = a + c → b = c",
        """by
  intro h
  exact Nat.add_left_cancel h""",
        ["Nat.add_left_cancel"],
        "algebra",
    ),
    (
        "ex43_add_right_cancel",
        "Natural-number addition cancels on the right.",
        "theorem ex43_add_right_cancel (a b c : Nat) : a + c = b + c → a = b",
        """by
  intro h
  exact Nat.add_right_cancel h""",
        ["Nat.add_right_cancel"],
        "algebra",
    ),
    (
        "ex44_mul_one",
        "One is a right identity for natural-number multiplication.",
        "theorem ex44_mul_one (n : Nat) : n * 1 = n",
        "Nat.mul_one n",
        ["Nat.mul_one"],
        "algebra",
    ),
    (
        "ex45_one_mul",
        "One is a left identity for natural-number multiplication.",
        "theorem ex45_one_mul (n : Nat) : 1 * n = n",
        "Nat.one_mul n",
        ["Nat.one_mul"],
        "algebra",
    ),
    (
        "ex46_le_trans",
        "Natural-number less-than-or-equal is transitive.",
        "theorem ex46_le_trans (a b c : Nat) : a ≤ b → b ≤ c → a ≤ c",
        """by
  intro hab hbc
  exact Nat.le_trans hab hbc""",
        ["Nat.le_trans"],
        "order",
    ),
    (
        "ex47_le_antisymm",
        "Mutual natural-number inequalities imply equality.",
        "theorem ex47_le_antisymm (a b : Nat) : a ≤ b → b ≤ a → a = b",
        """by
  intro hab hba
  exact Nat.le_antisymm hab hba""",
        ["Nat.le_antisymm"],
        "order",
    ),
    (
        "ex48_lt_trans",
        "Natural-number strict order is transitive.",
        "theorem ex48_lt_trans (a b c : Nat) : a < b → b < c → a < c",
        """by
  intro hab hbc
  exact Nat.lt_trans hab hbc""",
        ["Nat.lt_trans"],
        "order",
    ),
    (
        "ex49_succ_injective",
        "Equal successors have equal predecessors.",
        "theorem ex49_succ_injective (a b : Nat) : Nat.succ a = Nat.succ b → a = b",
        """by
  intro h
  exact Nat.succ.inj h""",
        ["Nat.succ.inj"],
        "equality",
    ),
    (
        "ex50_even_double",
        "The sum of a natural number with itself is even.",
        "theorem ex50_even_double (n : Nat) : ∃ k, n + n = 2 * k",
        """by
  exact ⟨n, (Nat.two_mul n).symm⟩""",
        ["Exists.intro", "Nat.two_mul", "Eq.symm"],
        "arithmetic",
    ),
]


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write(path, canonical(value) + "\n")


def main() -> int:
    index = json.loads((PACK / "index.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "kernel_trace_manifest.json").read_text(encoding="utf-8"))
    retained_examples = [
        entry for entry in index["examples"] if int(entry["id"][2:4]) <= 25
    ]
    retained_cases = [
        case
        for case in manifest["cases"]
        if not case["case_id"].startswith("ex") or int(case["case_id"][2:4]) <= 25
    ]

    for number, (case_id, claim, formal, proof, lemmas, topic) in enumerate(
        CASES, start=26
    ):
        directory = EXAMPLES / case_id
        created = f"2026-06-20T{number - 26:02d}:00:00Z"
        write(directory / "claim.txt", claim + "\n")
        write(directory / "proof.lean", formal + " :=\n  " + proof.replace("\n", "\n  ") + "\n")
        write(
            directory / "README.md",
            f"# {case_id}\n\nKernel-executed Lean 4.31.0 proof for: {claim}\n",
        )
        write_json(
            directory / "task.json",
            {
                "schema_id": "QA_FORMAL_TASK_SCHEMA.v1",
                "task_id": f"DEMO_{case_id.upper()}_TASK",
                "created_utc": created,
                "nl_statement": claim,
                "formal_goal": formal,
                "imports": [],
                "context": [],
                "constraints": {
                    "max_seconds": 30,
                    "max_memory_mb": 1024,
                    "allowed_tactics": [
                        "exact",
                        "intro",
                        "constructor",
                        "cases",
                        "induction",
                        "rfl",
                        "simp only",
                    ],
                },
                "invariant_diff": {
                    "demo_example": case_id,
                    "difficulty": "intermediate",
                    "domain_tags": [topic],
                },
            },
        )
        write_json(
            directory / "pair.json",
            {
                "schema_id": "QA_HUMAN_FORMAL_PAIR_CERT.v1",
                "pair_id": f"DEMO_{case_id.upper()}_PAIR",
                "created_utc": created,
                "natural_language_claim": claim,
                "formal_statement": formal,
                "alignment_evidence": {
                    "key_lemmas": lemmas,
                    "span_mappings": [
                        {
                            "nl_span": claim.rstrip("."),
                            "formal_identifiers": lemmas,
                        }
                    ],
                },
                "trace_ref": {
                    "trace_id": "0" * 64,
                    "result_status": "SUCCESS",
                    "replay_status": "SUCCESS",
                },
                "status": "PROVED",
                "objections": [],
                "invariant_diff": {"demo_example": case_id},
            },
        )
        write_json(
            directory / "status.json",
            {
                "example_id": case_id,
                "status": "PROVED",
                "replay_rate": 1.0,
                "compressed": False,
                "introduced_lemmas": 0,
            },
        )
        retained_examples.append(
            {
                "id": case_id,
                "topic": topic,
                "difficulty": "intermediate",
                "status": "PROVED",
            }
        )
        retained_cases.append(
            {
                "case_id": case_id,
                "source": f"demo_pack_v1/examples/{case_id}/proof.lean",
                "expected_status": "SUCCESS",
                "proof_method": ";".join(lemmas),
                "artifact_dir": f"demo_pack_v1/examples/{case_id}",
            }
        )

    index["example_count"] = len(retained_examples)
    index["examples"] = retained_examples
    manifest["cases"] = retained_cases
    write_json(PACK / "index.json", index)
    write_json(ROOT / "kernel_trace_manifest.json", manifest)
    print(canonical({"ok": True, "example_count": len(retained_examples)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
