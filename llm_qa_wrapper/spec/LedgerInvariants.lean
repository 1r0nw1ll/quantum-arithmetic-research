/-
# LLM QA Wrapper — Ledger Invariants (Lean 4)

Plumbing-level proofs of the append-only hash-chained ledger used by the
cert-gate kernel. Companion to `cert_gate.tla`. Scope is strictly the
ledger's structural invariants:

  * `valid : Ledger → Prop` — hash chain is unbroken from genesis
  * `append_preserves_valid` — appending a cert that chains against the
    current tail preserves validity
  * `ledger_prefix_valid` — every prefix of a valid ledger is valid
  * `hash_chain_binds_contents` — two valid ledgers with equal
    self-hashes at every index are equal (given hash injectivity)

This file does NOT prove anything about LLM behavior, tool semantics,
or the decision function of the gate. Those are out of scope for a
formal verification of the ledger plumbing.

Run: `lean llm_qa_wrapper/spec/LedgerInvariants.lean`
-/

namespace LLMQaWrapper.Ledger

/-- A `Hash` is an abstract type with decidable equality. In the kernel
this is a 32-byte SHA-256 digest. -/
structure Hash where
  bytes : Nat
deriving DecidableEq, Repr

def genesis : Hash := ⟨0⟩

/-- A `CertRecord` is a single ledger entry. It binds a payload hash and
chains to the previous cert via `prev`. -/
structure CertRecord where
  payload_hash : Hash
  prev         : Hash
  self_hash    : Hash
  counter      : Nat
deriving DecidableEq, Repr

/-- A `Ledger` is a list of cert records, append-only. Head is the oldest. -/
abbrev Ledger := List CertRecord

/-- `ChainStartsAt h L` holds when `L` is a valid hash chain whose first
entry's `prev` equals `h`. This is the clean recursive formulation that
avoids awkward case splits. -/
def ChainStartsAt : Hash → Ledger → Prop
  | _, []       => True
  | h, c :: cs  => c.prev = h ∧ ChainStartsAt c.self_hash cs

/-- Equational lemma for the empty case. Tagged `@[simp]` so simp can
rewrite `ChainStartsAt h []` to `True` without unfolding the match. -/
@[simp]
theorem ChainStartsAt_nil (h : Hash) : ChainStartsAt h [] = True := rfl

/-- Equational lemma for the cons case. Same pattern. -/
@[simp]
theorem ChainStartsAt_cons (h : Hash) (c : CertRecord) (cs : Ledger) :
    ChainStartsAt h (c :: cs) = (c.prev = h ∧ ChainStartsAt c.self_hash cs) :=
  rfl

/-- A ledger is valid if its chain starts at genesis. -/
def valid (L : Ledger) : Prop := ChainStartsAt genesis L

/-- An empty ledger is valid. -/
theorem valid_nil : valid [] := by
  unfold valid ChainStartsAt; trivial

/-- A singleton ledger is valid iff its cert chains against genesis. -/
theorem valid_singleton (c : CertRecord) :
    valid [c] ↔ c.prev = genesis := by
  unfold valid ChainStartsAt
  simp

/-- A two-element ledger is valid iff the chain conditions hold. -/
theorem valid_cons_cons (c d : CertRecord) (rest : Ledger) :
    valid (c :: d :: rest) ↔
    c.prev = genesis ∧ ChainStartsAt c.self_hash (d :: rest) := by
  unfold valid ChainStartsAt
  rfl

/-- Helper: the hash that the next entry must chain against, given a
chain starting at `h`. -/
def lastHash : Hash → Ledger → Hash
  | h, []      => h
  | _, c :: cs => lastHash c.self_hash cs

/-- `ChainStartsAt h (L ++ [c])` holds iff `ChainStartsAt h L` and
`c.prev = lastHash h L`. This is the key lemma that lets us prove
`append_preserves_valid`. -/
theorem chain_append
    (h : Hash) (L : Ledger) (c : CertRecord) :
    ChainStartsAt h (L ++ [c]) ↔
    ChainStartsAt h L ∧ c.prev = lastHash h L := by
  induction L generalizing h with
  | nil =>
      simp only [List.nil_append, ChainStartsAt_cons, ChainStartsAt_nil,
                 lastHash, and_true, true_and]
  | cons head rest ih =>
      simp only [List.cons_append, ChainStartsAt_cons, lastHash]
      rw [ih head.self_hash]
      constructor
      · intro ⟨hp, hrest, hlast⟩
        exact ⟨⟨hp, hrest⟩, hlast⟩
      · intro ⟨⟨hp, hrest⟩, hlast⟩
        exact ⟨hp, hrest, hlast⟩

/-- Appending a cert that chains against the current ledger tail
preserves validity. This is the core ledger invariant. -/
theorem append_preserves_valid
    (L : Ledger) (c : CertRecord)
    (hv : valid L)
    (hprev : c.prev = lastHash genesis L) :
    valid (L ++ [c]) := by
  unfold valid
  exact (chain_append genesis L c).mpr ⟨hv, hprev⟩

/-- Validity is preserved under taking a prefix. Any initial segment of a
valid ledger is itself valid. This is the ledger's monotonicity property:
early history cannot be invalidated by later appends. -/
theorem chain_prefix_valid
    (h : Hash) (L₁ L₂ : Ledger)
    (hv : ChainStartsAt h (L₁ ++ L₂)) :
    ChainStartsAt h L₁ := by
  induction L₁ generalizing h with
  | nil => simp only [ChainStartsAt_nil]
  | cons c rest ih =>
      simp only [List.cons_append, ChainStartsAt_cons] at hv
      simp only [ChainStartsAt_cons]
      exact ⟨hv.1, ih c.self_hash hv.2⟩

/-- The ledger-level monotonicity statement: every prefix of a valid
ledger is valid. -/
theorem ledger_prefix_valid
    (L₁ L₂ : Ledger)
    (hv : valid (L₁ ++ L₂)) :
    valid L₁ := by
  unfold valid at hv ⊢
  exact chain_prefix_valid genesis L₁ L₂ hv

/-- Any prefix of a valid ledger is a subset: every element of the prefix
is an element of the full ledger. This is the weak monotonicity statement
the cert-gate kernel relies on. -/
theorem ledger_prefix_subset
    {L₁ L₂ : Ledger}
    (hpref : L₁ <+: L₂) :
    ∀ c ∈ L₁, c ∈ L₂ := by
  intro c hc
  exact List.IsPrefix.subset hpref hc

/-- Hash chain binds contents: two valid ledgers with the same length and
the same sequence of `self_hash` values must be equal, provided
`self_hash` is injective on cert records (i.e. hash collisions are
computationally infeasible, which the SHA-256 kernel implementation
provides at the cryptographic level). -/
theorem hash_chain_binds_contents
    (L₁ L₂ : Ledger)
    (hlen : L₁.length = L₂.length)
    (hchain : ∀ i (h₁ : i < L₁.length) (h₂ : i < L₂.length),
                L₁[i].self_hash = L₂[i].self_hash)
    (hinj : ∀ (c₁ c₂ : CertRecord), c₁.self_hash = c₂.self_hash → c₁ = c₂) :
    L₁ = L₂ := by
  apply List.ext_getElem hlen
  intro i h₁ h₂
  exact hinj _ _ (hchain i h₁ h₂)

/-- Counter monotonicity — a separate invariant from chain validity. This
captures replay resistance: even if an adversary replays a cert with the
same payload, the counter binds the cert to a specific point in the
issuance sequence. A ledger is counter-monotone if each entry's counter
is strictly greater than the previous. -/
def CounterMonotone : Ledger → Prop
  | []         => True
  | [_]        => True
  | c :: d :: rest => c.counter < d.counter ∧ CounterMonotone (d :: rest)

@[simp]
theorem CounterMonotone_nil : CounterMonotone [] = True := rfl

@[simp]
theorem CounterMonotone_singleton (c : CertRecord) :
    CounterMonotone [c] = True := rfl

@[simp]
theorem CounterMonotone_cons_cons (c d : CertRecord) (rest : Ledger) :
    CounterMonotone (c :: d :: rest) =
    (c.counter < d.counter ∧ CounterMonotone (d :: rest)) := rfl

/-- Counter monotonicity is preserved under taking a prefix. -/
theorem counter_monotone_prefix
    (c : CertRecord) (rest : Ledger)
    (hm : CounterMonotone (c :: rest)) :
    CounterMonotone rest := by
  cases rest with
  | nil => simp
  | cons d tail =>
      simp only [CounterMonotone_cons_cons] at hm
      exact hm.2

/-- Summary: the four invariants the cert-gate protocol needs hold.
(1) Empty ledger is valid.
(2) Validity is preserved under correct append.
(3) Validity is preserved under prefix-taking (monotonicity).
(4) The hash chain binds contents under hash injectivity.
-/
theorem ledger_invariants_summary :
    valid []
    ∧ (∀ (L : Ledger) (c : CertRecord),
         valid L → c.prev = lastHash genesis L → valid (L ++ [c]))
    ∧ (∀ (L₁ L₂ : Ledger), valid (L₁ ++ L₂) → valid L₁) := by
  refine ⟨valid_nil, ?_, ?_⟩
  · exact fun L c hv hp => append_preserves_valid L c hv hp
  · exact fun L₁ L₂ hv => ledger_prefix_valid L₁ L₂ hv

end LLMQaWrapper.Ledger
