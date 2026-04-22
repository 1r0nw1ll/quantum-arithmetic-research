# Augmentation v2 Consistency Check — Primitive D

**Checker:** fresh subagent, 2026-04-22.
**Scope:** axiom compliance + conflict + precision check for the new projective-subspace-lattice primitive in `P^{k-1}(Z)`.

## Axiom compliance

- **A1 (No-Zero).** The object is `P^{k-1}(Z) = (Z^k \ {0}) / ~`. The zero vector is explicitly excluded by construction. Under projective equivalence, a homogeneous coordinate vector `[x_1:...:x_k]` with some zero entries (e.g. `e_i = [0:...:1:...:0]`) is fine — A1 forbids only the all-zero representative, not zero entries in a nonzero vector. The text names this correctly: "exclude the zero vector." **Pass.**
- **A2 (Derived coords).** Dimension of span, dimension of meet, and dimension of join are all derived integers (`dim = rank − 1`). The Grassmann-rearranged formula `dim(V ∩ W) = dim V + dim W − dim(V + W)` is an integer-valued derivation. **Pass.**
- **T1 (Path-time).** Span, meet, and join are stated as single-step operations. No iteration. **Pass.**
- **T2/NT (Firewall).** Pure integer linear algebra; no continuous layer. No float; no sqrt. **Pass.**
- **S1 (No `**2`).** No squaring appears in the primitive. **Pass.**
- **S2 (No float state).** Rank is over `Z`. Integer Gaussian elimination / Smith-normal-form / Bareiss algorithm compute rank exactly — no QR, no SVD, no floats. Text says "rank over `Z`" explicitly. **Pass.**

## Conflict check

- **v1 `P²(Z)` subsumption.** v1 lists points as `[x:y:z] ∈ P²(Z)`. Primitive D at `k=3` gives `P²(Z)` with points as 0-dim subspaces and lines as 1-dim subspaces — the standard v1 content. v1's case is the `k=3` specialization; generalization subsumes it cleanly. Text notes this explicitly. **No conflict.**
- **Standard-basis acceptor placement vs A1.** `e_i = [0:...:1:...:0]` has zero entries but the vector itself is nonzero, so A1 is satisfied. A1 forbids the all-zero element, not sparse nonzero vectors. **No conflict.**
- **Grassmann over `Z` vs `Q`.** Grassmann's dimension formula `dim(V+W) + dim(V∩W) = dim V + dim W` holds over any **field**. Over `Z` directly, modules need not be free and dimension is not universally well-defined. However: for finitely generated free submodules of `Z^k` (which is the case here — integer-spans of finite point sets), the rank-over-`Z` of a matrix equals the rank-over-`Q` of the same matrix (the rank is the invariant factor / pivot count, identical in both settings). The projectivization `P^{k-1}(Z) = (Z^k \ {0}) / ~` has the same line-set as `P^{k-1}(Q)` up to scaling (every rational point has an integer representative by clearing denominators). **Grassmann holds over `Z` via rank-equals-rank-over-`Q`.** The augmentation text says "rank over `Z`" but does not spell out the `Z ↪ Q` passage that justifies Grassmann. Minor precision gap — recommend a single sentence.

## Precision check

- **Paxos pigeonhole arithmetic.** Claim: `dim(span(Q_1) ∩ span(Q_2)) = |Q_1 ∩ Q_2| − 1` under standard-basis placement in `P^{N-1}`. Verified below. The bound `dim ≥ q_1 + q_2 − N − 1` matches pigeonhole `|Q_1 ∩ Q_2| ≥ q_1 + q_2 − N` shifted by `-1` from projectivization. Text states this correctly. **Pass.**
- **General position.** For standard basis, any `k` of `N` basis vectors are linearly independent in `Z^N`, so any `q`-subset spans a `(q-1)`-dim projective subspace. "General position" is trivially satisfied — no extra definition needed. Text says so. **Pass.**
- **Span/meet/join syntax.** `span(S)`, `V ∩ W`, `V + W` are the standard operator names. A reproducer can write e.g. `span(Q_1) ∩ span(Q_2)` unambiguously. **Pass.**

## Arithmetic verification

Spot-checking `dim(span(Q_1) ∩ span(Q_2)) = |Q_1 ∩ Q_2| − 1` under standard-basis placement:

- **N=3, q=2:** `Q_1={1,2}`, `Q_2={1,3}`. `span(Q_1)={[a:b:0]}`, `span(Q_2)={[a:0:c]}`. Intersection = `{[a:0:0]} = {e_1}`, dim 0. `|Q_1∩Q_2|−1 = 1−1 = 0`. ✓
- **N=4, q=3:** `Q_1={1,2,3}`, `Q_2={2,3,4}`. `span(Q_1)={[a:b:c:0]}`, `span(Q_2)={[0:b:c:d]}`. Intersection = `{[0:b:c:0]}`, line (dim 1). `|Q_1∩Q_2|−1 = 2−1 = 1`. ✓
- **N=5, q=3:** `Q_1={1,2,3}`, `Q_2={3,4,5}`. `span(Q_1)={[a:b:c:0:0]}`, `span(Q_2)={[0:0:c:d:e]}`. Intersection = `{[0:0:c:0:0]} = {e_3}`, dim 0. `|Q_1∩Q_2|−1 = 1−1 = 0`. ✓
- **N=5, q=4, disjoint-except-one:** `Q_1={1,2,3,4}`, `Q_2={4,5,?}` — re-parameterize: `Q_1={1,2,3,4}`, `Q_2={2,4,5}` so `|Q_1∩Q_2|=2`. `span(Q_1)={[a:b:c:d:0]}`, `span(Q_2)={[0:b:0:d:e]}`. Intersection = `{[0:b:0:d:0]}`, dim 1. `|Q_1∩Q_2|−1 = 1`. ✓
- **Bound-saturation, N=5, q_1=q_2=3:** pigeonhole floor = `3+3−5 = 1`, so `|Q_1∩Q_2| ≥ 1` always; dim floor `= 3+3−5−1 = 0`. Take `Q_1={1,2,3}`, `Q_2={3,4,5}`: intersection = `{e_3}`, dim 0 = floor. ✓ (tight)

All checks pass.

## Overall verdict

**augmentation-with-fixes** — one minor precision gap. The primitive is axiom-clean, conflict-free, and arithmetically correct. A single-sentence clarification on `Z`-rank = `Q`-rank for integer submodules would close the Grassmann-over-ring question.

## Precision gaps

1. The statement "rank of the matrix of homogeneous coords, rank over `Z`" is correct but does not explain why Grassmann's field-level dimension formula applies to a `Z`-module setting. A reader rigorously minded may worry that `Z`-modules can have torsion / non-free submodules. (They cannot here, because submodules of free `Z`-modules of finite rank are free; but this deserves one line.)

## Recommended fixes

- Add one sentence after "rank over `Z`":

  > "Grassmann's formula `dim(V+W) + dim(V∩W) = dim V + dim W` is stated field-theoretically, but for integer-span submodules of `Z^k` the rank-over-`Z` equals the rank-over-`Q` of the same generator matrix (submodules of free `Z`-modules are free, and rank is preserved under `Z ↪ Q`). So the formula holds exactly in the integer setting used here."

- Optionally add one explicit arithmetic example (e.g. the N=4, q=3 case above) inline, to make the primitive self-demonstrating for a reproducer.

No other fixes required. Primitive D is usable as written for a Mode B reproducer, modulo the one-line clarification.
