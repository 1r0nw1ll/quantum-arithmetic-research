# CLAIM — SCC decomposition of Caps(N,N) under full Σ, by parity of N

**Status:** candidate observation. Not promoted. Not verified for novelty.
Do not cite as a theorem until Lane B + Lane C both clear.

## Setting

Let `N ≥ 1`. Let `Caps(N, N) = { (b, e) ∈ ℤ × ℤ : 1 ≤ b ≤ N, 1 ≤ e ≤ N }`,
so `|Caps(N, N)| = N²`.

Let the four generators act on `Caps(N, N)`:

- `σ(b, e) = (b, e + 1)`      — legal iff `e ≤ N - 1`
- `μ(b, e) = (e, b)`          — always in-bounds on square `Caps`
- `λ₂(b, e) = (2b, 2e)`       — legal iff `b ≤ ⌊N/2⌋` and `e ≤ ⌊N/2⌋`
- `ν(b, e) = (b/2, e/2)`      — legal iff `b, e` both even

Let `Σ = {σ, μ, λ₂, ν}`. Let `G_Σ` be the directed transition graph on
`Caps(N, N)` whose edges are the legal applications of the generators in `Σ`.

## The claim

**(A) Even-N case.** For every even `N ≥ 2`:

```
#SCC(G_Σ) = 1
max |SCC| = N²
```

Equivalently, `G_Σ` is strongly connected.

**(B) Odd-N case.** For every odd `N ≥ 3`:

```
#SCC(G_Σ) = N + 1
max |SCC| = (N - 1)²
```

With structural decomposition:

1. One *inner* SCC of size `(N - 1)²`, consisting of the states in
   `Caps(N - 1, N - 1) ⊂ Caps(N, N)`.
2. `N - 1` *border* 2-cycles of the form `{(N, k), (k, N)}` for
   `k ∈ {1, 2, …, N - 1}`. Each is a μ-orbit of size 2.
3. One *singleton* SCC `{(N, N)}`.

Verifying: `1 + (N - 1) + 1 = N + 1` components; sizes `(N - 1)² + 2(N - 1) + 1
= N²`.

**(C) Boundary.** `N = 1`: `Caps(1, 1) = {(1, 1)}`, `#SCC = 1`, `max|SCC| = 1`.
`N = 2`: even case applies, `#SCC = 1`, `max|SCC| = 4`.

## Named interpretation (optional, from QA architecture)

Under the project's orbit taxonomy the (B) decomposition has three kinds of
component:

- inner `(N - 1)²` SCC → **Cosmos**-shaped
- `N - 1` border 2-cycles → **Satellite**-shaped (2-cycle orbits)
- `{(N, N)}` singleton → **Singularity**

If this interpretation holds, the three QA orbits are the components of `G_Σ`
on odd `N`. This is the structural interpretation — it is NOT part of the
quantitative claim above and should be verified separately.

## Monotonicity corollary (conditional)

`#SCC(G_Σ)` as a function of `N` has a discontinuity on parity: drops from
`N + 1` (odd `N`) to `1` (even `N + 1`). This is a claim about the sequence,
not about any single `N`.

## Domain assumptions

- 1-based indexing for `Caps` as in `paper1_qa_control.tex`. If the referent
  paper or spec uses 0-based, the counts shift and the claim must be
  re-expressed — do not compare across conventions without re-deriving.
- `λ₂` and `ν` are treated as distinct directed edges, not as an undirected
  pair.
- `μ` is an involution with fixed points on the diagonal `{(b, b)}`.
