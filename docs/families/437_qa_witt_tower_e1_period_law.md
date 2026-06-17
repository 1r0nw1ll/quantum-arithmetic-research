<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Wall (1960) doi:10.1080/00029890.1960.11989541 -->
# [437] QA Witt Tower Ramified Prime e=1 Period Law

**Cert family**: `qa_witt_tower_e1_period_law_cert_v1`
**Claim**: When `e = ord(lambda0 mod p) = 1` (i.e. `p | (t-2)` in the
det=+1 companion family), the period-set formula from [435] still holds
but the orbit-count formula changes structurally: the fixed-point count
rises from `e = 1` to `p`, and the frozen-layer multiplicity changes from
`(p+1)·p^(L-1)` to `(p²-1)·p^(L-1)`.

## The gap

[436] proved that `e` is rigid — it equals 1 whenever `p | (t-2)` in the
det=+1 family.  But [436] explicitly left the `e=1` case unanalysed,
noting only the analogy with [435]'s `p=2` stall.  [435]'s orbit-count
formula for general `e` predicts:

```
orbit count(1, k) = e = 1            ← wrong for e=1
orbit count(e·p^L, k>L) = (p+1)·p^(L-1)  ← wrong for e=1
```

When `e=1` these formulas fail in two independent ways.

## The correct law (for `v_p(t-2) = 1`, `p ≥ 5`)

```
Period set:          {p^L : L = 0, 1, ..., k}
orbit count(1, k)  = p                         for all k
orbit count(p^L, L = k) = (p-1)·p^(L-1)       (birth layer)
orbit count(p^L, L < k) = (p²-1)·p^(L-1)      (frozen layer)
```

The key difference: fixed-point orbits number `p` (not `1 = e`), and the
frozen-layer multiplier is `p+1` times `(p-1)·p^(L-1)` — i.e., `(p²-1)` —
rather than `(p+1)` alone.

## Nilpotent mechanism

`N = M - I` satisfies `N² = (t-2)·M` as an **exact integer identity** (no
approximation).  When `p | (t-2)`, `N² ≡ 0 mod p`, making `N` nilpotent
of degree 2 mod `p`.

Consequence: `ker(N mod p)` is a 1-dimensional subspace of `(Z/pZ)²` — it
has `p` elements.  Since the fixed points of `M` mod `p^k` equal
`ker(N mod p)` (shown by explicit Hensel lifting), the count is `p` at
**every** tower level `k`, invariant.

## Exception: p = 3 stall

For `p = 3` with `c = (t-2)/3 ≡ 2 mod 3`, a stall occurs at `k = 2`:
`M³ ≡ I mod 9` prematurely, so no period-9 orbits form at level `k = 2`.
The formula holds at `k = 1` even in the stall case, and holds at all `k`
when `c ≡ 1 mod 3`.  For `p ≥ 5` the formula holds universally.

This is directly analogous to [435]'s `p = 2` stall.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 E1_PERIOD_SET | p=5: period set = {p^L:L=0..k}, t=p+2..200 step p, k=1..3 | **PASS** |
| C2 E1_FIXED_COUNT | count(period=1) = p for p=3,5,7,11, mul=1..5, k=1..4 (v_p=1 only) | **PASS** |
| C3 E1_ORBIT_FORMULA | Full formula, p=5 k=1..3 and p=7 k=1..2 (v_p=1 only) | **PASS** |
| C4 NILPOTENT | N²=(t-2)M exact integer t=3..299; N²≡0 mod p when p\|(t-2) | **PASS** |
| C5 P3_STALL | Stall: no period-9 at k=2 for t=8,17,26,35; formula OK for t=5,14,23,32 | **PASS** |
| C6 LEGACY | t=7/p=5, t=5/p=3, t=12/p=5 reproduce formula | **PASS** |

## What this cert does NOT claim

- Does not analyse the doubly-ramified case `v_p(t-2) ≥ 2` — that
  requires separate treatment.
- Does not prove the formula for `p = 3` when `c ≡ 2 mod 3` at levels
  `k ≥ 2` — those are the stall cases documented in C5.
- Does not modify the period-set formula from [435] — the same period
  set `{1, e} ∪ {e·p^L}` specialised to `e=1` gives `{p^L:L=0..k}`.

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_e1_period_law_cert_v1
python3 qa_witt_tower_e1_period_law_cert_validate.py
```

Expected: `{"ok": true, ..., "fixture_summary": "7/7 passed"}`

## Lineage

- Fills the gap explicitly noted in **[436]** ("e=1 case is analogous to
  the p=2 stall in [435] and is not analysed here")
- Directly refines **[435]** by showing the orbit-count formula changes
  structurally at e=1, not merely parametrically
- Consistent with **[432]** (scaling isomorphism) and **[433]**
  (recursive refinement law); those certs are agnostic to the value of e
- Confirms **[434]**'s e=4 and **[435]**'s e=2 formulas are the generic
  case; e=1 is the nilpotent exception

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi:10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — nilpotent lifting, ramified p-adic extensions
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7 — Hensel lifting, primitive roots
- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi:10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano-period structure for quadratic recurrences
