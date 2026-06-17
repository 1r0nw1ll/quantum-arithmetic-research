<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Wall (1960) doi:10.1080/00029890.1960.11989541 -->
# [438] QA Witt Tower Doubly-Ramified Period Law

**Cert family**: `qa_witt_tower_v2_period_law_cert_v1`
**Claim**: When `v_p(t-2) = 2` (p² divides t-2 exactly) in the det=+1
companion family, the orbit-count structure differs from both the
`v_p=1` case ([437]) and the `e=2` case ([435]): the fixed-point count
**jumps** from `p` at `k=1` to `p²` at all `k ≥ 2`, and the top two orbit
layers carry equal counts (joint birth).

## The gap

[437] explicitly excluded doubly-ramified cases (`v_p(t-2) ≥ 2`) when
checking C2 and C3.  The data there showed `count(1,k) = 9 = p²` for
`t=11, p=3` (where `t-2=9=3²`) — qualitatively different from [437]'s `p`.
This cert characterises that structure.

## The orbit-count law

For `v_p(t-2) = 2`, `M = [[t,-1],[1,0]]`, on `(Z/p^k Z)²`:

```
k=1:  {1: p,    p: p-1}                          (indistinguishable from v_p=1)

k≥2:  period set = {p^L : L = 0, ..., k}
      count(1)        = p²
      count(p^L)      = (p²-1)·p^L   for 1 ≤ L ≤ k-2   (frozen)
      count(p^(k-1))  = (p-1)·p^(k-1)                    (joint birth)
      count(p^k)      = (p-1)·p^(k-1)                    (birth)
```

The top two layers (`L=k-1` and `L=k`) carry **identical counts**.

## Comparison with v_p=1 ([437])

| Property | v_p=1 ([437]) | v_p=2 (this cert) |
|---|---|---|
| count(1) at k=1 | p | p |
| count(1) at k≥2 | p | **p²** |
| Frozen formula | (p²-1)·p^(L-1) | (p²-1)·p^L |
| Top-layer birth | single layer | **joint (L=k-1 = L=k)** |
| p=3 stall? | yes (c≡2 mod 3 at k=2) | **no** |

## Double-nilpotent mechanism

`N = M - I` satisfies `N² = (t-2)·M = p²·c₂·M` exactly (integer identity,
same algebraic fact as used throughout the chain).  When `p² | (t-2)`:

- **v_p=1**: `N² ≡ 0 mod p` only; `ker(N mod p²)` has `p` elements
- **v_p=2**: `N² ≡ 0 mod p²` (doubly nilpotent); `ker(N mod p²)` = p² elements

For `t = p²·c₂ + 2`, the condition `N·x ≡ 0 mod p²` reduces to just
`a ≡ b mod p²` — imposing no modular constraint on `a` itself — giving
`p²` fixed points at every tower level `k ≥ 2`.

## No p = 3 exception

Unlike [437]'s v_p=1 case (where `c ≡ 2 mod 3` caused a stall at `k=2`),
the v_p=2 formula holds for **all** primes `p ≥ 3` and **all** values of
`c₂ = (t-2)/p²`.  The doubled nilpotency removes the stall mechanism.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 V2_PERIOD_SET | p=5: period set={p^L:L=0..k}, c₂=1..4, k=1..4 | **PASS** |
| C2 V2_FIXED_JUMP | count(1)=p at k=1; p² for k≥2; p=3,5,7, all c₂, k=1..4 | **PASS** |
| C3 V2_ORBIT_FORMULA | Full formula, p=3,5,7, all c₂∈{1..p-1}, k=1..4 | **PASS** |
| C4 DOUBLE_NILPOTENT | N²≡0 mod p², ker=p², t=3..299 | **PASS** |
| C5 NO_P3_EXCEPTION | p=3 formula holds for c₂=1 and c₂=2 at k=1..5 | **PASS** |
| C6 K1_INVISIBLE | v_p=1 and v_p=2 distributions identical at k=1 | **PASS** |

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_v2_period_law_cert_v1
python3 qa_witt_tower_v2_period_law_cert_validate.py
```

Expected: `{"ok": true, ..., "fixture_summary": "7/7 passed"}`

## Lineage

- Fills gap explicitly excluded by **[437]** (which restricted C2/C3 to `v_p=1`)
- Shares nilpotent mechanism `N²=(t-2)M` with **[435]**/**[436]**/**[437]**;
  the v_p=2 specialisation `N²≡0 mod p²` is the new element
- Consistent with **[432]**/**[433]** — those certs are agnostic to `v_p`
- Natural next question: `v_p(t-2) = r` for general `r` — each new power of
  `p` in the ramification adds one level to the joint-birth structure

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi:10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — higher ramification groups, doubly-ramified extensions
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7 — Hensel lifting, nilpotent elements mod p^k
- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi:10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano-period structure for quadratic recurrences
