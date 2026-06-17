<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Wall (1960) doi:10.1080/00029890.1960.11989541 -->
# [439] QA Witt Tower General v_p Period Law

**Cert family**: `qa_witt_tower_general_vp_period_law_cert_v1`
**Claim**: For ANY v_p(t-2) = r ≥ 1, the orbit-count distribution of the
det=+1 companion matrix M=[[t,−1],[1,0]] on (Z/p^k Z)² follows a unified
law parameterised by r, p, and k.

## Unified orbit-count formula

Let r = v_p(t−2), c_r = (t−2)/p^r with gcd(c_r, p)=1.

**Case k ≤ r** (tower depth within ramification depth):
```
count(1)     = p^k              (grows with k, not yet saturated)
count(p^L)   = (p−1)·p^(k−1)   for L=1..k   (all layers joint birth)
```

**Case k > r** (tower depth exceeds ramification depth):
```
count(1)     = p^r              (SATURATION — caps at p^r)
count(p^L)   = (p²−1)·p^(L+r−2)   for L=1..k−r   (frozen)
count(p^L)   = (p−1)·p^(k−1)       for L=k−r+1..k  (joint birth, r layers)
```

## Algebraic mechanism

`N = M − I` satisfies `N² = (t−2)·M = p^r·c_r·M` exactly (integer identity,
holds for all r). The fixed-point condition `N·x ≡ 0 mod p^k` reduces to:

```
a = b  (mod p^k)   and   c_r·a ≡ 0  (mod p^(k−r))
```

Since gcd(c_r, p)=1, this forces `a ≡ 0 mod p^max(k−r,0)`. Therefore:

```
ker(N mod p^k) = {(a, a) : a ∈ p^max(k−r,0)·Z/p^k Z} = p^min(r,k) elements
```

Each unit of ramification r contributes exactly one free lifting step. Once
k exceeds r, further lifting is constrained — the fixed-point count saturates
at p^r.

## Unification of the Witt tower chain

| Cert | Case | Recovered by |
|------|------|--------------|
| [437] | r=1 (simple ramification) | Set r=1: frozen=(p²−1)·p^(L−1), 1 birth layer |
| [438] | r=2 (doubly-ramified) | Set r=2: frozen=(p²−1)·p^L, 2 joint birth layers |
| **[439]** | **r≥1 (general)** | **This cert — complete unification** |

The three formulas are NOT three separate laws — they are one law at r=1, r=2,
and general r. The frozen-layer exponent shift (p^(L−1) → p^L → p^(L+1) → ···)
and the growing number of joint-birth layers (1 → 2 → 3 → ···) are both
explained by the single kernel formula `ker = p^min(r,k)`.

## The saturation transition

At k=r the count(1) reaches its maximum p^r and stays there for all k>r.
This is observable as a "plateau" in the fixed-point growth curve:

```
k:          1    2    3    4    5    6    ...
count(1):   p    p²   p³   p³   p³   p³  ...   (for r=3)
```

The plateau begins exactly at k=r. No analogous saturation exists for the
non-fixed-point orbits — they continue growing with k.

## Exception: p=3 stall at r=1

For p=3, r=1, c=(t−2)/p ≡ 2 mod 3 (i.e. c ≡ p−1 mod p):
the orbit distribution at k=2 differs from the formula — period-p² orbits
never appear, with all non-fixed-point orbits retaining period p. This is the
p=3 stall documented in [437]. The general formula applies to all other
(p, r, c_r) combinations.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 GEN_PERIOD_SET | Period set={p^L:L=0..k} for r=3, p=3,5, k=1..5 | **PASS** |
| C2 GEN_FIXED_SATURATION | count(1)=p^min(r,k); p=3,5,7, r=1..4, c_r=1..2, k=1..5 | **PASS** |
| C3 GEN_UNIFIED_FORMULA | Full orbit law, p=3/5/7, r=1..3, k=1..5 (excl. p=3 stall) | **PASS** |
| C4 ALGEBRAIC_KER | ker(N mod p^k)=p^min(r,k) for t=3..200, p=3,5,7 | **PASS** |
| C5 RECOVERS_437 | r=1 formula = [437] closed form exactly | **PASS** |
| C6 RECOVERS_438 | r=2 formula = [438] closed form exactly | **PASS** |

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_general_vp_period_law_cert_v1
python3 qa_witt_tower_general_vp_period_law_cert_validate.py
```

Expected: `{"ok": true, ..., "fixture_summary": "7/7 passed"}`

## Lineage

- Closes the gap left by **[437]** (r=1) and **[438]** (r=2) by proving the general law
- Uses the same nilpotent identity `N²=(t−2)M` as **[435]**/**[436]**/**[437]**/**[438]**
- Mechanism `ker(N mod p^k) = p^min(r,k)` is new: it unifies both the saturation
  and the joint-birth structure in a single algebraic statement
- **[432]**/**[433]** (scaling, recursion) are agnostic to v_p and remain upstream

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi:10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — higher ramification groups, Witt vectors, p-adic lifting
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7 — Hensel lifting, nilpotent elements mod p^k
- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi:10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano-period structure for quadratic recurrences
