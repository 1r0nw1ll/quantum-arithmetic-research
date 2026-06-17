<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Wall (1960) doi:10.1080/00029890.1960.11989541 -->
# [440] QA Witt Tower det=−1 General v_p Period Law

**Cert family**: `qa_witt_tower_dm1_general_vp_cert_v1`
**Claim**: Twin of [439] for the det=−1 companion matrix M=[[t,1],[1,0]]
(char poly x²−tx−1). For v_p(t²+4)=r (p≡1 mod 4), the orbit structure
is governed by the same Witt tower law as [439], with all non-trivial
orbit multiplicities divided by 4 and period-4 playing the role of period-1.

## The det=−1 companion matrix

For char poly x²−tx−1, the companion matrix is **M=[[t,1],[1,0]]** (det=−1).
The ramification condition is **p | (t²+4)** (discriminant t²+4 ≡ 0 mod p),
which requires **p ≡ 1 mod 4** (since −4 must be a quadratic residue).

The double eigenvalue mod p is λ₀ = t/2 mod p, satisfying λ₀²≡−1 (since
(t/2)²=(t²+4)/4 − 1 ≡ −1 when p|t²+4). So λ₀ has order exactly **4** — the
defining feature of this family.

## Structural constant: count(1) = 1

**Always, for all (p, t, k).** The only M-fixed point on (Z/p^k Z)² is zero.

*Proof*: M·(a,b)^T = (ta+b, a)^T = (a,b)^T requires a=b and ta+a=a, i.e.,
ta ≡ 0 mod p^k. Since p|t²+4 forces p∤t (otherwise p|4, impossible for odd p),
we get a≡0, b≡0.

This contrasts sharply with [439]'s det=+1 family where count(1)=p^min(r,k).

## Unified orbit-count formula

Let r=v_p(t²+4), c_r=(t²+4)/p^r with gcd(c_r,p)=1.

**Period set**: {1} ∪ {4} ∪ {4·p^L : L=1..k}

**Period-4 orbit count** (grows then saturates):
```
count(4) = (p^min(r,k) − 1) / 4
```

**Case k ≤ r** (within ramification depth):
```
count(4·p^L) = (p−1)/4 · p^(k−1)   for L=1..k   (k joint birth layers)
```

**Case k > r** (beyond ramification depth):
```
count(4·p^L) = (p²−1)/4 · p^(L+r−2)   for L=1..k−r   (frozen)
count(4·p^L) = (p−1)/4 · p^(k−1)       for L=k−r+1..k (r joint birth layers)
```

## Comparison with [439] (det=+1 family)

| Quantity | det=+1 ([439]) | det=−1 ([440]) |
|---|---|---|
| Base period | 1 | 4 |
| count(base) | p^min(r,k) | (p^min(r,k)−1)/4 |
| Frozen | (p²−1)·p^(L+r−2) | **(p²−1)/4**·p^(L+r−2) |
| Birth per layer | (p−1)·p^(k−1) | **(p−1)/4**·p^(k−1) |
| Structural const | none | count(1)=1 always |

All non-trivial orbit multiplicities are **exactly 1/4** of the det=+1 values.
The 4-fold dilution reflects the forced 4-cycle base imposed by λ₀²=−1.

## Algebraic mechanism

Define K = tI − 2M = [[−t,−2],[−2,t]]. Then:

```
det(K) = −t² − 4 = −(t²+4) = −p^r · c_r   (exact integer identity)
```

The period-4 equation M^4·x=x reduces (via M²=tM+I, M^4=(t³+2t)M+(t²+1)I)
to **K·x ≡ 0 mod p^k**. Therefore:

```
|ker(K mod p^k)| = p^min(r,k)
```

The p^min(r,k) elements of ker(K) include 1 fixed point (zero) and
p^min(r,k)−1 elements with period exactly 4, giving count(4)=(p^min(r,k)−1)/4.

This is the **exact structural twin** of [439]'s mechanism ker(N)=p^min(r,k)
with N=M−I; the only difference is K=tI−2M vs N=M−I.

## No p≡3 mod 4 exception — structural impossibility

For p≡3 mod 4: −4 is a non-residue mod p, so t²+4≡0 mod p has **no solution**.
The det=−1 family therefore has **no ramified primes** at p≡3 mod 4. This is
not a stall (like [437]'s p=3 exception) — it is structurally impossible.

The requirement p≡1 mod 4 is not a restriction on the theory but a necessary
condition for the eigenvalue equation λ²=−1 to have solutions mod p.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 DM1_PERIOD_SET | Period set={1,4,4p^L:L=1..k}; p=5, r=1,2, k=1..4 | **PASS** |
| C2 DM1_COUNT4_SATURATION | count(4)=(p^min(r,k)−1)/4; p=5,13, r=1..3, k=1..4 | **PASS** |
| C3 DM1_UNIFIED_FORMULA | Full formula; p=5 k=1..4, p=13 k=1..2, r=1..3 | **PASS** |
| C4 DM1_FIXED_ALWAYS_1 | count(1)=1 for all ramified (t,p,k) | **PASS** |
| C5 DM1_ALGEBRAIC_KER | ker(tI−2M mod p^k)=p^min(r,k); t=1..100, p=5,13, k=1..3 | **PASS** |
| C6 DM1_NO_RAMIFIED_P3MOD4 | p=7,11,19,23 admit no t with p|t²+4 | **PASS** |

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_dm1_general_vp_cert_v1
python3 qa_witt_tower_dm1_general_vp_cert_validate.py
```

Expected: `{"ok": true, ..., "fixture_summary": "7/7 passed"}`

## Lineage

- **[436]** proved e=4 is forced for ALL odd ramified primes in det=−1 family
- **[439]** established the det=+1 general v_p law (twin of this cert)
- **[437]**, **[438]** covered det=+1 for r=1, r=2 specifically
- This cert closes the det=−1 side: complete Witt tower orbit law for all r
- Together, [439]+[440] give the **complete orbit classification** for both
  companion-matrix families across all ramification depths

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi:10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — p-adic ramification, Witt towers
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.5,7 — quadratic residues (p≡1 mod 4 condition), Hensel lifting mod p^k
- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi:10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano-period structure for quadratic recurrences
