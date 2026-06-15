<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Sun-Sun (1992) Acta Arithmetica 61, McIntosh-Roettger (2007) doi:10.1090/S0025-5718-07-01955-2 -->
# [420] QA Wall-Sun-Sun Depth Zero

**Cert family**: `qa_fibonacci_wss_cert_v1`
**Claim**: A prime p is a Wall-Sun-Sun prime iff δ(p) = 0. This is equivalent to the
classical condition p² | F_{p−(5/p)}, and the equivalence is exact for all odd primes p ≠ 5.
No Wall-Sun-Sun prime is known.

## The equivalence

The Lifting-the-Exponent (LTE) identity for Fibonacci sequences (Sun-Sun 1992):
```
v_p(F_{k·α}) = v_p(F_α) + v_p(k)    for p odd, p | F_α, p ∤ k
```

Set α = α(p) and m = p − (5/p), write m = r·α where r = m/α. The bound r < p holds by
case analysis (using α(p) | p−(5/p) from cert [416]):
- **Split** (5/p) = +1: r = (p−1)/α(p) ≤ p−1 < p
- **Inert** (5/p) = −1: r = (p+1)/α(p) ≤ (p+1)/2 < p  (since α(p) ≥ 2)

So p ∤ r, giving v_p(r) = 0, and therefore:
```
v_p(F_{p−(5/p)}) = v_p(F_{r·α}) = v_p(F_α) + 0 = v_p(F_α)
```
Hence **p² | F_{p−(5/p)} iff p² | F_{α(p)} iff δ(p) = 0**.

## The depth-invariant closure

Cert [419] characterizes δ(p) = 1: Fibonacci primes plus {41, 193, 1621, ...}.
Cert [420] characterizes δ(p) = 0: the WSS conjecture, open since 1960.
Together they close the two extreme depth layers; δ takes all values in {0, ..., p−1}.

## What WSS would imply

If p is WSS, then F_{α(p)} ≡ 0 (mod p²), meaning the QA T-orbit "vanishes to depth 2"
at its first return — the orbit closes not merely mod p but mod p². Wall (1960) showed
this is equivalent to asking whether the Fibonacci sequence mod p² has the same period
as mod p (i.e., whether Pisano(p²) = p · Pisano(p) fails).

## The LTE at k = p

The k = p case of LTE shows:
```
v_p(F_{p·α(p)}) = v_p(F_{α(p)}) + 1
```
So even if p is WSS (v_p(F_α) ≥ 2), the orbit at p·α has depth ≥ 3. The LTE telescopes:
δ lives in the first stratum; WSS means the first stratum is empty for that prime.

## Checks

- **C1**: LTE identity verified for 59 cases (5 k-values + k=p) across 10 test primes — **PASS**
- **C2**: δ(p) = 0 iff p² | F_{p−(5/p)} for all 93 odd primes ≠ 5 in [3,500] — **PASS**
- **C3**: No WSS prime among 41,536 odd primes ≠ 5 up to 500,000 — **PASS**
- **C4**: r = (p−(5/p))/α(p) < p for all 93 odd primes ≠ 5 in [3,500] — **PASS**

## Computational record

McIntosh-Roettger (2007) found no WSS prime below 9.7 × 10¹⁴. The heuristic
(δ(p) roughly uniform on {0,...,p−1}) predicts O(log log X) WSS primes up to X,
so ~3–4 are "expected" below the search bound — yet none has been found.

## Chain

| Cert | Claim |
|------|-------|
| [416] | α(p) \| p−(5/p) — rank of apparition divides the Frobenius eigenvalue gap |
| [417] | α(p²) = p·α(p); δ(p) = F_α/p mod p defined and non-zero a.e. |
| [418] | α(p) parity encodes (−1) quadratic residue via Cassini identity |
| [419] | δ(p) decomposes by parity; δ=1 census {2,3,5,13,41,89,193,233,1597,1621} |
| **[420]** | **δ(p) = 0 iff WSS; equivalence via LTE; no WSS prime ≤ 5×10⁵** |

**Open**: characterize non-Fibonacci δ=1 primes beyond maximum-rank condition ([419]).
**Open**: does any Wall-Sun-Sun prime exist? (This cert names the question; it does not answer it.)
