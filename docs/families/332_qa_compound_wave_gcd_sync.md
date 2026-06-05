# [332] QA Compound Wave GCD Sync Count

**Family**: `qa_compound_wave_gcd_sync_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, pp.15-19  
"COMPOUND WAVES", "COMPLEX WAVES"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Coprime pairs: lcm=p×q; sync_in_lcm = sync_in_product = 1 | PASS |
| C2 | Non-coprime pairs: sync_in_product = gcd(p,q); sync_in_lcm = 1 always | PASS |
| C3 | Identity: lcm(p,q)×gcd(p,q) = p×q for all 105 pairs in {2..15}²; sync_in_product=gcd exhaustively | PASS |
| C4 | QA mod-24: gcd determines sync multiplicity in p×q cycle; lcm is waiting time between syncs | PASS |
| C5 | Pairwise-coprime triples: lcm(p,q,r)=p×q×r; exactly 1 triple-sync per lcm-cycle | PASS |

## Core Structural Result

For two waves of periods p and q:

| Cycle length | Sync count |
|-------------|-----------|
| lcm(p,q) | **always 1** |
| p×q | **= gcd(p,q)** |

The identity connecting both: **lcm(p,q) × gcd(p,q) = p × q**

This means:
- The *waiting time* between syncs is lcm(p,q) = p×q / gcd(p,q)
- Non-coprime waves sync gcd times more frequently than coprime waves at the same product-cycle
- In every lcm-cycle there is exactly one sync, regardless of gcd

## Coprime vs Non-Coprime (C1, C2)

| Pair (p,q) | gcd | lcm | sync_in_lcm | sync_in_p×q |
|-----------|-----|-----|------------|------------|
| (3,4) | 1 | 12 | 1 | 1 |
| (3,5) | 1 | 15 | 1 | 1 |
| (4,6) | 2 | 12 | 1 | 2 |
| (6,9) | 3 | 18 | 1 | 3 |
| (4,8) | 4 | 8 | 1 | 4 |
| (8,12) | 4 | 24 | 1 | 4 |

## QA Mod-24 Connection (C4)

The mod-24 QA system has natural period 24. For wave pairs in the 24-unit structure:

| Pair | gcd | lcm | Waiting time | Syncs per p×q cycle |
|------|-----|-----|-------------|-------------------|
| (3,8) | 1 | 24 | 24 | 1 |
| (4,6) | 2 | 12 | 12 | 2 |
| (3,6) | 3 | 6 | 6 | 3 |
| (4,12) | 4 | 12 | 12 | 4 |

gcd determines how many times waves "phase together" within a single p×q window.

## Pairwise-Coprime Triples (C5)

For triples (p,q,r) pairwise coprime: lcm(p,q,r) = p×q×r and there is exactly one triple-sync per full triple-cycle. This is the three-wave generalization: no pairwise gcd means no early sync, and the triple-product is minimal.

## Observer Projection Note (Theorem NT)

"Compound wave", "sync event", "waiting time" are observer-layer labels. The causal structure: integer gcd/lcm and divisibility counting. The structural identity lcm×gcd=product is a pure integer identity. No continuous wave functions enter.

**Depends on**: [331] Phasing In Odd-Wave, [327] Harmonic Cycle Platonic Inscription
