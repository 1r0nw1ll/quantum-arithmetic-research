<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi.org/10.1080/00029890.1960.11989541, Dale (2026) Pythagorean Five Families paper -->
# [398] QA Five Families Complete Partition

**Cert family**: `qa_five_families_partition_cert_v1`
**Claim**: The five Fibonacci-family seeds partition all 81 digital-root pairs {1,...,9}² in sizes 24+24+24+8+1 = 81, pairwise disjoint, matching the paper's Table 1.

## Statement

This cert is the **computational proof of Theorem 2** (Complete Partition of Digital Root Pairs) in the *Pythagorean Five Families* paper.

The five generalized Fibonacci sequences, evolved under digital-root T-step T(b,e)=(e, dr(b+e)):

| Family | Seed | Period | Layer |
|---|---|---|---|
| Fibonacci | (1,1) | 24 | Cosmos |
| Lucas | (2,1) | 24 | Cosmos |
| Phibonacci | (3,1) | 24 | Cosmos |
| Tribonacci | (3,3) | 8 | Satellite |
| Ninbonacci | (9,9) | 1 | Singularity |

produce pairwise-disjoint orbits whose union is exactly all 81 = 9² digital-root pairs.

## The 9×9 Classification Table (Table 1)

Rows = dr(b) ∈ {1,...,9}, columns = dr(e) ∈ {1,...,9}:

```
dr(b)\dr(e)  1  2  3  4  5  6  7  8  9
dr(b)=1      F  F  L  P  F  P  L  F  F
dr(b)=2      L  L  F  L  P  P  L  F  L
dr(b)=3      P  P  T  L  F  T  F  L  T
dr(b)=4      F  P  F  P  P  L  L  P  P
dr(b)=5      P  L  L  P  P  F  P  F  P
dr(b)=6      L  F  T  F  L  T  P  P  T
dr(b)=7      F  L  P  P  L  F  L  L  L
dr(b)=8      F  L  P  F  P  L  F  F  F
dr(b)=9      F  L  T  P  P  T  L  F  N
```

`F=Fibonacci  L=Lucas  P=Phibonacci  T=Tribonacci  N=Ninbonacci`

## QA Orbit Shadow

The five families are the mod-9 shadow of the 72-8-1 orbit partition:
- **Cosmos (72 pairs)** = Fibonacci(24) + Lucas(24) + Phibonacci(24)
- **Satellite (8 pairs)** = Tribonacci(8)
- **Singularity (1 pair)** = Ninbonacci(1)

The Pisano-period connection (cert [281]): π(9) = 24 (Cosmos period) and π(3) = 8 (Satellite period).

## Checks

- **C1**: Orbit closure — each seed's T-orbit has exactly the stated period — PASS
- **C2**: No premature closure — minimum period = stated period — PASS
- **C3**: Pairwise disjoint — all 10 intersection checks clean — PASS
- **C4**: Complete coverage — union = all 81 pairs — PASS
- **C5**: Table match — all 81 cells match the paper's Table 1 exactly — PASS

## Chain Position

This cert directly supports the Five Families paper's main theorem. Extends:
- [281] (Pisano-Orbit Correspondence: period=8=π(3), 24=π(9))
- [212] (Fibonacci Hypergraph: orbit multiset (24,24,24,8,1))
- [211] (Cayley Bateson Filtration: Gamma_L1 components)
