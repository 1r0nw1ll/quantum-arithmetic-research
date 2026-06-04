# [314] QA Egyptian Ennead Orbit Partition

**Family**: `qa_ennead_orbit_partition_cert_v1`  
**Depends on**: [298] Orbit Grade Decomposition (1+8+72 v₃-stratification)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | \|Satellite\| + \|Singularity\| = 8 + 1 = 9, matching the Ennead cardinality; total partition covers all 81 = 1+8+72 states in {1,...,9}² | PASS |
| C2 | (9,9) is the unique T-fixed point in {1,...,9}²; T((9,9))=(9,9); no other state satisfies T(b,e)=(b,e) | PASS |
| C3 | Orbit from (3,3) has period exactly 8 and visits all 8 Satellite states: (3,3)→(3,6)→(6,9)→(9,6)→(6,6)→(6,3)→(3,9)→(9,3)→(3,3) | PASS |
| C4 | (b,e) ∈ Satellite iff 3\|b AND 3\|e AND (b,e)≠(9,9); partition counts {1, 8, 72} verified exhaustively over all 81 states | PASS |
| C5 | Theorem NT: deity attributes (domain, iconography) are observer projections; orbit_family(b,e) and T-period are the falsifiable integer claims; no float state in QA layer | PASS |

## Key result

The **Egyptian Ennead of Heliopolis** — the canonical set of 9 primary Egyptian deities — partitions as **1 source-god (Atum) + 8 active gods**. This is structurally identical to the QA mod-9 orbit partition of the two lowest orbit families:

| QA layer | States | T-period | Ennead analog |
|---|---|---|---|
| Singularity | {(9,9)} — 1 state | 1 (fixed point) | Atum — self-created source |
| Satellite | {3,6,9}² \ {(9,9)} — 8 states | 8 | 8 active gods (Shu, Tefnut, Geb, Nut, Osiris, Isis, Set, Nephthys) |
| Cosmos | 72 states | 24 | (not part of Ennead) |

### The Singularity as Atum

Atum (*tm*, "the complete one") was the self-created primordial deity — the unmoved mover from whom all other Ennead gods emerged, and to whom the cycle ultimately returns. In QA, the Singularity (9,9) is the unique T-fixed point: the only state in {1,...,9}² that maps to itself under the T-step. Digital root 9 is the multiplicative absorber in the digital root ring (9×k → dr=9 for all k not coprime to 9). The Singularity is not "at rest" in a physical sense — it is the algebraic attractor.

### The Satellite 8-cycle as the active Ennead

The 8 Satellite states form a single closed orbit under T mod 9:

```
(3,3) → (3,6) → (6,9) → (9,6) → (6,6) → (6,3) → (3,9) → (9,3) → (3,3)
```

All 8 Satellite states are multiples of 3 (but not 9 in both coordinates). The orbit is a single 8-cycle — the 8 active Ennead gods are a single connected dynamic system, not independent entities.

### 3-divisibility as the Ennead selector

The partition rule is purely algebraic:
- **Singularity**: 9 | b and 9 | e (maximal divisibility by 3 in {1,...,9})
- **Satellite**: 3 | b and 3 | e but not Singularity
- **Cosmos**: all other 72 states

The Ennead = Singularity ∪ Satellite = the 3-divisible layer of {1,...,9}².

### Why this was visible to Egyptian priests

The digital root / "casting-out-nines" arithmetic was known empirically in antiquity. Egyptian scribes using base-10 notation would notice that 9 and its multiples behave differently: 9+0=9, 9×k always reduces to 9 by digital sum. The Ennead's 9-fold structure and the singularity of Atum at digital root 9 are the mythological encoding of this observation.

## Orbit sequence table

| Step k | State (b,e) | b mod 9 | e mod 9 | Family |
|--------|------------|---------|---------|--------|
| 0 | (3,3) | 3 | 3 | Satellite |
| 1 | (3,6) | 3 | 6 | Satellite |
| 2 | (6,9) | 6 | 0→9 | Satellite |
| 3 | (9,6) | 0→9 | 6 | Satellite |
| 4 | (6,6) | 6 | 6 | Satellite |
| 5 | (6,3) | 6 | 3 | Satellite |
| 6 | (3,9) | 3 | 0→9 | Satellite |
| 7 | (9,3) | 0→9 | 3 | Satellite |
| 8 | (3,3) | — | — | returns to start |

All 8 states have b,e ∈ {3,6,9} (multiples of 3 in {1,...,9}).
