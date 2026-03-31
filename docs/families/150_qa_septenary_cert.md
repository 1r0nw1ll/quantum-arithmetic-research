# Family [150] QA_SEPTENARY_CERT.v1

## One-line summary

The septenary numbers {1,2,4,5,7,8} = (Z/9Z)* form a 6-cycle under doubling mod 9, with complement {0,3,6} = QA singularity set.

## Mathematical content

### The septenary set

{1, 2, 4, 5, 7, 8} = the units of Z/9Z = elements coprime to 9. These are exactly the residues that appear in QA cosmos and satellite orbits.

### Doubling cycle

2 is a primitive root mod 9. Powers of 2 mod 9:

| k | 2^k mod 9 |
|---|-----------|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 7 (=16 mod 9) |
| 5 | 5 (=32 mod 9) |
| 6 | 1 (=64 mod 9) |

Period = 6 = phi(9) = Euler totient of 9.

### Complement = singularity

{0, 3, 6} = multiples of 3 mod 9 = non-units. In QA: the singularity fixed point (9,9) ≡ (0,0) mod 9 lives here. The {3,6,9} set is isolated from the doubling dynamic — 3 doubles to 6, 6 doubles to 3, 9 doubles to 9.

### Diagonal completion

Opposite pairs on the doubling circle sum to 9: (1,8), (2,7), (4,5). This is the Yin-Yang completion property: each element pairs with its 9-complement.

### Parity cross-over

- Even septenary {2, 4, 8}: sum = 14, digital root = 5 (odd)
- Odd septenary {1, 5, 7}: sum = 13, digital root = 4 (even)
- Cross-parity: even sums to odd, odd sums to even
- 4 + 5 = 9 (completion)

### QA orbit connection

- phi(9) = 6 = septenary cycle period
- pi(9) = 24 = Pisano period = cosmos orbit period
- 24/6 = 4 = ratio of full orbit to unit cycle
- Septenary = the "active" mod-9 residues; complement = the "fixed" ones

## Checks

| ID | Description |
|----|-------------|
| SP_1 | schema_version == 'QA_SEPTENARY_CERT.v1' |
| SP_GROUP | {1,2,4,5,7,8} = (Z/9Z)* = units mod 9 |
| SP_CYCLE | doubling mod 9 has period 6 and closes |
| SP_COMP | complement {0,3,6} = non-units = multiples of 3 |
| SP_DIAG | diagonal pairs sum to 9 |
| SP_PAR | even sum=14, odd sum=13 (parity cross-over) |
| SP_W | all 6 elements witnessed in cycle |
| SP_F | cycle starts from 1 |

## Source grounding

- **Ben Iverson**: QA mod-9 orbit structure; {3,6,9} = singularity; cosmos/satellite live in units
- **Grant/Ghannam, Philomath Ch 1**: doubling cycle {1,2,4,8,7,5}; Yin-Yang diagonal completion; {3,6,9} isolated
- **Number theory**: (Z/9Z)* = cyclic group of order phi(9)=6; 2 is primitive root mod 9

## Connection to other families

- **[128] Spread Period**: pi(9)=24 = cosmos period; phi(9)=6 = septenary period; 24/6=4
- **[130] Origin of 24**: 24 = fundamental area quantum; septenary period 6 divides 24
- **[147] Synchronous Harmonics**: coprime sync at product; septenary elements are coprime to 9
- **[148] Sixteen Identities**: C=4-par, G=5-par — both 4-par and 5-par residues are in the septenary set

## Fixture files

- `fixtures/sp_pass_group_structure.json` — full group proof: cycle, complement, diagonals, parity
- `fixtures/sp_pass_qa_connection.json` — QA orbit connection: singularity partition, Euler/Pisano ratio
