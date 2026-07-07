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

**{3, 6, 9}** = multiples of 3 = non-units (A1 no-zero convention: standard Z/9Z residue notation would write this {0, 3, 6}, since (9,9) ≡ (0,0) mod 9, but this project's QA state alphabet is always {1,...,9} — corrected 2026-07-06, see Verification Note). This set is isolated from the doubling dynamic — 3 doubles to 6, 6 doubles to 3, 9 doubles to 9.

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
- `fixtures/sp_fail_bad_cycle.json` — Falsifier: doubling cycle with a wrong step (8→9 instead of 8→7) and a diagonal pair (2,6) that doesn't sum to 9 (added 2026-07-07)

## Sources

- Hardy, G.H. & Wright, E.M. (2008), *An Introduction to the Theory of Numbers*, 6th ed., Oxford University Press, ISBN 978-0-19-921986-5, Ch. VI — primitive roots, cyclic structure of (Z/nZ)*.
- Grant/Ghannam, *Philomath*, Ch. 1 — doubling cycle, Yin-Yang diagonal completion (publication year not independently confirmed in this pass).

## Verification Note (2026-07-06)

Independently recomputed every claim from scratch: `(Z/9Z)* = {1,2,4,5,7,8}`
(gcd check), doubling powers of 2 mod 9 give period exactly 6, diagonal
pairs (1,8)/(2,7)/(4,5) all sum to 9, even-subset sum=14/digital-root=5,
odd-subset sum=13/digital-root=4 — all confirmed exactly, no bugs in the
underlying arithmetic. The validator (`qa_septenary_cert_validate.py`)
was already genuinely recomputing everything live from the declared
`(b,e)`-style data (units via `gcd`, doubling steps, parity sums) rather
than fixture-trusting — no hardening needed there.

**Found and fixed a real A1-consistency bug**: the doc and both fixtures
declared the QA singularity complement as `{0, 3, 6}` (standard Z/9Z
residue notation, where 0 ≡ 9), but this project's hard axiom A1
requires the QA state alphabet to always be `{1,...,9}`, never
`{0,...,8}` — every other cert in this project (e.g. [181]'s singularity
fixed point `(M,M,M,M)`, never `(0,0)`) uses `9`, not `0`. The doc even
self-contradicted within one section, first writing "{0, 3, 6}" then
"{3,6,9} set is isolated" two sentences later without reconciling. More
seriously, `sp_pass_qa_connection.json`'s `qa_connections.singularity_set`
field explicitly declared `[0, 3, 6]` as *the QA-state-level* singularity
set — a genuine A1-adjacent inconsistency, since a downstream script that
trusted this field literally could introduce a zero-state into QA
arithmetic. Changed `COMPLEMENT` in the validator and `complement_set`/
`singularity_set` in both fixtures to `{3, 6, 9}` throughout, with an
explicit note on the 0≡9 mod-9 equivalence for anyone expecting standard
residue notation. `qa_axiom_linter.py` was already clean on this file
before and after (it doesn't statically flag pure math constants), but
the fixture-level data is now consistent with the rest of the project.
`--self-test` passes on both fixtures; verified the hardened `SP_COMP`
check still correctly rejects a reintroduced `[0,3,6]`.

**Follow-up (2026-07-07)**: this family had zero FAIL fixtures (part of
the 13-family zero-FAIL-fixture cluster). No `result=="FAIL"`
short-circuit exists (no print-corruption bug risk). Added
`fixtures/sp_fail_bad_cycle.json` with two independent planted defects
(a wrong doubling-cycle step 8→9 instead of 8→7; a diagonal pair (2,6)
that sums to 8 not 9) and wired it into `self_test()`; verified SP_CYCLE
and SP_DIAG both genuinely catch their respective defects.
