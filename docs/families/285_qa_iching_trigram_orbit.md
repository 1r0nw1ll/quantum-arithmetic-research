<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert documentation; primary sources cited in mapping_protocol_ref.json and validator -->

# [285] QA I Ching Trigram Orbit

**Cert family**: `qa_iching_trigram_orbit_cert_v1`
**Primary sources**:
- Iverson, B. (n.d.). "Eight Keynotes." Sympathetic Vibratory Physics articles. www.svpvril.com/svpweb39.html. Accessed 2026-05-30. — establishes 8 I Ching trigrams as the 8 QA keynote classes
- Wilhelm, R. (trans. Baynes, C.F.) (1950). *The I Ching or Book of Changes*. Princeton University Press. Bollingen Series XIX. ISBN 0-691-09750-X. — standard 3-bit binary encoding of trigrams
- Mechanism: cert [279] QA Orbit Access Theorem; `orbit_family on (code, 9)`

## Encoding

Each of the 8 I Ching trigrams is a stack of 3 lines (solid ─── = 1, broken ╌╌╌ = 0), read bottom-to-top with LSB=bottom (Fuxi / Earlier Heaven arrangement):

| Code | Name | Symbol | Element | Orbit Class |
|------|------|--------|---------|-------------|
| 0 | Kun  | ☷ | Earth    | **A1-excluded** |
| 1 | Zhen | ☳ | Thunder  | Cosmos-only |
| 2 | Kan  | ☵ | Water    | Cosmos-only |
| 3 | Dui  | ☱ | Lake     | **Satellite access** |
| 4 | Gen  | ☶ | Mountain | Cosmos-only |
| 5 | Li   | ☲ | Fire     | Cosmos-only |
| 6 | Xun  | ☴ | Wind     | **Satellite access** |
| 7 | Qian | ☰ | Heaven   | Cosmos-only |

## Claim

For the 8 I Ching trigram binary codes {0,...,7} under QA mod-9 orbit rules ([279]):

1. **Kun (code=0) is A1-excluded** — QA states are in {1,...,N}; the all-broken (pure yin) trigram has no valid QA state. Philosophically: Kun represents pure receptivity/emptiness — the unmoved substrate that cannot be a source of QA dynamic paths (cf. Keely Law 1: "atomoles in constant vibratory motion").

2. **Exactly Dui (3) and Xun (6) have Satellite access** — both are divisible by 3 but not by 9. These are the two trigrams with one unbroken-pair (011₂ and 110₂ respectively), the only codes in {1,...,7} not coprime to 3.

3. **No trigram has Singularity access** — this is a structural impossibility: max code = 7 < 9, so 9 cannot divide any trigram code. The 9-fold singularity is absent from the I Ching trigram layer; it appears only at the hexagram level (64 codes, some divisible by 9).

4. **Five remaining codes are Cosmos-only** — Zhen(1), Kan(2), Gen(4), Li(5), Qian(7) all coprime to 3.

## Structural Note

The result is independent of bit-ordering convention: swapping LSB/MSB exchanges Dui(3)↔Xun(6) and Zhen(1)↔Gen(4), Kan(2)↔Xun(6) — but the SET of Satellite-access codes {3,6} is preserved (the two codes remain divisible by 3 regardless of which trigrams map to them). The structural partition is encoding-convention-stable.

## Scope Boundaries

- Does **not** certify Hz frequency values (Iverson's published 55–163 Hz range are floats; S2 violation)
- Does **not** claim Iverson's Hz values are assigned to specific named trigrams (he did not publish that assignment)
- Does **not** claim the orbit partition has consequences for I Ching divination practice
- Does **not** extend to hexagrams (64 codes — a future cert candidate)
- Does **not** claim the QA Singularity appears at any trigram level

## Gates

- **KOA_1**: Kun (code=0) classified as `a1_excluded`
- **KOA_2**: Dui (code=3) and Xun (code=6) classified as `mul_3_not_9` (Satellite access)
- **KOA_3**: No code in {0,...,7} is `mul_9` (structural impossibility: 7 < 9)
- **KOA_4**: All 8 trigram codes exhaustively classified per EXPECTED_CLASS table
- **SRC**: `mapping_protocol_ref.json` present and well-formed
- **F**: every FAIL fixture declares `expected_fail_type` and fires

6 PASS fixtures, 4 FAIL fixtures. Validator: `qa_iching_trigram_orbit_cert_validate.py --self-test`.

## Lineage

Discovered 2026-05-30 during the Keely harmonics domain sweep. Prior OB note referenced keynote values (1580 Ken, 1050 Li, 2226 Chen) which could not be traced to any on-disk or publicly accessible source. Web search confirmed Iverson's svpweb39 article as the source for I Ching → QA keynotes connection, but Iverson's published Hz values (55–163 Hz range) are floats. The natural integer representation is the trigram binary codes (0–7), giving a clean exhaustive claim grounded in published sources.

## Verification Note (2026-07-05)

Independently re-verified both primary sources and the arithmetic.
**Wilhelm/Baynes (1950)** confirmed real — exact ISBN match
(0-691-09750-X), Bollingen Series XIX, Princeton University Press;
described by the New York Times Book Review as "still regarded as the
best and most authentic" translation. **Iverson's "Eight Keynotes"**
(svpweb39.html) confirmed real and accessible — fetched directly and
found it lists exactly the claimed 55.461–163.528 Hz range (8 values),
and — critically — confirms this cert's own honest self-correction:
the source text explicitly does **not** assign specific frequencies to
named trigrams ("the author does not associate specific I Ching
trigram names with particular frequencies"), exactly matching this
cert's "Lineage" note and "Scope Boundaries" disclaimer. The project's
own prior-session correction (dropping an untraceable frequency-to-
trigram mapping in favor of the verifiable binary-code claim) holds up
under independent re-checking.

**Trigram line-composition arithmetic independently reconfirmed**: each
trigram's actual line pattern (solid=1/broken=0, LSB=bottom) gives Kun
☷=000=0, Zhen ☳=001=1, Kan ☵=010=2, Dui ☱=011=3, Gen ☶=100=4, Li
☲=101=5, Xun ☴=110=6, Qian ☰=111=7 — matches the cert's table exactly.
3 and 6 are the only multiples of 3 in {1,...,7} (7<9, so no multiple of
9 is possible) — confirms KOA_2/KOA_3 trivially and correctly.

Validator (`_orbit_class`) already computes classification from
divisibility rules at runtime, not fixture-trusting. No bugs found;
this cert's scope-boundary discipline and lineage transparency are
already a strong example.
