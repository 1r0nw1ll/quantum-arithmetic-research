<!-- PRIMARY-SOURCE-EXEMPT: Iverson, B. (n.d.). Eight Keynotes. svpvril.com/svpweb39.html. Wilhelm, R. (trans. Baynes, C.F.) (1950). The I Ching or Book of Changes. Princeton University Press. ISBN 0-691-09750-X. -->

# [286] QA I Ching Hexagram Orbit

**Cert family**: `qa_iching_hexagram_orbit_cert_v1`
**Status**: PASS
**Depends on**: [279] QA Orbit Access Theorem, [285] QA I Ching Trigram Orbit

## Primary Sources

1. Iverson, B. (n.d.). "Eight Keynotes." www.svpvril.com/svpweb39.html — establishes I Ching trigram/hexagram system as a QA-compatible integer encoding.
2. Wilhelm, R. (trans. Baynes, C.F.) (1950). *The I Ching or Book of Changes*. Princeton University Press. ISBN 0-691-09750-X — standard Fuxi binary encoding of trigrams and hexagrams.

## Encoding

Each of the 64 I Ching hexagrams is assigned a natural integer code:

```
hexagram_code = lower_trigram_code + 8 × upper_trigram_code
```

where `lower_trigram_code` and `upper_trigram_code` are the 3-bit Fuxi codes from cert [285] (LSB = bottom line, solid = 1, broken = 0), each in {0, …, 7}.

| Code | Hex | Lower (bottom 3) | Upper (top 3) |
|------|-----|------------------|----------------|
| 0    | ䷁  | Kun=0 (☷)        | Kun=0 (☷)     |
| 9    | ䷲  | Zhen=1 (☳)       | Zhen=1 (☳)   |
| 18   | ䷜  | Kan=2 (☵)        | Kan=2 (☵)    |
| 27   | ䷹  | Dui=3 (☱)        | Dui=3 (☱)    |
| 36   | ䷳  | Gen=4 (☶)        | Gen=4 (☶)    |
| 45   | ䷝  | Li=5 (☲)         | Li=5 (☲)     |
| 54   | ䷸  | Xun=6 (☴)        | Xun=6 (☴)    |
| 63   | ䷀  | Qian=7 (☰)       | Qian=7 (☰)   |

## Algebraic Theorem

Since 8 ≡ −1 (mod 9):

```
hexagram_code mod 9 ≡ (lower − upper) mod 9
```

This gives the orbit partition directly from the trigram codes:

| Condition | Class | Count |
|-----------|-------|-------|
| code = 0 (both Kun) | A1-excluded | 1 |
| lower = upper (doubled trigram), code > 0 | Singularity (mul_9) | 7 |
| lower ≡ upper (mod 3), lower ≠ upper | Satellite (mul_3_not_9) | 14 |
| lower ≢ upper (mod 3) | Cosmos (coprime_to_3) | 42 |
| **Total** | | **64** |

The 7 Singularity-access hexagrams are exactly the **doubled-trigram** hexagrams (lower = upper): codes {9, 18, 27, 36, 45, 54, 63}.

## Claim

For every hexagram code in {0, …, 63}, `orbit_class(code)` computed by divisibility-by-3 and divisibility-by-9 is the correct QA orbit class. The algebraic identity `code mod 9 = (lower − upper) mod 9` is verified exhaustively for all 64 codes.

## Scope Boundaries

- Does **not** certify King Wen sequence positions (only natural binary codes).
- Does **not** claim divination interpretation or cosmological meaning.
- Does **not** cover hexagram codes outside {0, …, 63}.
- Encoding-convention stability: the claim is conditional on the Fuxi 3-bit encoding defined in cert [285]; alternative encoding conventions yield different integer codes and different orbit assignments.
