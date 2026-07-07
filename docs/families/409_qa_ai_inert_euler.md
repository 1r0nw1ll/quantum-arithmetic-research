<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Arthur-Clozel (1989) ISBN 978-0-691-08517-3, Shimura (1971) ISBN 978-0-691-08092-5 -->
# [409] QA Langlands AI Inert Prime Euler Factor — GL₄/ℚ at p ≡ 2,3 mod 5

**Cert family**: `qa_ai_inert_euler_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a and prime p ≡ 2 or 3 mod 5 (p inert in ℚ(√5)/ℚ, p ≠ 5):

```
P_p^{inert}(Y) = 1 + p²·Y⁴
```

All middle coefficients a₁=a₂=a₃=0.

## Three-Step Derivation

**Step 1 — CM vanishing theorem.**
For p ≡ 2,3 mod 5: ord_{(ℤ/5ℤ)^×}(p) = 4. This means p generates the full group
Gal(ℚ(ζ₅)/ℚ(√5))^c — equivalently, 𝔭=(p) is inert in K=ℚ(ζ₅) over F=ℚ(√5).
By CM theory: a_𝔭 = 0 for f at any prime 𝔭 of F inert in K/F.

**Step 2 — GL₂/F local factor at inert 𝔭.**
With a_𝔭=0 and N(𝔭)=p², the local Euler polynomial at 𝔭 is:
```
Q_𝔭(Z) = 1 − a_𝔭·Z + N(𝔭)·Z² = 1 + p²·Z²     (Z = N(𝔭)^{−s} = p^{−2s})
```
The roots ±i·p^{-1} have magnitude p^{-1}.

**Step 3 — Automorphic induction at inert p.**
For p inert in F/ℚ, the AI-induction formula gives:
```
L_p(Y, AI(f)) = L_𝔭(Y², f)     (Y = p^{−s}, Y² = p^{−2s} = Z)
```
Substituting: `Q_𝔭(Y²) = 1 + p²·Y⁴`. 

## Complete Prime Classification for AI(f)

| Prime type | Condition | Euler polynomial | Cert |
|---|---|---|---|
| Split | p ≡ 1,4 mod 5 | 1−TY+(N+2p)Y²−pTY³+p²Y⁴ | [404] |
| **Inert** | **p ≡ 2,3 mod 5** | **1 + p²Y⁴** | **[409]** |
| Ramified | p = 5 | separate (conductor 125=5³) | — |

Together [404]+[409] classify ALL unramified GL₄/ℚ AI Euler factors for f.

## Structural Gap from Naive Substitution (C4)

A key distinction: setting T=N=0 in the SPLIT formula (cert [404]) gives:
```
(1 + pY²)² = 1 + 2p·Y² + p²·Y⁴     (naive, WRONG for inert p)
```
The correct induction gives:
```
1 + p²·Y⁴                              (correct, no Y² term)
```
**Y² coefficient gap = 2p for every inert prime p.** This shows the inert formula is NOT obtained by zeroing out traces — the local structure is fundamentally different.

The reason: at split p, the AI Euler factor is a PRODUCT of two GL₂/F local factors (one at 𝔭, one at 𝔭̄). At inert p, there is only ONE prime 𝔭=(p) and the AI factor is a COMPOSITION L_p(Y) = L_𝔭(Y²).

## Palindrome and Ramanujan

Palindrome (weight-2, GL₄): a₄=p²·a₀=p², a₃=p·a₁=0 ✓

GL₄ Ramanujan at inert primes: roots of 1+p²Y⁴=0 satisfy |Y|⁴=p^{−2}, so |Y|=p^{−1/2}. This is the same magnitude as split prime roots (cert [404]): consistent GL₄ Ramanujan across all prime types.

## Inert Prime Table (22 primes p ≤ 200)

| p | p mod 5 | P_p coefficients |
|---|---|---|
| 2 | 2 | [1, 0, 0, 0, 4] |
| 3 | 3 | [1, 0, 0, 0, 9] |
| 7 | 2 | [1, 0, 0, 0, 49] |
| 13 | 3 | [1, 0, 0, 0, 169] |
| 17 | 2 | [1, 0, 0, 0, 289] |
| 23 | 3 | [1, 0, 0, 0, 529] |
| 37 | 2 | [1, 0, 0, 0, 1369] |
| ... | ... | [1, 0, 0, 0, p²] |

## Checks

- **C1**: p ≡ 2 or 3 mod 5 (inert condition) — PASS (22/22)
- **C2**: Palindrome a₄=p², a₁=a₂=a₃=0 — PASS (22/22)
- **C3**: GL₄ Ramanujan |Y|=p^{−1/2} for all 4 roots — PASS (22/22)
- **C4**: Structural gap: inert a₂=0 vs naive-split a₂=2p, gap=2p>0 — PASS (22/22)

## Chain

- Parallel to [404] (split prime AI Euler factor)
- Both branch from [403] (GL₂/F CM Ramanujan — proves |σᵢ(a_p)|<2√p at split primes)
- The CM vanishing theorem (Step 1) is the inert analog of cert [403]'s Ramanujan equality
- [409] completes the unramified prime classification; p=5 (ramified, level 125=5³) is separate

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified the inert Euler
polynomial [1,0,0,0,p²] for all 7 displayed primes {2,3,7,13,17,23,37}
in a fresh script — exact match. This table is much simpler than its
[404]-[408] siblings (no T,N-dependent coefficients to transcribe), and
had no errors.
