<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Shimura (1975) doi:10.1007/BF01403156, Gelbart-Jacquet (1978) doi:10.2307/1971237 -->
# [408] QA Langlands Sym² Full GL₆/Q Euler Factor at Split Primes

**Cert family**: `qa_sym2_full_euler_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a and split prime p (p ≡ 1 mod 5), the full Sym² Euler factor at p is:

```
V_p(Y) = (1−pY)² · Σ_p(Y)
```

where Σ_p is the degree-4 reduced factor from cert [407].

## Expanded Form

```
V_p[k] = Σ_p[k] − 2p·Σ_p[k−1] + p²·Σ_p[k−2]   (Σ_p[−1]=Σ_p[−2]=0)
```

giving a degree-6 integer polynomial:
```
V_p = [1, −(S+2p), Q+2p²+2pS, central, p²(Q+2p²+2pS), −p⁴(S+2p), p⁶]
```
where S=T²−2N−4p and Q=(N+2p)²−2pT² from cert [407].

## Langlands Ladder — Complete Sym² Branch

| Rung | Cert | Content |
|---|---|---|
| GL₂/F CM | [403] | Ramanujan equality + Universal Pell |
| GL₄/Q AI | [404] | P_p(Y) induction Euler factor |
| GL₃/F Sym² reduced | [407] | Σ_p — product of quadratic factors |
| **GL₆/Q Sym² full** | **[408]** | **THIS — V_p = (1−pY)²·Σ_p** |

Parallel (∧² branch): [404]→[405]→[406], V_p² = (1−pY)²·R_p.

## The (1−pY)² Factor

The two trivial GL₁/F components of Sym²(f) at 𝔭 each contribute a factor `(1−pY)` to the Euler product. Their product `(1−pY)²` reflects the CM structure: the central Satake parameter at each 𝔭 is the norm p.

Together with Σ_p = (1−c₁Y+p²Y²)·(1−c₂Y+p²Y²) (product of the non-trivial parts), the full GL₃/F Sym² factor at 𝔭 is `(1−pY)·(1−c₁Y+p²Y²)`, and at 𝔭̄ is `(1−pY)·(1−c₂Y+p²Y²)`. The base-change GL₆/Q Sym²/Q Euler factor at p is their product:

```
V_p = (1−pY)²·(1−c₁Y+p²Y²)·(1−c₂Y+p²Y²) = (1−pY)²·Σ_p
```

## GL₆ Palindrome

V_p satisfies the weight-2 GL₆ functional equation:
```
V_p(Y) = p⁶·Y⁶·V_p(1/(p²Y))
```
In coordinates: a₆=p⁶, a₅=p⁴·a₁, a₄=p²·a₂, a₃ self-dual.

## GL₆ Ramanujan (C4)

- (1−pY)² roots: Y=p⁻¹ (multiplicity 2), |root|=p⁻¹ ✓
- Σ_p quadratic factor roots: |root|=p⁻¹ from cert [407] C4 ✓
- All 6 roots of V_p have magnitude p⁻¹ — Ramanujan for GL₆/Q Sym² ✓

## V_p Sample Values

| p | T | N | Coefficients [a₀,...,a₆] |
|---|---|---|---|
| 11 | −1 | −31 | [1, −41, 840, −11220, 101640, −600281, 1771561] |
| 31 | −11 | −1 | [1, −61, −960, 117180, −922560, −56334781, 887503681] |
| 41 | 9 | −11 | [1, −21, −1560, 60680, −2622360, −59340981, 4750104241] |
| 311 | 49 | 569 | [1, −641, 227040, −77383020, 21959535840, −5996524130081, 904820297013361] |

## Checks

- **C1**: All 7 integer coefficients ∈ ℤ — PASS (22/22)
- **C2**: Weight-2 palindrome a₆=p⁶, a₅=p⁴a₁, a₄=p²a₂ — PASS (22/22)
- **C3**: V_p = (1−pY)² · Σ_p by dual convolution paths — PASS (22/22)
- **C4**: GL₆ Ramanujan — all 6 roots |root|=p⁻¹ — PASS (22/22)

## Chain

- Builds on [407] (Σ_p reduced factor) and [404] (GL₄ AI, provides T,N)
- Parallel to [406] (∧² branch: W_p = (1−pY)²·R_p)
- Together [406]+[408] certify both tensor branches of AI(f) at split primes
- Full Sym²/Q L-function: L(s, Sym²f/Q) = ∏_p V_p(p⁻ˢ)⁻¹ (split p)

## Verification Note (2026-07-07)

Found and fixed one wrong row: p=311 showed `[1,−681,197688,−90427260,...]`
in the doc but the validator's own `sym2_full_poly` formula gives
`[1,−641,227040,−77383020,21959535840,−5996524130081,
904820297013361]` — confirmed by fresh independent recomputation. This
is the second doc in this sub-cluster (after [407]) where specifically
the p=311 sample row was mistranscribed while the smaller-prime rows
were correct — worth flagging as a recurring specific-value error when
citing p=311 data from this branch of the ladder. The p=11/31/41 rows
were already correct. Validator itself unaffected (`ok:true` confirmed,
recomputes fresh at runtime).
