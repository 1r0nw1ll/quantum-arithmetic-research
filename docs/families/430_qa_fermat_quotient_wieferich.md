<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wieferich (1909) doi:10.1515/crll.1909.136.293, Sun & Sun (1992) doi:10.4064/aa-60-4-371-388 -->
# [430] QA Fermat Quotient Parallel Structure

**Cert family**: `qa_fermat_quotient_wieferich_cert_v1`
**Claim**: The p²-layer Wall-Sun-Sun condition on Fibonacci numbers ([429]) and
the p²-layer Wieferich condition on powers of 2 are structurally **parallel**
Fermat-quotient vanishings — but **not equivalent and not implicative**. This
cert verifies the Wieferich side directly and confirms the two known
Wieferich primes are NOT Wall-Sun-Sun, giving direct evidence of independence.

## Correction to [429]'s Closing Note

Cert [429] closed with a speculative remark: "if p is WSS, then 2^(p-1) ≡ 1
(mod p²) ... under certain conditions." **This is imprecise and not a known
theorem.** No proof of a WSS → Wieferich implication (or converse) exists in
the literature. This cert documents the corrected relationship.

## The Parallel Structure

| Structure | p-layer (universal, all primes) | p²-layer (rare, exceptional) |
|---|---|---|
| Fibonacci | F_{p−(5/p)} ≡ 0 (mod p) — [428] C3 | F_{p−(5/p)} ≡ 0 (mod p²) — **WSS**, none known < 9.7×10¹⁴ |
| Powers of 2 | 2^(p−1) ≡ 1 (mod p) — Fermat's little theorem | 2^(p−1) ≡ 1 (mod p²) — **Wieferich**, known: {1093, 3511} |

Both p²-layer conditions are exactly the vanishing of a Fermat-quotient-style
residue mod p:

```
Fibonacci:    delta(p) := (F_{p-(5/p)} / p) mod p   = 0   <=>  p is WSS
Powers of 2:  q_p(2)   := (2^(p-1) - 1) / p mod p   = 0   <=>  p is Wieferich
```

These are two instances of the same *shape* of condition (a depth-1 vanishing
that is forced by general theory, paired with a much rarer depth-2 vanishing
that defines an exceptional prime class) applied to different sequences
(Fibonacci numbers vs. powers of 2). Same shape does **not** mean the same
primes satisfy both, or that one forces the other.

## Why Sun & Sun (1992) Connects Them

Sun & Sun showed that **if** the first case of Fermat's Last Theorem failed
for a prime p (i.e., x^p + y^p = z^p with p ∤ xyz), **then** p would have to
be **both** a Wieferich prime **and** a Wall-Sun-Sun prime. Both conditions
are independently *necessary* preconditions for that (now provably impossible,
per Wiles 1995) scenario. This is a shared consequence of a hypothetical
event, not a relationship between the conditions themselves — it does not
make WSS and Wieferich equivalent, nor does either imply the other.

## C1: FLT Baseline

**Claim**: 2^(p−1) ≡ 1 (mod p) for all odd primes p ≤ 5000 (Fermat's little
theorem). The depth-1 universal vanishing, analogous to [428] C3's
F_{p−(5/p)} ≡ 0 (mod p).

Verified: 669/669 primes.

## C2: Fermat Quotient Well-Defined

**Claim**: q_p(2) = (2^(p−1) − 1)/p (mod p) is a well-defined integer in
[0, p−1] for all odd primes p ≤ 5000.

**Implementation note**: computed via `pow(2, p-1, p*p)` — Python's 3-argument
modular exponentiation gives 2^(p−1) mod p² directly, without ever
materializing the full big integer 2^(p−1) (which has ~1500 decimal digits
for p = 5000). Since C1 guarantees 2^(p−1) ≡ 1 (mod p), the result r satisfies
r ≡ 1 (mod p), so (r−1) is exactly divisible by p and `(r-1)//p` is the
Fermat quotient. Pure integer throughout; S2-compliant.

Verified: 669/669 primes.

## C3: Wieferich Primes Exact Match

**Claim**: The primes p ≤ 5000 with q_p(2) ≡ 0 (mod p) are exactly {1093, 3511}
— matching the historical record (Meissner 1913 found 1093; Beeger 1922 found
3511; no others are known to exist at all, let alone below 5000).

Verified: exact set match.

## C4: Independence Check

**Claim**: 1093 and 3511 (the two known Wieferich primes) are **not**
Wall-Sun-Sun: F_{α(p)} ≢ 0 (mod p²) for both, where α(p) is the rank of
apparition (computed via the same `rank_of_apparition`/`fib_fast` used in
[428]/[429]).

This is direct empirical evidence that the p²-layer Fibonacci vanishing and
the p²-layer powers-of-2 vanishing are **uncoupled** — the only two primes
known to satisfy one condition do not satisfy the other.

Verified: 2/2 (neither Wieferich prime is WSS).

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 | 2^(p−1) ≡ 1 (mod p); 669 primes ≤ 5000 | **PASS** |
| C2 | q_p(2) well-defined in [0,p−1]; 669 primes | **PASS** |
| C3 | Wieferich primes ≤ 5000 = {1093, 3511} exactly | **PASS** |
| C4 | 1093, 3511 are NOT WSS (independence) | **PASS** |

## Theorem NT Factorisation

```
QA layer (pure integer):
  pow(2, p-1, p) / pow(2, p-1, p*p) — modular exponentiation, builtin, integer
  fermat_quotient_2(p) — (r-1)//p exact integer division
  fib_fast(n, p*p), rank_of_apparition(p) — fast doubling / linear walk, as in [428]/[429]

Observer layer: none (no floats, no statistics, direct equality checks only)
```

## Langlands Ladder

| Cert | Rung |
|------|------|
| [428] | α(p) ∣ p−(5/p); F_p=(5/p); L_p=1 — depth-1 Fibonacci baseline |
| [429] | α(p²)=p·α(p); T(p²)=p·T(p); no WSS prime ≤ 500 — depth-2 Fibonacci, rare |
| **[430]** | **q_p(2) well-defined; Wieferich={1093,3511}; independent of WSS — parallel depth-2 structure, corrected relationship** |

[430] closes the WSS/Wieferich speculation opened (imprecisely) at the end of
[429]: the two depth-2 conditions are parallel in *shape*, independently
necessary for a now-impossible FLT scenario (Sun & Sun 1992), but neither
implies nor is equivalent to the other.
