# [393] QA BEDA Hard Problem Analysis

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_beda_hard_problem_cert_v1/`

## Claim

The BEDA cipher exists in three distinct instantiations with fundamentally different security properties:

| Instantiation | Hard Problem | Security |
|---|---|---|
| BEDA-toy (mod 9, Cosmos×Satellite) | Exhaustive search over 192 elements | ZERO — O(192) brute force |
| BEDA-DLP (Fibonacci orbit mod prime p) | DLP in ⟨φ⟩ ⊂ (ℤ[φ]/p)× | Classical: hard. Post-quantum: BROKEN (Shor) |
| BEDA-LWE (hypothetical Module-LWE over ℤ[φ]^k) | Module-LWE in rank-k module | Post-quantum IF k≥128, not yet instantiated |

## BEDA-toy: zero security

The proof-of-concept (beda-cipher-poc_v1.md) uses:
- Cosmos orbit: 24 states
- Satellite orbit: 8 states  
- Private key: (cosmos_idx, satellite_idx) ∈ {0..23} × {0..7}
- Public key: cosmos[c] + satellite[s] (component-wise mod-9 addition)

**Keyspace = 24 × 8 = 192.** Any public key is recovered by enumerating all 192 combinations. The cipher's own source code includes a `bruteForceAttack` function demonstrating this. The Perplexity assessment that called it "elegant" was evaluating the mathematical structure, not the security level.

## BEDA-DLP: classically hard, not post-quantum

A hardened version replaces the mod-9 linear combination with the Fibonacci discrete logarithm:

- Private key: k ∈ {0,...,π(p)−1}
- Public key: σ^k(1,0) mod p
- Shared secret: σ^(a+b)(1,0) mod p = σ^a(σ^b(1,0)) mod p

This is Diffie-Hellman in the cyclic group ⟨φ⟩ of order π(p) inside (ℤ[φ]/p)×.

**Classical hardness**: For p=2017, π(2017)=4036 with largest prime factor 1009. Baby-step giant-step costs O(√1009) ≈ 32 steps for this toy prime; for a cryptographic prime with π(p) itself prime and ≈ 2^256, cost is O(2^128) — classically hard.

**Post-quantum vulnerability**: The group ⟨φ⟩ is cyclic of **publicly known** order π(p) (computable in O(p) time; proved exactly by cert [392]). Shor's algorithm (1994) solves DLP in any finite cyclic group of known order in polynomial quantum time. Therefore BEDA-DLP is **not post-quantum secure**. Notably, the proved period from cert [392] makes the attack *easier*, not harder — the attacker doesn't need to determine the order.

## Path to post-quantum: Module-LWE over ℤ[φ]

ℤ[φ] = ℤ[x]/(x²−x−1) is the ring of integers of ℚ(√5), degree 2. CRYSTALS-Kyber uses ℤ[x]/(x^256+1), degree 256. A post-quantum extension of BEDA would require:

1. **Dimensional extension**: Replace ℤ[φ] (degree 2) with a degree-256 ring, or use a rank-128 module ℤ[φ]^128 to reach effective dimension 256.
2. **LWE noise parameter**: Add Gaussian noise e with ‖e‖ ≪ q/2 to enable correct decryption while making the noiseless system hard to recover.
3. **Security reduction**: Reduce breaking the scheme to solving Module-LWE in ℤ[φ]^k — a standard assumption in lattice crypto but not yet proved for this specific ring.

**Current status**: structural motivation only. The ℤ[φ] ring connection to CRYSTALS-Kyber is real, but no concrete parameter set, noise schedule, or security reduction exists for a ℤ[φ]-based PQC scheme.

## Checks

| Check | Result |
|---|---|
| TOY_KEYSPACE: private key space = 192 | PASS |
| TOY_BRUTE_FORCE: all 20 test keys cracked in ≤ 192 attempts | PASS |
| TOY_PROTOCOL_CORRECT: shared secrets always match | PASS |
| DLP_GROUP_ORDER: σ^π(p)(1,0) = (1,0) for p ∈ {31,47,89,113} | PASS |
| DLP_CLASSICAL_HARD: π(2017)=4036, LPF=1009 > 500 | PASS |
| DLP_SHOR_BREAKS: group cyclic + order public → Shor applies | PASS |
| LWE_DIMENSION_GAP: toy dimension 2, required 256, gap=254 | PASS |

8 fixtures: 7 PASS, 1 designed FAIL (claim keyspace > 10^30).

## Primary sources

- Stinson, D.R. (2006). *Cryptography: Theory and Practice*, 3rd ed. ISBN 978-1-58488-508-5. Key exchange security reductions, DH hardness assumptions.
- Boneh, D. & Shoup, V. (2023). *A Graduate Course in Applied Cryptography*. https://toc.cryptobook.us. DLP hardness, Shor's algorithm reduction, LWE.
- Wall, D.D. (1960). Fibonacci series modulo m. *American Mathematical Monthly* 67(6):525–532. doi:10.1080/00029890.1960.11989541. Group order = π(p), used by Shor attack.
