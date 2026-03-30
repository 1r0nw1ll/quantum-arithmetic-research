---
family: 124
title: QA Security Competency Cert
schema_version: QA_SECURITY_COMPETENCY_CERT.v1
status: ACTIVE
parent: QA_AGENT_COMPETENCY_CERT.v1 (family 123)
created: 2026-03-27
---

# [124] QA Security Competency Cert

## Purpose

Certifies that a security algorithm profile is structurally valid and **quantum-resilience-complete** within the QA Lab immune system.

Extends family [123] (QA Agent Competency) with immune system structure: security role, immune function, and post-quantum readiness. The defining new invariant is **SC5 (quantum resilience)**: any identity or membrane algorithm that is classical-only must declare a migration path to a post-quantum replacement. This is a machine-checkable commitment — not advisory.

## The QA Lab Immune System

QA Lab's immune architecture has three layers:

| Layer | Function | Algorithms |
|---|---|---|
| **Detection** | Recognize malformed certs, drift, spoofed spawns, protocol violations | Ed25519/ML-DSA (verify), SHA-3 (fingerprint), Bloom filter (fast screen), ZKP |
| **Containment** | Quarantine bad agents, block unsafe organ formation, freeze suspicious requests | AES-GCM/ChaCha20 (seal channel), Merkle tree (tamper-detect), HMAC (auth) |
| **Recovery** | Rekey, rotate trust roots, rebuild from stem, revalidate through certs | HKDF (rekey), Shamir (key reconstruction), Transparency log (audit), DKG (ceremony) |

## Security Roles (SC2)

| Role | Meaning | Example algorithms |
|---|---|---|
| `identity` | Signatures, attestations, provenance | Ed25519, ML-DSA, RSA-PSS, SLH-DSA |
| `membrane` | Encryption, authenticated channels, key exchange | AES-GCM, ChaCha20-Poly1305, X25519, ML-KEM |
| `integrity` | Hashes, Merkle proofs, append-only logs | SHA-3, SHA-2, HMAC, Merkle tree, Transparency log |
| `self_nonself` | Policy validation, cert lineage, membership tests | Bloom filter, Certificate pinning, ZKP |
| `healing` | Rollback, rekey, dedifferentiate, rebuild from stem | HKDF, PBKDF2, Argon2 |
| `collective` | Quorum, threshold, distributed trust | Shamir, Threshold signature, MPC, DKG |

## PQ Readiness Levels (SC4)

| Level | Meaning |
|---|---|
| `fips_final` | NIST FIPS standard finalized — **deploy now** |
| `in_progress` | NIST selected but spec not yet final |
| `classical_only` | No PQ variant — must declare `pq_migration_path` (SC5) |
| `hybrid_transitional` | Classical + PQ hybrid — transition-era |

## NIST PQC Standards (deploy now)

| Algorithm | Standard | Role | Status |
|---|---|---|---|
| ML-KEM | FIPS 203 (Aug 2024) | membrane (KEM) | **Deploy now** |
| ML-DSA | FIPS 204 (Aug 2024) | identity (signature) | **Deploy now** |
| SLH-DSA | FIPS 205 (Aug 2024) | identity (hash-based sig) | **Deploy now** |
| Falcon/FN-DSA | FIPS 206 (in progress) | identity (compact sig) | In progress |
| HQC | Selected 2024, spec pending | membrane (backup KEM) | In progress |

## Validator Checks

| Code | Check | Error |
|---|---|---|
| SC1 | schema_version matches | `SCHEMA_VERSION_MISMATCH` |
| SC2 | security_role is known | `UNKNOWN_SECURITY_ROLE` |
| SC3 | immune_function is known | `UNKNOWN_IMMUNE_FUNCTION` |
| SC4 | pq_readiness is known | `UNKNOWN_PQ_READINESS` |
| SC5 | identity/membrane + classical_only → pq_migration_path | `PQ_MIGRATION_REQUIRED` |
| SC6 | fips_final → nist_fips non-empty | `MISSING_FIPS_DESIGNATION` |
| SC7 | failure_modes non-empty | `EMPTY_FAILURE_MODES` |
| SC8 | composition_rules non-empty | `EMPTY_COMPOSITION_RULES` |
| SC9 | cell/orbit consistency | `CELL_ORBIT_MISMATCH` |
| SC10 | goal ≥ 10 chars | `GOAL_TOO_SHORT` |
| SC11 | result field matches actual | `RESULT_MISMATCH` |

## Fixtures

| File | Algorithm | Expected | Key check |
|---|---|---|---|
| `scc_pass_ml_kem.json` | ml_kem | PASS | fips_final + FIPS 203 |
| `scc_pass_ed25519.json` | ed25519 | PASS | classical_only + migration path declared |
| `scc_fail_pq_migration_required.json` | rsa_1024_legacy | FAIL | SC5: identity + classical_only, no migration path |

## Organ Templates

### Detection spine
```
Ed25519 (verify) → SHA-3 (fingerprint) → Bloom filter (fast reject) → Merkle proof (inclusion)
```

### Containment organ
```
HMAC (authenticate) → AES-GCM (seal) → Transparency log (record event)
```

### Collective trust organ
```
Threshold signature (quorum sign) ← Shamir (key shares) ← DKG (key ceremony)
```

### PQ transition organ (hybrid mode)
```
X25519 + ML-KEM (hybrid KEX) → HKDF → AES-GCM or ChaCha20
Ed25519 + ML-DSA (dual signature) → cert attestation
```

## QA Mapping

Security algorithms map to QA orbit concepts:

- **Hash functions** = orbit step hash: H(state) = next orbit state fingerprint
- **Signatures** = orbit commitment: prove knowledge of orbit seed without revealing it
- **Key exchange** = orbit meeting point: both parties reach the same group element via different paths
- **Merkle tree** = orbit commitment tree: commits the entire cert ecosystem (128+ families) to a single hash
- **Transparency log** = orbit history tape: immutable record of every orbit-step (cert issuance)
- **Threshold signature** = collective orbit step: requires K agents to jointly advance the orbit
- **ZKP** = orbit membership proof without trace disclosure

## SC5 Rationale — The Quantum Resilience Invariant

SC5 is the defining check of this cert family. It enforces:

> Any algorithm responsible for identity or channel security (roles: identity, membrane) that has no post-quantum variant MUST declare how it will be replaced when quantum computers arrive.

This is not advisory. It is a machine-checked invariant. Certs that declare `classical_only` without a `pq_migration_path` fail with `PQ_MIGRATION_REQUIRED`.

The invariant captures the NIST position: "The three PQC standards [FIPS 203/204/205] can and should be put into use now." Classical-only identity and membrane algorithms are on a countdown.
