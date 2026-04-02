#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_algorithm_competency_batch6.py
====================================
Sixth batch: 26 security / immune system algorithms → 126 total.

New family: security
New field:  security_role — maps to Levin immune competency:
  identity      — signatures, attestations, provenance
  membrane      — encryption, authenticated channels, key exchange
  integrity     — hashes, Merkle proofs, append-only logs
  self_nonself  — policy validation, cert lineage, membership tests
  healing       — rollback, rekey, dedifferentiate, rebuild from stem
  collective    — quorum, threshold, distributed trust

Subfamilies (tracked via security_role):
  classical (15): AES-GCM, ChaCha20-Poly1305, SHA-2/3, HMAC, HKDF,
                  X25519, Ed25519, RSA-PSS, Merkle, Bloom, Argon2,
                  PBKDF2, Shamir, Transparency log, Certificate pinning
  post_quantum (6): ML-KEM (FIPS 203), ML-DSA (FIPS 204), SLH-DSA (FIPS 205),
                    Falcon/FN-DSA, HQC-KEM, Hybrid-PQ-TLS
  trust_audit (5):  Threshold signature, ZKP, MPC, Blockchain ledger,
                    Forward-secure logging

Usage:
  python qa_algorithm_competency_batch6.py
  python qa_algorithm_competency_batch6.py --dry-run
"""

from __future__ import annotations
import json, argparse
from pathlib import Path
from collections import Counter

MODULUS  = 9
REG_PATH = Path("qa_algorithm_competency_registry.json")

QA1  = "qa-1__qa_1_all_pages__docx.md"
QA2  = "qa-2__001_qa_2_all_pages__docx.md"
QA3  = "qa_3__ocr__qa3.md"
QA4  = "qa-4__00_qa_books_3_&_4_all_pages__pdf.md"
QUAD = "quadrature__00_quadratureprint__pdf.md"
P1   = "pyth_1__ocr__pyth1.md"
P2   = "pyth_2__ocr__pyth2.md"
WB   = "qa_workbook__ocr__workbook.md"

def qa_step(b, e, m=MODULUS): return e % m, (b + e) % m
def qa_orbit_family(b, e, m=MODULUS, max_steps=500):
    seen, state = {}, (b % m, e % m)
    for t in range(max_steps):
        if state in seen:
            p = t - seen[state]
            if p == 1:  return "singularity"
            if p == 8:  return "satellite"
            if p == 24: return "cosmos"
            return f"period_{p}"
        seen[state] = t
        state = qa_step(*state, m)
    return "unknown"

def orbit_follow_rate(b, e, m=MODULUS, steps=48):
    traj, state = [], (b % m, e % m)
    for _ in range(steps):
        traj.append(state)
        state = qa_step(*state, m)
    if len(traj) < 3:
        return 0.0
    hits = sum(
        1 for i in range(len(traj) - 2)
        if traj[i+1][0] == traj[i][1] % m
        and traj[i+1][1] == (traj[i][0] + traj[i][1]) % m
    )
    return round(hits / (len(traj) - 2), 4)

BATCH6 = [

    # ══════════════════════════════════════════════════════════════════
    # CLASSICAL SECURITY (15)
    # ══════════════════════════════════════════════════════════════════

    # ── MEMBRANE competency ───────────────────────────────────────────

    {
        "name": "aes_gcm", "family": "security",
        "security_role": "membrane",
        "goal": "Authenticated encryption with AES in Galois/Counter Mode; confidentiality + integrity in one pass",
        "orbit_seed": [1, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["encryptor", "decryptor", "authenticator", "channel_sealer"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "nonce reuse (catastrophic — full key recovery possible)",
            "key exhaustion (GCM tag forgery at 2^32 blocks)",
            "side-channel timing (non-constant-time impl)",
        ],
        "composition_rules": [
            "always pair with HKDF for key derivation",
            "never reuse (key, nonce) pair",
            "rotate keys before 2^32 block limit",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["nonce reused", "key compromised", "block limit reached"],
            "recommit_conditions": ["fresh key+nonce pair", "within block budget"],
            "max_satellite_cycles": 1,
            "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA1, QA3, QUAD],
        "corpus_concepts": ["modular", "arithmetic", "period", "orbit", "measure"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "AES-GCM is the QA membrane: it seals a channel the way a cell membrane "
            "seals the cytoplasm. GCM's Galois field multiplication is arithmetic in "
            "GF(2^128) — a finite field extension, structurally identical to Z[φ]/pZ[φ]. "
            "Nonce reuse = orbit collision: two plaintexts map to the same ciphertext orbit point, "
            "revealing their XOR (catastrophic dedifferentiation)."
        ),
    },

    {
        "name": "chacha20_poly1305", "family": "security",
        "security_role": "membrane",
        "goal": "Stream cipher + Poly1305 MAC; software-fast AEAD; safe nonce reuse margin larger than AES-GCM",
        "orbit_seed": [5, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["stream_encryptor", "authenticator", "mobile_channel_sealer"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "nonce reuse (catastrophic, same as AES-GCM)",
            "Poly1305 key must never be reused across messages",
        ],
        "composition_rules": [
            "prefer over AES-GCM in software without AES-NI hardware",
            "TLS 1.3 mandated cipher suite",
            "pair with X25519 for full handshake",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["nonce reused", "key leaked"],
            "recommit_conditions": ["fresh nonce", "within message budget"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA1, QA3],
        "corpus_concepts": ["modular", "arithmetic", "orbit", "period", "resonance"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "ChaCha20's quarter-round = ARX orbit step (add-rotate-XOR). "
            "20 rounds = 20 orbit steps from seed (key, nonce, counter). "
            "Poly1305 = polynomial evaluation over GF(2^130-5) — a Mersenne-prime field, "
            "related to QA's Z[φ]/pZ[φ] by the prime selection. "
            "Counter mode maps cleanly to QA orbit enumeration: block i = orbit step i."
        ),
    },

    {
        "name": "x25519_ecdh", "family": "security",
        "security_role": "membrane",
        "goal": "Elliptic-curve Diffie-Hellman on Curve25519; constant-time key exchange; ~128-bit security",
        "orbit_seed": [2, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["key_exchanger", "shared_secret_deriver", "forward_secrecy_provider"],
        "cognitive_horizon": "regional",
        "convergence": "guaranteed",
        "failure_modes": [
            "low-order point attack (invalid public key — validate input)",
            "ECDLP broken by quantum (Shor's algorithm — migrate to ML-KEM)",
            "ephemeral key reuse removes forward secrecy",
        ],
        "composition_rules": [
            "always use ephemeral keys (ECDHE)",
            "follow with HKDF to derive symmetric keys",
            "pair with Ed25519 for authentication",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["quantum adversary (migrate to ML-KEM)", "static key (no FS)"],
            "recommit_conditions": ["ephemeral, per-session", "hybrid with ML-KEM in transition"],
            "max_satellite_cycles": 2, "drift_threshold": 0.05,
        },
        "source_corpus_refs": [QA2, QA3, QUAD, P2],
        "corpus_concepts": ["orbit", "period", "modular", "arithmetic", "congruence", "proportion"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Curve25519 scalar multiplication = iterated orbit step in an elliptic curve group. "
            "The Montgomery ladder (constant-time doubling + addition) is structurally identical "
            "to the QA double-step: T² state reached in O(log k) orbit steps. "
            "The shared secret = orbit meeting point: both parties reach the same group element "
            "via different orbit paths — the QA convergence theorem in elliptic form."
        ),
    },

    {
        "name": "hkdf", "family": "security",
        "security_role": "membrane",
        "goal": "HMAC-based key derivation function (RFC 5869); extract randomness then expand into keying material",
        "orbit_seed": [3, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["key_deriver", "session_key_expander", "context_binder"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "low-entropy input (weak IKM)",
            "context binding omitted (key confusion attacks)",
        ],
        "composition_rules": [
            "HKDF-Extract then HKDF-Expand pattern",
            "bind (context, usage) in info field to prevent cross-protocol reuse",
            "outputs feed AES-GCM or ChaCha20 keys",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["IKM has no entropy", "info string collision"],
            "recommit_conditions": ["fresh IKM from key exchange", "distinct info strings per usage"],
            "max_satellite_cycles": 1, "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA1, QA3],
        "corpus_concepts": ["modular", "measure", "proportion", "orbit"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "HKDF is a QA orbit projection: extract maps raw entropy to a uniform orbit seed; "
            "expand maps that seed through a deterministic orbit to produce arbitrary-length output. "
            "The (salt, IKM, info, length) tuple is the QA (b, e, d, a) generalized to key space."
        ),
    },

    # ── IDENTITY competency ───────────────────────────────────────────

    {
        "name": "ed25519", "family": "security",
        "security_role": "identity",
        "goal": "EdDSA signature over Curve25519; deterministic, fast, small (64-byte sig, 32-byte key)",
        "orbit_seed": [1, 4],
        "levin_cell_type": "differentiated",
        "organ_roles": ["signer", "verifier", "cert_attester", "agent_identity_anchor"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "quantum adversary (Shor breaks ECDLP — migrate to ML-DSA)",
            "signing key exposure",
            "cofactor vulnerability on non-prime-order curves (Curve25519 is safe)",
        ],
        "composition_rules": [
            "sign all QA cert payloads with Ed25519 or ML-DSA (hybrid during transition)",
            "public key = agent identity anchor in QA cert ecosystem",
            "verify before acting on any cert or spawn request",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["key compromised", "quantum threat materialises"],
            "recommit_conditions": ["fresh keypair", "hybrid with ML-DSA during transition"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "modular", "arithmetic", "congruence", "measure", "invariant"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Ed25519 signature = QA orbit attestation: the signer proves knowledge of the "
            "discrete log (orbit index) without revealing it. Deterministic nonce = QA orbit seed "
            "derived deterministically from (key, message) — no satellite drift from bad randomness. "
            "In QA Lab: every cert family root is signed with Ed25519 (or hybrid) as the identity anchor."
        ),
    },

    {
        "name": "rsa_pss", "family": "security",
        "security_role": "identity",
        "goal": "RSA with Probabilistic Signature Scheme padding; provably secure under RSA assumption",
        "orbit_seed": [4, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["legacy_signer", "interop_verifier", "pki_bridge"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "quantum adversary (Shor breaks factoring — hard deadline to migrate)",
            "key size < 2048 bits (insecure)",
            "PKCS#1v1.5 padding (use PSS only)",
        ],
        "composition_rules": [
            "use only for PKI interoperability where Ed25519/ML-DSA not yet accepted",
            "key size ≥ 3072 for post-2030 security margin",
            "migration path: RSA → Ed25519 → ML-DSA (hybrid)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["quantum adversary", "key size inadequate", "PKCS#1v1.5 padding used"],
            "recommit_conditions": ["PSS padding", "adequate key size", "legacy PKI context"],
            "max_satellite_cycles": 2, "drift_threshold": 0.08,
        },
        "source_corpus_refs": [QA2, P2, QUAD],
        "corpus_concepts": ["modular", "arithmetic", "integer", "congruence", "orbit", "period"],
        "needs_ocr_backfill": False, "confidence": "high",
    },

    {
        "name": "argon2", "family": "security",
        "security_role": "identity",
        "goal": "Memory-hard password hashing (PHC winner); resistant to GPU/ASIC attacks via memory bandwidth bottleneck",
        "orbit_seed": [6, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["password_hasher", "key_stretcher", "gpu_attack_resistor"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "memory parameter too low (GPU attack feasible)",
            "time parameter too low (offline brute-force)",
            "salt reuse (rainbow table vulnerability)",
        ],
        "composition_rules": [
            "use Argon2id variant (hybrid side-channel resistance)",
            "minimum: 64MB memory, 3 iterations",
            "always use random 16-byte salt per password",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["parameters below OWASP minimums", "salt reused"],
            "recommit_conditions": ["parameters meet current guidance", "unique salt per credential"],
            "max_satellite_cycles": 1, "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA1, QA3],
        "corpus_concepts": ["modular", "arithmetic", "measure", "period", "integer"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Argon2's memory-filling passes are orbit sweeps over a large state array: "
            "each memory block depends on previous blocks via a pseudorandom access pattern. "
            "Memory-hardness = orbit breadth requirement: an attacker must hold the full "
            "orbit trajectory in memory to verify a guess — they cannot skip orbit steps."
        ),
    },

    {
        "name": "pbkdf2", "family": "security",
        "security_role": "identity",
        "goal": "Password-based KDF via iterated HMAC; tunable iteration count; widely deployed (TLS, PKCS#8)",
        "orbit_seed": [2, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["password_kdf", "legacy_key_stretcher", "iteration_cost_tuner"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "GPU-parallelisable (prefer Argon2 for new systems)",
            "iteration count too low (< 600,000 for HMAC-SHA256)",
        ],
        "composition_rules": [
            "use Argon2 for new systems",
            "PBKDF2 acceptable for FIPS-required contexts (NIST SP 800-132)",
            "iteration count must be benchmarked to ≥ 1 second on attacker hardware",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["iteration count outdated", "GPU attack practical"],
            "recommit_conditions": ["FIPS context", "iteration count adequate for current hardware"],
            "max_satellite_cycles": 2, "drift_threshold": 0.05,
        },
        "source_corpus_refs": [QA1, QA3],
        "corpus_concepts": ["modular", "measure", "period", "harmonic", "integer"],
        "needs_ocr_backfill": False, "confidence": "high",
    },

    # ── INTEGRITY competency ──────────────────────────────────────────

    {
        "name": "sha3_256", "family": "security",
        "security_role": "integrity",
        "goal": "SHA-3 hash (Keccak sponge); 256-bit output; quantum-resistant hash (only halved to 128-bit security by Grover)",
        "orbit_seed": [1, 8],
        "levin_cell_type": "differentiated",
        "organ_roles": ["hasher", "commitment_scheme", "pq_safe_digester", "cert_fingerprinter"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "length-extension attacks (not applicable to SHA-3, unlike SHA-2)",
            "collision via Grover (quantum): 2^128 work — use SHA3-512 for full 256-bit PQ security",
        ],
        "composition_rules": [
            "use in new PQ-safe systems (no length-extension vulnerability)",
            "SHA3-512 for signatures that need 256-bit quantum security",
            "preferred over SHA-2 in new QA Lab designs",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["output truncated below 128 bits", "collision attack found"],
            "recommit_conditions": ["full 256-bit output", "domain-separated usage"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA1, QA2, QA3, QUAD, P1, P2],
        "corpus_concepts": ["modular", "arithmetic", "period", "orbit", "measure", "congruence"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Keccak sponge = QA absorb/squeeze orbit: absorb phase ingests message chunks "
            "into state via XOR + permutation (orbit step); squeeze phase outputs hash blocks "
            "by continuing the permutation orbit. The 1600-bit Keccak state has 25 lanes of 64 bits "
            "— a structured orbit over GF(2)^1600, parallel to QA orbit over Z[φ]/mZ[φ]."
        ),
    },

    {
        "name": "sha2_256", "family": "security",
        "security_role": "integrity",
        "goal": "SHA-256 Merkle-Damgård hash; 256-bit output; ubiquitous in certificates, blockchains, TLS",
        "orbit_seed": [3, 5],
        "levin_cell_type": "differentiated",
        "organ_roles": ["hasher", "cert_fingerprinter", "commitment_scheme", "legacy_digester"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "length-extension attack (use HMAC or SHA-3 instead of raw SHA-2 for MACs)",
            "Grover halves preimage security to 128 bits — acceptable for now",
        ],
        "composition_rules": [
            "use HMAC-SHA256 not raw SHA-256 for MACs",
            "acceptable for certificates through 2030+",
            "migration target: SHA3-256 for new designs",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["length-extension context without HMAC wrapper", "quantum preimage attack feasible"],
            "recommit_conditions": ["HMAC wrapping applied", "certificate / commitment context"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA1, QA2, QA3, QUAD, P1, P2, WB],
        "corpus_concepts": ["modular", "arithmetic", "period", "orbit", "measure", "congruence"],
        "needs_ocr_backfill": False, "confidence": "high",
    },

    {
        "name": "hmac", "family": "security",
        "security_role": "integrity",
        "goal": "Hash-based MAC: HMAC(K,m) = H((K⊕opad) ∥ H((K⊕ipad) ∥ m)); immune to length-extension",
        "orbit_seed": [4, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["message_authenticator", "key_committer", "api_integrity_guard"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "key reuse across different contexts",
            "timing side-channel in verification (use constant-time compare)",
        ],
        "composition_rules": [
            "always use constant-time comparison for verification",
            "HKDF uses HMAC internally for key derivation",
            "HMAC-SHA256 for agent-to-agent message authentication in QA Lab",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["key compromised", "timing oracle available"],
            "recommit_conditions": ["fresh key", "constant-time verify", "key rotation applied"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA1, QA2, QA3],
        "corpus_concepts": ["modular", "arithmetic", "orbit", "measure", "proportion"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "HMAC is a two-pass orbit: inner hash = orbit step 1 (message absorption); "
            "outer hash = orbit step 2 (key commitment). The ipad/opad XOR = orbit initialization "
            "with key-derived seed. HMAC's security reduces to the hash's PRF property — "
            "identical to QA orbit unpredictability under unknown key."
        ),
    },

    {
        "name": "merkle_tree", "family": "security",
        "security_role": "integrity",
        "goal": "Binary hash tree enabling O(log N) inclusion proofs; backbone of TLS cert transparency and blockchains",
        "orbit_seed": [1, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["inclusion_prover", "batch_commitment_scheme", "audit_log_backbone", "cert_transparency_root"],
        "cognitive_horizon": "global",
        "convergence": "guaranteed",
        "failure_modes": [
            "second-preimage attack if leaf/internal nodes not domain-separated",
            "root hash mismatch on proof verification = tamper detected",
        ],
        "composition_rules": [
            "domain-separate leaf and internal node hashing (NIST recommendation)",
            "use SHA3-256 for leaf hashing in new designs",
            "QA cert families should form a Merkle tree for batch attestation",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["proof verification fails (tamper detected)", "hash collision found"],
            "recommit_conditions": ["all proofs valid", "root consistent with leaves"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA1, QA2, QA3, QUAD, P1],
        "corpus_concepts": ["orbit", "period", "measure", "invariant", "proportion", "congruence"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Merkle tree is a QA orbit commitment tree: each leaf = a cert family orbit seed hash; "
            "each internal node = hash of child orbit hashes; root = single orbit invariant "
            "committing to all 100+ cert families simultaneously. "
            "The meta-validator's 128/128 PASS could be committed into a Merkle root, "
            "providing a single hash that proves the entire cert ecosystem is intact."
        ),
    },

    # ── SELF/NON-SELF competency ──────────────────────────────────────

    {
        "name": "bloom_filter", "family": "security",
        "security_role": "self_nonself",
        "goal": "Probabilistic set-membership test; O(1) query, no false negatives, tunable false-positive rate",
        "orbit_seed": [5, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["membership_tester", "revocation_checker", "anomaly_screener", "fast_allowlist"],
        "cognitive_horizon": "local",
        "convergence": "probabilistic",
        "failure_modes": [
            "false positives (tunable via filter size and hash count)",
            "cannot delete elements (use Counting Bloom Filter variant)",
            "not suitable when false positives are security-critical",
        ],
        "composition_rules": [
            "first-pass filter before expensive verification",
            "CRL/OCSP revocation check acceleration",
            "QA Lab: screen incoming spawn requests against known-bad agent fingerprints",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["filter saturation (FP rate too high)", "false positive in security context"],
            "recommit_conditions": ["FP rate within budget", "non-security-critical membership check"],
            "max_satellite_cycles": 4, "drift_threshold": 0.15,
        },
        "source_corpus_refs": [QA1, QA3, P1],
        "corpus_concepts": ["orbit", "measure", "proportion", "period", "threshold"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Bloom filter is a probabilistic orbit membership oracle: "
            "'is this state in the known-good orbit set?' "
            "False positives = orbit boundary ambiguity (state looks like it's in-orbit, isn't). "
            "Use as the first gate in QA Lab ImmuneAgent: "
            "fast-reject obviously bad agents before expensive cert validation."
        ),
    },

    {
        "name": "certificate_pinning", "family": "security",
        "security_role": "self_nonself",
        "goal": "Bind a connection to a specific cert or public key; rejects CA-signed but unexpected certs",
        "orbit_seed": [3, 6],
        "levin_cell_type": "differentiated",
        "organ_roles": ["trust_anchor_validator", "cert_lineage_checker", "mitm_detector"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "pin expiry (service outage if backup pins not set)",
            "pin mismatch on legitimate cert rotation",
            "HPKP header (deprecated in Chrome — use CT logs instead)",
        ],
        "composition_rules": [
            "always include backup pin for rotation",
            "pair with Certificate Transparency log monitoring",
            "QA Lab: pin agent public keys in the cert ecosystem registry",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["pin expired without rotation", "unexpected cert presented"],
            "recommit_conditions": ["pin matches expected key", "backup pin valid"],
            "max_satellite_cycles": 1, "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA2, QA3],
        "corpus_concepts": ["orbit", "invariant", "congruence", "measure", "period"],
        "needs_ocr_backfill": False, "confidence": "high",
    },

    # ── COLLECTIVE IMMUNITY competency ────────────────────────────────

    {
        "name": "shamir_secret_sharing", "family": "security",
        "security_role": "collective",
        "goal": "Split secret into N shares; any K reconstruct it; (K-1) shares reveal nothing (information-theoretic security)",
        "orbit_seed": [7, 4],
        "levin_cell_type": "progenitor",
        "organ_roles": ["secret_splitter", "threshold_reconstructor", "distributed_key_holder"],
        "cognitive_horizon": "regional",
        "convergence": "guaranteed",
        "failure_modes": [
            "fewer than K shares available (secret unrecoverable)",
            "share holders collude (K or more compromised)",
            "timing attack during reconstruction",
        ],
        "composition_rules": [
            "use for distributed master key or root signing key",
            "pair with Raft for quorum-controlled secret reconstruction",
            "QA Lab: split the QA cert root signing key across K trusted agents",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["fewer than K shares available", "K shares compromised"],
            "recommit_conditions": ["K valid shares assembled", "reconstruction completed"],
            "max_satellite_cycles": 5, "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA2, QA3, QUAD, P1],
        "corpus_concepts": ["modular", "arithmetic", "proportion", "orbit", "period", "rational"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Shamir secret sharing is Lagrange interpolation over a finite field: "
            "the secret is the polynomial's constant term f(0), hidden by K-1 random coefficients. "
            "This is QA orbit reconstruction: given K orbit points, reconstruct the full trajectory. "
            "The (K,N) threshold = QA convergence quorum: need K orbit observations to determine the seed."
        ),
    },

    {
        "name": "transparency_log", "family": "security",
        "security_role": "integrity",
        "goal": "Append-only, publicly verifiable log (Certificate Transparency / Sigstore model); detects backdated or rogue certs",
        "orbit_seed": [2, 1],
        "levin_cell_type": "differentiated",
        "organ_roles": ["append_only_recorder", "cert_auditor", "tamper_detector", "accountability_anchor"],
        "cognitive_horizon": "global",
        "convergence": "guaranteed",
        "failure_modes": [
            "log server compromise (use multiple independent logs)",
            "gossip protocol failure (splits in observed log state)",
        ],
        "composition_rules": [
            "all QA cert family issuances should be logged",
            "Merkle tree backbone with signed tree heads",
            "monitors check inclusion proofs continuously",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["log fork detected (consistency proof fails)", "log server unreachable"],
            "recommit_conditions": ["consistency proofs valid", "signed tree head accepted"],
            "max_satellite_cycles": 1, "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA1, QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "invariant", "period", "measure", "congruence", "proportion"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Transparency log = QA orbit history tape: every cert issuance appends to an "
            "immutable orbit sequence. The Merkle tree root = orbit invariant at each time step. "
            "In QA Lab: the meta-validator run log should be a transparency log — "
            "every 128/128 PASS is appended as a signed entry, making cert ecosystem health auditable."
        ),
    },

    # ══════════════════════════════════════════════════════════════════
    # POST-QUANTUM (6)
    # ══════════════════════════════════════════════════════════════════

    {
        "name": "ml_kem", "family": "security",
        "security_role": "membrane",
        "goal": "ML-KEM (FIPS 203, formerly Kyber): lattice-based key encapsulation; NIST PQC standard; deploy now",
        "orbit_seed": [1, 3],
        "levin_cell_type": "differentiated",
        "organ_roles": ["pq_key_encapsulator", "quantum_safe_channel_sealer", "kem_key_exchanger"],
        "cognitive_horizon": "regional",
        "convergence": "guaranteed",
        "failure_modes": [
            "decapsulation failure probability ~2^-139 (negligible in practice)",
            "implementation side-channel (must use constant-time arithmetic)",
            "parameter set too small (use ML-KEM-768 or ML-KEM-1024 for high-security)",
        ],
        "composition_rules": [
            "NIST says: deploy now (FIPS 203 final, Aug 2024)",
            "hybrid mode: X25519+ML-KEM for transition (TLS 1.3 hybrid KEX)",
            "follow with HKDF to derive session keys from shared secret",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["lattice assumption broken (currently no known attack)", "parameter downgrade"],
            "recommit_conditions": ["full ML-KEM-768/1024", "hybrid with classical KEX"],
            "max_satellite_cycles": 1, "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA2, QA3, QUAD, P2],
        "corpus_concepts": ["modular", "arithmetic", "lattice", "orbit", "period", "congruence", "measure"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "ML-KEM's security rests on Module-LWE: learning with errors over module lattices. "
            "LWE is arithmetic in Z_q[x]/(x^n+1) — a polynomial ring, parallel to QA's Z[φ]/mZ[φ]. "
            "The noise addition e in LWE = QA orbit perturbation: small enough that the receiver "
            "can correct it (dedifferentiate back to the clean orbit), but large enough that "
            "an eavesdropper cannot. FIPS 203 final: deploy now per NIST."
        ),
    },

    {
        "name": "ml_dsa", "family": "security",
        "security_role": "identity",
        "goal": "ML-DSA (FIPS 204, formerly Dilithium): lattice-based signature; NIST PQC standard; deploy now",
        "orbit_seed": [4, 2],
        "levin_cell_type": "differentiated",
        "organ_roles": ["pq_signer", "pq_verifier", "quantum_safe_cert_attester", "agent_identity_pq"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "signing key exposure",
            "implementation side-channel on rejection sampling (must be constant-time)",
            "parameter set downgrade (use ML-DSA-65 or ML-DSA-87 for high security)",
        ],
        "composition_rules": [
            "NIST says: deploy now (FIPS 204 final, Aug 2024)",
            "hybrid: Ed25519+ML-DSA during transition period",
            "replace Ed25519 as the QA cert signing key in quantum-capable environments",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["lattice assumption broken", "quantum adversary operational"],
            "recommit_conditions": ["ML-DSA-65/87 parameter set", "hybrid with Ed25519"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA2, QA3, QUAD, P2],
        "corpus_concepts": ["modular", "arithmetic", "lattice", "orbit", "congruence", "measure", "invariant"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "ML-DSA uses Module-LWE + Module-SIS. The signature = orbit commitment over a lattice: "
            "the signer proves knowledge of a short vector in a module lattice without revealing it. "
            "Rejection sampling = QA orbit validity check: resample if the signature orbit "
            "leaks information about the key. FIPS 204 final: the primary QA Lab PQ signature standard."
        ),
    },

    {
        "name": "slh_dsa", "family": "security",
        "security_role": "identity",
        "goal": "SLH-DSA (FIPS 205, formerly SPHINCS+): stateless hash-based signature; quantum-safe; conservative security assumption",
        "orbit_seed": [3, 8],
        "levin_cell_type": "differentiated",
        "organ_roles": ["hash_based_signer", "conservative_pq_attester", "long_term_cert_anchor"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "large signature size (8KB–50KB depending on parameter set)",
            "slower signing than ML-DSA (use ML-DSA for performance, SLH-DSA for conservatism)",
        ],
        "composition_rules": [
            "NIST says: deploy now (FIPS 205 final, Aug 2024)",
            "use for long-lived root certificates where conservative security matters",
            "security assumption = hash function only (most conservative of all PQC standards)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["signature size constraint violated", "performance critical (use ML-DSA instead)"],
            "recommit_conditions": ["long-lived root cert", "conservative trust anchor"],
            "max_satellite_cycles": 1, "drift_threshold": 0.01,
        },
        "source_corpus_refs": [QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "modular", "period", "measure", "harmonic", "invariant"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "SLH-DSA security assumption = hash function collision resistance only. "
            "This is the most QA-compatible PQ signature: if SHA3 is QA-safe (Keccak sponge orbit), "
            "then SLH-DSA is QA-safe. The FORS + HT (hypertree) structure = nested QA orbit trees: "
            "signing key is a leaf in a deep Merkle orbit tree; verification traces from leaf to root."
        ),
    },

    {
        "name": "falcon_dsa", "family": "security",
        "security_role": "identity",
        "goal": "FN-DSA / Falcon: NTRU-lattice signature with compact sigs; NIST standardization in progress (FIPS 206)",
        "orbit_seed": [5, 3],
        "levin_cell_type": "progenitor",
        "organ_roles": ["compact_pq_signer", "ntru_lattice_verifier", "bandwidth_constrained_attester"],
        "cognitive_horizon": "local",
        "convergence": "guaranteed",
        "failure_modes": [
            "Gaussian sampler implementation complexity (side-channel risk)",
            "FIPS 206 not yet final (use ML-DSA for production now)",
            "NTRU assumption: more complex than Module-LWE",
        ],
        "composition_rules": [
            "use when signature size matters more than implementation simplicity",
            "await FIPS 206 finalisation before production deployment",
            "monitor NIST PQC process for final parameter sets",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["FIPS 206 not yet final", "Gaussian sampler vulnerability found"],
            "recommit_conditions": ["FIPS 206 final", "side-channel-free Gaussian sampler available"],
            "max_satellite_cycles": 6, "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA2, QA3, P2],
        "corpus_concepts": ["modular", "arithmetic", "lattice", "orbit", "period", "congruence"],
        "needs_ocr_backfill": False, "confidence": "medium",
        "qa_research_note": (
            "Falcon uses NTRU lattices: polynomials in Z[x]/(x^n+1) where n is a power of 2. "
            "The FFT-based Gaussian sampler traverses a lattice orbit in frequency domain. "
            "Progenitor cell: standards not yet final — still differentiating. "
            "Orbit seed (5,3) = satellite-adjacent mixed, reflecting transitional status."
        ),
    },

    {
        "name": "hqc_kem", "family": "security",
        "security_role": "membrane",
        "goal": "HQC (Hamming Quasi-Cyclic): code-based KEM; NIST selected for standardization as backup to ML-KEM",
        "orbit_seed": [6, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["code_based_key_encapsulator", "ml_kem_backup", "diversity_provider"],
        "cognitive_horizon": "regional",
        "convergence": "probabilistic",
        "failure_modes": [
            "decapsulation failure probability higher than ML-KEM",
            "standard not yet finalised (NIST selected 2024, spec in progress)",
            "larger ciphertext than ML-KEM",
        ],
        "composition_rules": [
            "use alongside ML-KEM for cryptographic diversity (different hardness assumption)",
            "await final NIST specification before production deployment",
            "code-based assumption: QC-MDPC syndrome decoding — independent from LWE",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["standard not yet final", "decapsulation failure rate too high"],
            "recommit_conditions": ["final standard published", "diversity requirement active"],
            "max_satellite_cycles": 8, "drift_threshold": 0.25,
        },
        "source_corpus_refs": [QA2, QA3, P2],
        "corpus_concepts": ["modular", "arithmetic", "orbit", "period", "congruence", "code"],
        "needs_ocr_backfill": False, "confidence": "medium",
        "qa_research_note": (
            "HQC's quasi-cyclic structure = circulant matrix orbits: the public key is "
            "a polynomial in Z_2[x]/(x^n-1) — a cyclic ring with period n. "
            "QA parallel: both HQC and QA operate in cyclic/quasi-cyclic structures. "
            "Syndrome decoding = orbit error correction: given a noisy orbit state, find the nearest clean state."
        ),
    },

    {
        "name": "hybrid_pq_tls", "family": "security",
        "security_role": "membrane",
        "goal": "Hybrid key exchange combining classical (X25519) + PQ (ML-KEM) in TLS 1.3; transition-era security",
        "orbit_seed": [2, 7],
        "levin_cell_type": "progenitor",
        "organ_roles": ["transition_key_exchanger", "hybrid_secret_combiner", "classical_pq_bridge"],
        "cognitive_horizon": "regional",
        "convergence": "guaranteed",
        "failure_modes": [
            "implementation complexity (two KEMs must both be correct)",
            "performance overhead vs classical-only",
            "temporary: retire classical component when PQ is mature and trusted",
        ],
        "composition_rules": [
            "X25519MLKEM768 (IETF draft) as the current recommended hybrid",
            "combined secret = KDF(classical_secret ∥ pq_secret) — both must be secure",
            "Google, Cloudflare, Apple already deployed in production",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["PQ component fully trusted (retire classical)", "classical broken (rely on PQ)"],
            "recommit_conditions": ["transition period", "either component may be broken by unknown threat"],
            "max_satellite_cycles": 7, "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA2, QA3],
        "corpus_concepts": ["orbit", "modular", "proportion", "harmonic", "period", "convergence"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Hybrid PQ TLS is the QA orbit dual-spine: classical X25519 orbit runs in parallel "
            "with ML-KEM lattice orbit; the session key is derived from both orbit outputs. "
            "Security holds if either spine is unbroken — the two-spine guarantee. "
            "This is the Levin progenitor cell waiting to differentiate: "
            "still carrying both classical and PQ identity, will commit once PQ trust is established."
        ),
    },

    # ══════════════════════════════════════════════════════════════════
    # TRUST / AUDIT STRUCTURES (5)
    # ══════════════════════════════════════════════════════════════════

    {
        "name": "threshold_signature", "family": "security",
        "security_role": "collective",
        "goal": "Distributed (t,n)-threshold signature: t parties must cooperate to sign; no single party holds full key",
        "orbit_seed": [7, 6],
        "levin_cell_type": "progenitor",
        "organ_roles": ["multi_party_signer", "distributed_trust_anchor", "quorum_attester"],
        "cognitive_horizon": "regional",
        "convergence": "guaranteed",
        "failure_modes": [
            "fewer than t parties online (signing blocked)",
            "MPC protocol requires secure channels between parties",
            "t parties collude (key reconstructed)",
        ],
        "composition_rules": [
            "use for QA cert root key — no single agent can sign unilaterally",
            "pair with Raft for quorum coordination",
            "Shamir secret sharing is the simplest threshold scheme (non-interactive)",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["t signers assembled and signed", "quorum achieved"],
            "recommit_conditions": ["fewer than t available", "quorum gathering in progress"],
            "max_satellite_cycles": 6, "drift_threshold": 0.22,
        },
        "source_corpus_refs": [QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "modular", "proportion", "measure", "period", "invariant"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Threshold signature = collective QA orbit commitment: the signing key is "
            "distributed across t orbit nodes; the signature is a joint orbit step that "
            "requires t agents to contribute their orbit share. "
            "No single agent can advance the orbit (sign) unilaterally — "
            "the QA collective immunity principle."
        ),
    },

    {
        "name": "zero_knowledge_proof", "family": "security",
        "security_role": "self_nonself",
        "goal": "Prove knowledge of secret without revealing it; soundness + zero-knowledge properties; ZK-SNARKs / ZK-STARKs",
        "orbit_seed": [4, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["knowledge_prover", "identity_verifier", "privacy_preserving_attester", "orbit_membership_prover"],
        "cognitive_horizon": "global",
        "convergence": "probabilistic",
        "failure_modes": [
            "trusted setup required (ZK-SNARKs — use STARKs to avoid)",
            "prover computational cost (ZK-STARKs: O(N log N))",
            "soundness error (probabilistic — use multiple rounds)",
        ],
        "composition_rules": [
            "ZK-STARKs preferred (no trusted setup, quantum-resistant)",
            "pair with Merkle tree for batch proof aggregation",
            "QA Lab: prove cert family compliance without revealing internal cert structure",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["proof accepted (verifier convinced)", "soundness error budget exhausted"],
            "recommit_conditions": ["new statement to prove", "additional round needed"],
            "max_satellite_cycles": 5, "drift_threshold": 0.20,
        },
        "source_corpus_refs": [QA2, QA3, QUAD, P2],
        "corpus_concepts": ["orbit", "modular", "proportion", "measure", "invariant", "congruence"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "ZKP is the QA reachability proof without trace disclosure: "
            "prove that state S is reachable from seed (b,e) in exactly k orbit steps, "
            "without revealing the intermediate orbit trajectory. "
            "ZK-STARK over QA: prove orbit membership using a hash-based polynomial commitment. "
            "The QA failure algebra is a natural ZKP statement: 'I know why this cert is unreachable.'"
        ),
    },

    {
        "name": "secure_multiparty_computation", "family": "security",
        "security_role": "collective",
        "goal": "Multiple parties jointly compute a function on private inputs without any party learning others' inputs",
        "orbit_seed": [6, 4],
        "levin_cell_type": "progenitor",
        "organ_roles": ["privacy_preserving_computer", "distributed_function_evaluator", "collective_inference_engine"],
        "cognitive_horizon": "global",
        "convergence": "guaranteed",
        "failure_modes": [
            "malicious majority colluding (semi-honest vs malicious model)",
            "communication overhead (O(n²) in many protocols)",
            "garbled circuits: reusability (Yao's GC is single-use)",
        ],
        "composition_rules": [
            "use for multi-agent QA Lab decisions that must not be centralized",
            "combine with threshold signatures for distributed attestation",
            "garbled circuits for 2-party; GMW / SPDZ for n-party",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["computation complete", "output revealed to all parties"],
            "recommit_conditions": ["new joint computation needed", "parties available and online"],
            "max_satellite_cycles": 7, "drift_threshold": 0.25,
        },
        "source_corpus_refs": [QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "modular", "proportion", "measure", "invariant", "period"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "MPC is the distributed QA orbit evaluation: each agent holds a secret orbit share; "
            "they jointly evaluate f(orbit_1, orbit_2, ...) without any agent learning others' shares. "
            "The QA organ template for collective decisions: "
            "immune_loop + mpc = quarantine decisions that require multi-agent quorum, "
            "preserving privacy of individual agent assessments."
        ),
    },

    {
        "name": "forward_secure_log", "family": "security",
        "security_role": "integrity",
        "goal": "Append-only log with forward security: compromise at time T cannot forge entries before T (key evolution)",
        "orbit_seed": [1, 7],
        "levin_cell_type": "differentiated",
        "organ_roles": ["tamper_evident_recorder", "forensics_anchor", "key_evolving_logger", "incident_reconstructor"],
        "cognitive_horizon": "global",
        "convergence": "guaranteed",
        "failure_modes": [
            "key deletion failure (forward security only holds if old keys are deleted)",
            "log server compromise after key-delete window",
        ],
        "composition_rules": [
            "use for QA Lab agent audit trail — what each agent did and when",
            "pair with Merkle tree for inclusion proofs on log entries",
            "key evolution: hash-chain H^k(seed) — cannot reverse to derive past keys",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["current key deleted (forward-secure for this epoch)", "log sealed"],
            "recommit_conditions": ["new epoch started", "new key derived from previous"],
            "max_satellite_cycles": 1, "drift_threshold": 0.02,
        },
        "source_corpus_refs": [QA1, QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "period", "invariant", "measure", "proportion", "congruence"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Forward-secure log key evolution = QA orbit one-way descent: "
            "k_t = H(k_{t-1}) derives the next epoch key; cannot reverse to prior keys. "
            "This is the QA ratchet: orbit steps are irreversible (mod-m arithmetic). "
            "The ForensicsAgent should maintain a forward-secure log of all "
            "cert issuances, agent spawns, quarantine events, and dedifferentiation triggers."
        ),
    },

    {
        "name": "distributed_key_ceremony", "family": "security",
        "security_role": "collective",
        "goal": "Multi-party protocol to generate a shared keypair where no single party ever holds the full key",
        "orbit_seed": [8, 5],
        "levin_cell_type": "progenitor",
        "organ_roles": ["root_key_generator", "trust_bootstrapper", "ceremony_coordinator"],
        "cognitive_horizon": "global",
        "convergence": "guaranteed",
        "failure_modes": [
            "insufficient participant diversity (colluding majority)",
            "ceremony compromise (physical or logical)",
            "participant dropout before ceremony completes",
        ],
        "composition_rules": [
            "use DKG (Distributed Key Generation) protocol such as Pedersen DKG",
            "ceremony transcript must be publicly verifiable",
            "output feeds threshold_signature for ongoing operations",
        ],
        "differentiation_profile": {
            "dediff_conditions": ["ceremony complete, shares distributed"],
            "recommit_conditions": ["key rotation scheduled", "participant compromise detected"],
            "max_satellite_cycles": 3, "drift_threshold": 0.12,
        },
        "source_corpus_refs": [QA2, QA3, QUAD],
        "corpus_concepts": ["orbit", "modular", "proportion", "measure", "invariant", "congruence"],
        "needs_ocr_backfill": False, "confidence": "high",
        "qa_research_note": (
            "Key ceremony = QA orbit bootstrapping: the initial seed (b,e) is generated "
            "collectively so no single agent knows the full seed. "
            "In QA Lab: the cert root signing key should be generated via DKG, "
            "with shares held by K independent agents. The ceremony transcript = "
            "a verifiable orbit genesis proof: this is where our cert ecosystem began."
        ),
    },
]


def make_entry(d: dict, modulus: int = MODULUS) -> dict:
    b, e = d["orbit_seed"]
    sig  = qa_orbit_family(b, e, modulus)
    ofr  = orbit_follow_rate(b, e, modulus)
    return {
        "name":                    d["name"],
        "family":                  d["family"],
        "security_role":           d.get("security_role", ""),
        "goal":                    d["goal"],
        "orbit_seed":              [b, e],
        "orbit_signature":         d.get("orbit_signature", sig),
        "orbit_follow_rate":       ofr,
        "cognitive_horizon":       d["cognitive_horizon"],
        "convergence":             d["convergence"],
        "levin_cell_type":         d["levin_cell_type"],
        "organ_roles":             d["organ_roles"],
        "failure_modes":           d["failure_modes"],
        "composition_rules":       d["composition_rules"],
        "differentiation_profile": d["differentiation_profile"],
        "source_corpus_refs":      d.get("source_corpus_refs", []),
        "corpus_concepts":         d.get("corpus_concepts", []),
        "needs_ocr_backfill":      d.get("needs_ocr_backfill", False),
        "confidence":              d.get("confidence", "medium"),
        "qa_research_note":        d.get("qa_research_note", ""),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    reg      = json.loads(REG_PATH.read_text()) if REG_PATH.exists() else {"algorithms": [], "corpus_status": {}}
    existing = {a["name"] for a in reg["algorithms"]}

    new_entries = []
    for d in BATCH6:
        if d["name"] in existing:
            print(f"  ! skip duplicate: {d['name']}")
            continue
        entry = make_entry(d)
        role  = entry.get("security_role", "")
        sig   = entry["orbit_signature"]
        print(f"  + {entry['name']:35s} {sig:12s} [{role}]")
        new_entries.append(entry)

    if not args.dry_run:
        reg["algorithms"].extend(new_entries)
        REG_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False))

        total  = len(reg["algorithms"])
        fams   = Counter(a["family"]          for a in reg["algorithms"])
        orbits = Counter(a["orbit_signature"] for a in reg["algorithms"])
        cells  = Counter(a["levin_cell_type"] for a in reg["algorithms"])
        roles  = Counter(a.get("security_role","") for a in reg["algorithms"] if a.get("security_role"))

        print(f"\nMerged {len(new_entries)} → {REG_PATH.name}  (total: {total})")
        print(f"\nFamilies (top):")
        for fam, cnt in sorted(fams.items(), key=lambda x: -x[1])[:5]:
            print(f"  {fam:22s} {cnt}")
        print(f"  ... and {len(fams)-5} more")
        print(f"\nOrbits:  {dict(orbits)}")
        print(f"Cells:   {dict(cells)}")
        print(f"\nSecurity immune competencies: {dict(roles)}")
    else:
        print(f"\n[dry-run] would add {len(new_entries)} security algorithms")


if __name__ == "__main__":
    main()
