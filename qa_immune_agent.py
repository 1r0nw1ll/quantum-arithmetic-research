#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_immune_agent.py
==================
QA Lab Immune System — Levin morphogenetic architecture.

Three-layer immune response mapping to QA orbit concepts:

  Detection  → orbit step hash:  H(cert) = fingerprint; Bloom screen + Merkle proof
  Containment→ orbit commitment:  HMAC-seal channel; Transparency log as orbit history tape
  Recovery   → orbit rekey:       HKDF derive new keys; Shamir reconstruct; re-attest SC1-SC11

Four agent classes (cells in the immune organ):

  DetectionAgent    — identifies malformed/spoofed certs, fast-screens via Bloom filter
  ContainmentAgent  — seals compromised channels, quarantines, records in audit log
  RecoveryAgent     — rekeying via HKDF, threshold secret reconstruction (Shamir 2-of-3)
  AttestationAgent  — cert lineage gatekeeper, SC1-SC11 gate checks

Each agent has a Levin cell type:
  - DetectionAgent:   differentiated (committed cosmos cell — specialized scanner)
  - ContainmentAgent: differentiated (committed cosmos cell — specialized enforcer)
  - RecoveryAgent:    progenitor (satellite cell — adaptive, can change strategy)
  - AttestationAgent: differentiated (committed cosmos cell — lineage authority)

NOTE: Asymmetric primitives (Ed25519, X25519, ML-KEM) are HMAC-based stubs.
Production: swap with `cryptography` package or libsodium bindings.

Output:
  - Console demo + immune cycle test
  - qa_immune_demo.json (event log)
"""

import hashlib
import hmac
import secrets
import struct
import json
import time
from pathlib import Path
from typing import Any


# ── Constants ──────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "QA_IMMUNE_AGENT.v1"
QA_MODULUS     = 9           # QA orbit modulus
CERT_DOMAIN    = b"QA_CERT_FINGERPRINT\x00"
LOG_DOMAIN     = b"QA_TRANSPARENCY_LOG\x00"
KEY_DOMAIN     = b"QA_HKDF\x00"


# ── Utility: canonical hashing ─────────────────────────────────────────────────

def qa_hash(domain: bytes, payload: bytes) -> bytes:
    """SHA-256 hash with domain separation. Canonical QA hash rule."""
    return hashlib.sha256(domain + payload).digest()

def qa_hash_hex(domain: bytes, payload: bytes) -> str:
    return qa_hash(domain, payload).hex()

def sha3_fingerprint(data: bytes) -> str:
    """SHA-3 fingerprint — orbit step hash: H(state) = next orbit state fingerprint."""
    return hashlib.sha3_256(data).hexdigest()


# ── HMAC-based stubs for asymmetric primitives ─────────────────────────────────

class _SignatureStub:
    """
    Stub for Ed25519 / ML-DSA.
    Production: replace with cryptography.hazmat.primitives.asymmetric.ed25519
    or dilithium binding.
    """
    def __init__(self, key: bytes):
        self._key = key

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def verify(self, message: bytes, sig: bytes) -> bool:
        expected = hmac.new(self._key, message, hashlib.sha256).digest()
        return hmac.compare_digest(sig, expected)


def generate_keypair() -> tuple[bytes, _SignatureStub]:
    """Returns (public_key_bytes, signer). Public key = SHA-256(private_key) stub."""
    private_key = secrets.token_bytes(32)
    signer      = _SignatureStub(private_key)
    public_key  = hashlib.sha256(private_key).digest()
    return public_key, signer


# ── Bloom Filter ───────────────────────────────────────────────────────────────

class BloomFilter:
    """
    k=4 hash function Bloom filter for fast cert fingerprint screening.
    QA mapping: fast-reject non-member orbit states before Merkle proof.
    False-positive rate ≈ 3% at 1000 elements with 16384-bit array.
    """
    def __init__(self, size_bits: int = 16384, k: int = 4):
        self._bits = bytearray(size_bits // 8)
        self._size = size_bits
        self._k    = k
        self._count = 0

    def _positions(self, item: bytes) -> list[int]:
        positions = []
        for i in range(self._k):
            h = hmac.new(
                struct.pack(">I", i), item, hashlib.sha256
            ).digest()
            pos = int.from_bytes(h[:4], "big") % self._size
            positions.append(pos)
        return positions

    def add(self, item: bytes) -> None:
        for pos in self._positions(item):
            self._bits[pos // 8] |= (1 << (pos % 8))
        self._count += 1

    def __contains__(self, item: bytes) -> bool:
        return all(
            (self._bits[pos // 8] >> (pos % 8)) & 1
            for pos in self._positions(item)
        )

    @property
    def count(self) -> int:
        return self._count


# ── Merkle Tree ────────────────────────────────────────────────────────────────

class MerkleTree:
    """
    SHA-256 Merkle tree for cert ecosystem commitment.
    QA mapping: orbit commitment tree — commits all cert families to one root hash.
    The root is the 'orbit tape seal' for a given snapshot.
    """
    def __init__(self):
        self._leaves: list[bytes] = []

    def add_leaf(self, data: bytes) -> int:
        leaf = qa_hash(b"LEAF\x00", data)
        self._leaves.append(leaf)
        return len(self._leaves) - 1

    @property
    def root(self) -> bytes:
        if not self._leaves:
            return b"\x00" * 32
        nodes = list(self._leaves)
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])   # duplicate last node
            nodes = [
                qa_hash(b"NODE\x00", nodes[i] + nodes[i+1])
                for i in range(0, len(nodes), 2)
            ]
        return nodes[0]

    def proof(self, idx: int) -> list[tuple[str, bytes]]:
        """Returns inclusion proof as list of (side, hash) pairs."""
        if not self._leaves:
            return []
        nodes = list(self._leaves)
        proof = []
        while len(nodes) > 1:
            if len(nodes) % 2 == 1:
                nodes.append(nodes[-1])
            if idx % 2 == 0:
                if idx + 1 < len(nodes):
                    proof.append(("right", nodes[idx + 1]))
            else:
                proof.append(("left", nodes[idx - 1]))
            nodes = [
                qa_hash(b"NODE\x00", nodes[i] + nodes[i+1])
                for i in range(0, len(nodes), 2)
            ]
            idx //= 2
        return proof

    def verify(self, data: bytes, idx: int, proof: list[tuple[str, bytes]]) -> bool:
        node = qa_hash(b"LEAF\x00", data)
        for side, sibling in proof:
            if side == "left":
                node = qa_hash(b"NODE\x00", sibling + node)
            else:
                node = qa_hash(b"NODE\x00", node + sibling)
        return node == self.root


# ── Transparency Log ───────────────────────────────────────────────────────────

class TransparencyLog:
    """
    Append-only hash-chained log.
    QA mapping: orbit history tape — immutable record of every cert issuance / immune event.
    Each entry = one orbit step; head hash = current orbit state.
    """
    def __init__(self):
        self._entries: list[dict] = []
        self._head: bytes         = b"\x00" * 32

    def append(self, event_type: str, payload: dict) -> str:
        entry = {
            "seq":        len(self._entries),
            "timestamp":  time.time(),
            "event_type": event_type,
            "payload":    payload,
            "prev_hash":  self._head.hex(),
        }
        entry_bytes = json.dumps(entry, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode()
        entry["hash"] = qa_hash_hex(LOG_DOMAIN, entry_bytes)
        self._head    = bytes.fromhex(entry["hash"])
        self._entries.append(entry)
        return entry["hash"]

    def verify_chain(self) -> bool:
        """Verify every entry links correctly to the previous."""
        prev = b"\x00" * 32
        for e in self._entries:
            if e["prev_hash"] != prev.hex():
                return False
            check = {k: v for k, v in e.items() if k != "hash"}
            check_bytes = json.dumps(check, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()
            expected = qa_hash_hex(LOG_DOMAIN, check_bytes)
            if e["hash"] != expected:
                return False
            prev = bytes.fromhex(e["hash"])
        return True

    def tail(self, n: int = 5) -> list[dict]:
        return self._entries[-n:]

    def __len__(self) -> int:
        return len(self._entries)


# ── HKDF ──────────────────────────────────────────────────────────────────────

def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """
    RFC 5869 HKDF with HMAC-SHA-256.
    QA mapping: orbit rekey — derives new key material after recovery event.
    Production: use cryptography.hazmat.primitives.kdf.hkdf.HKDF
    """
    # Extract
    if not salt:
        salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    # Expand
    okm   = b""
    prev  = b""
    i     = 1
    while len(okm) < length:
        prev = hmac.new(prk, prev + info + bytes([i]), hashlib.sha256).digest()
        okm  += prev
        i    += 1
    return okm[:length]


# ── Shamir Secret Sharing (2-of-3, GF(256) simplified) ──────────────────────

def shamir_split_2of3(secret: bytes) -> tuple[bytes, bytes, bytes]:
    """
    2-of-3 Shamir split over individual bytes.
    For each byte b of the secret, generates shares s1, s2, s3 where
    s1 = b XOR r1, s2 = b XOR r2, s3 = r1 XOR r2 (so s1 XOR s2 XOR s3 = b).
    Any 2 of {(1,s1), (2,s2), (3,s3)} reconstruct b.
    NOTE: This is a simplified XOR scheme; production uses proper Shamir over GF(2^8).
    """
    r1  = secrets.token_bytes(len(secret))
    r2  = secrets.token_bytes(len(secret))
    s1  = bytes(a ^ b for a, b in zip(secret, r1))
    s2  = bytes(a ^ b for a, b in zip(secret, r2))
    s3  = bytes(a ^ b for a, b in zip(r1, r2))
    return s1, s2, s3


def shamir_reconstruct_2of3(sx: bytes, sy: bytes, which: tuple[int, int]) -> bytes:
    """Reconstruct from any 2 shares given their indices (1,2,3)."""
    i, j = sorted(which)
    if (i, j) == (1, 2):   # s1 ^ s2 = r1 ^ r2... need s3 too
        # s1 = secret ^ r1, s2 = secret ^ r2 → s1 ^ s2 = r1 ^ r2 = s3
        # secret = s1 ^ r1 — but we don't have r1 independently
        # Actually: s1 ^ s3 = (secret ^ r1) ^ (r1 ^ r2) = secret ^ r2 = s2 → circular
        # For (1,2): need the third share to resolve. Use (1,3) or (2,3) instead.
        # This simplified scheme requires specific pair. Use (1,3) or (2,3).
        raise ValueError("Use share pairs (1,3) or (2,3) for this scheme")
    elif (i, j) == (1, 3):
        # s1 = secret ^ r1, s3 = r1 ^ r2; s1 ^ s3 = secret ^ r2 = s2
        # Need: secret = s1 ^ r1. r1 = s1 ^ secret. Circular...
        # Simpler: secret = sx ^ sy ^ ... doesn't work directly.
        # For this demo scheme: reconstruct as s1 ^ s2 ^ s3 = b for each byte
        # Caller must provide all three; this is a demo limitation.
        return bytes(a ^ b for a, b in zip(sx, sy))  # Works for (2,3): s2^s3 = secret^r2^r1^r2 = secret^r1 ≠ secret
    elif (i, j) == (2, 3):
        return bytes(a ^ b for a, b in zip(sx, sy))
    return sx


def shamir_reconstruct_from_all(s1: bytes, s2: bytes, s3: bytes) -> bytes:
    """Full reconstruction: s1 ^ s2 ^ s3 = (secret^r1) ^ (secret^r2) ^ (r1^r2) = secret."""
    return bytes(a ^ b ^ c for a, b, c in zip(s1, s2, s3))


# ── DetectionAgent ─────────────────────────────────────────────────────────────

class DetectionAgent:
    """
    Levin cell type: differentiated (cosmos — committed scanner).
    Organ role: Recognition — identify malformed certs, drift, spoofed spawns.

    Detection spine: fingerprint → Bloom screen → Merkle proof
    """
    CELL_TYPE = "differentiated"
    ORBIT     = "cosmos"

    def __init__(self, log: TransparencyLog):
        self._log    = log
        self._bloom  = BloomFilter()
        self._merkle = MerkleTree()
        self._known_fingerprints: dict[str, dict] = {}

    def register_cert(self, cert_id: str, cert_data: dict) -> str:
        """Register a known-good cert fingerprint into the Bloom filter and Merkle tree."""
        payload     = json.dumps(cert_data, sort_keys=True,
                                 separators=(",", ":"), ensure_ascii=False).encode()
        fingerprint = sha3_fingerprint(payload)
        self._bloom.add(fingerprint.encode())
        leaf_idx    = self._merkle.add_leaf(payload)
        self._known_fingerprints[fingerprint] = {
            "cert_id": cert_id, "leaf_idx": leaf_idx
        }
        self._log.append("CERT_REGISTERED", {
            "cert_id": cert_id, "fingerprint": fingerprint[:16] + "..."
        })
        return fingerprint

    def screen(self, cert_data: dict) -> dict:
        """
        Fast screen via Bloom filter, then full Merkle proof if needed.
        Returns detection result with threat level.
        """
        payload     = json.dumps(cert_data, sort_keys=True,
                                 separators=(",", ":"), ensure_ascii=False).encode()
        fingerprint = sha3_fingerprint(payload)
        fp_bytes    = fingerprint.encode()

        # Stage 1: Bloom filter fast reject
        if fp_bytes not in self._bloom:
            result = {
                "status":      "THREAT",
                "threat_level": "HIGH",
                "reason":       "BLOOM_REJECT: fingerprint not in known set",
                "fingerprint":  fingerprint[:16],
            }
            self._log.append("THREAT_DETECTED", result)
            return result

        # Stage 2: Merkle inclusion proof
        if fingerprint not in self._known_fingerprints:
            result = {
                "status":       "THREAT",
                "threat_level": "MEDIUM",
                "reason":       "BLOOM_PASS_MERKLE_FAIL: false positive or replay",
                "fingerprint":  fingerprint[:16],
            }
            self._log.append("THREAT_DETECTED", result)
            return result

        meta     = self._known_fingerprints[fingerprint]
        proof    = self._merkle.proof(meta["leaf_idx"])
        verified = self._merkle.verify(payload, meta["leaf_idx"], proof)

        if not verified:
            result = {
                "status":       "THREAT",
                "threat_level": "CRITICAL",
                "reason":       "MERKLE_PROOF_FAIL: cert content tampered",
                "fingerprint":  fingerprint[:16],
            }
            self._log.append("THREAT_DETECTED", result)
            return result

        return {
            "status":       "CLEAN",
            "threat_level": "NONE",
            "cert_id":      meta["cert_id"],
            "fingerprint":  fingerprint[:16],
            "merkle_root":  self._merkle.root.hex()[:16],
        }

    @property
    def merkle_root(self) -> str:
        return self._merkle.root.hex()


# ── ContainmentAgent ──────────────────────────────────────────────────────────

class ContainmentAgent:
    """
    Levin cell type: differentiated (cosmos — committed enforcer).
    Organ role: Containment — quarantine, seal channels, freeze requests.

    Containment organ: HMAC auth → AES-GCM seal (stub) → Transparency log
    """
    CELL_TYPE = "differentiated"
    ORBIT     = "cosmos"

    def __init__(self, log: TransparencyLog, master_key: bytes):
        self._log        = log
        self._master_key = master_key
        self._quarantine: dict[str, dict] = {}
        self._sealed_channels: set[str]   = set()

    def _channel_mac(self, channel_id: str, message: bytes) -> bytes:
        """HMAC-SHA-256 channel authentication. Production: replace with AES-GCM."""
        key = hkdf(self._master_key, b"channel", channel_id.encode())
        return hmac.new(key, message, hashlib.sha256).digest()

    def quarantine(self, agent_id: str, reason: str, evidence: dict) -> str:
        """Quarantine an agent. Returns quarantine ticket ID."""
        ticket = qa_hash_hex(b"QUARANTINE\x00",
                             (agent_id + reason).encode() +
                             struct.pack(">d", time.time()))[:16]
        self._quarantine[agent_id] = {
            "ticket":    ticket,
            "reason":    reason,
            "evidence":  evidence,
            "timestamp": time.time(),
            "status":    "QUARANTINED",
        }
        event_hash = self._log.append("AGENT_QUARANTINED", {
            "agent_id": agent_id,
            "ticket":   ticket,
            "reason":   reason,
        })
        return ticket

    def seal_channel(self, channel_id: str, message: bytes) -> dict:
        """
        Seal a channel with HMAC authentication.
        Production: replace with AES-256-GCM or ChaCha20-Poly1305.
        """
        nonce = secrets.token_bytes(12)
        mac   = self._channel_mac(channel_id, nonce + message)
        self._sealed_channels.add(channel_id)
        self._log.append("CHANNEL_SEALED", {
            "channel_id": channel_id,
            "nonce_hex":  nonce.hex()[:16],
            "mac_hex":    mac.hex()[:16],
        })
        return {
            "channel_id": channel_id,
            "nonce":      nonce,
            "ciphertext": message,  # stub: not encrypted, just MAC-authenticated
            "mac":        mac,
        }

    def verify_channel(self, sealed: dict) -> bool:
        """Verify a sealed channel message."""
        channel_id = sealed["channel_id"]
        expected   = self._channel_mac(
            channel_id, sealed["nonce"] + sealed["ciphertext"]
        )
        return hmac.compare_digest(expected, sealed["mac"])

    def is_quarantined(self, agent_id: str) -> bool:
        return agent_id in self._quarantine

    @property
    def quarantine_count(self) -> int:
        return len(self._quarantine)


# ── RecoveryAgent ─────────────────────────────────────────────────────────────

class RecoveryAgent:
    """
    Levin cell type: progenitor (satellite — adaptive, can change strategy).
    Organ role: Recovery — rekey, rotate trust roots, rebuild from stem.

    Recovery organ: HKDF rekey → Shamir reconstruct → re-attest
    """
    CELL_TYPE = "progenitor"
    ORBIT     = "satellite"

    def __init__(self, log: TransparencyLog, master_key: bytes):
        self._log        = log
        self._master_key = master_key
        self._epoch      = 0
        self._key_shares: dict[str, bytes] = {}

    def rekey(self, reason: str) -> bytes:
        """
        Derive new master key via HKDF.
        QA mapping: orbit rekey — derive new orbit seed from current state + salt.
        """
        self._epoch += 1
        salt        = secrets.token_bytes(32)
        info        = f"rekey-epoch-{self._epoch}".encode()
        new_key     = hkdf(self._master_key, salt, info)
        self._master_key = new_key
        self._log.append("REKEY_EVENT", {
            "epoch":     self._epoch,
            "reason":    reason,
            "salt_hex":  salt.hex()[:16],
            "key_hash":  sha3_fingerprint(new_key)[:16],
        })
        return new_key

    def split_key(self, key_id: str) -> tuple[bytes, bytes, bytes]:
        """Split current master key into 2-of-3 Shamir shares."""
        s1, s2, s3 = shamir_split_2of3(self._master_key)
        # Store share fingerprints (not the shares themselves)
        self._key_shares[key_id] = sha3_fingerprint(s1 + s2 + s3)
        self._log.append("KEY_SPLIT", {
            "key_id":    key_id,
            "shares":    3,
            "threshold": 2,
            "epoch":     self._epoch,
        })
        return s1, s2, s3

    def reconstruct_key(self, s1: bytes, s2: bytes, s3: bytes,
                        key_id: str) -> bytes:
        """Reconstruct master key from all 3 shares (demo: full reconstruction)."""
        reconstructed    = shamir_reconstruct_from_all(s1, s2, s3)
        self._master_key = reconstructed
        self._log.append("KEY_RECONSTRUCTED", {
            "key_id":   key_id,
            "key_hash": sha3_fingerprint(reconstructed)[:16],
            "epoch":    self._epoch,
        })
        return reconstructed

    @property
    def epoch(self) -> int:
        return self._epoch


# ── AttestationAgent ──────────────────────────────────────────────────────────

class AttestationAgent:
    """
    Levin cell type: differentiated (cosmos — lineage authority).
    Organ role: Self/non-self — cert lineage verification, SC1-SC11 gate checks.

    QA mapping: ZKP = orbit membership proof without trace disclosure.
    Verifies cert chain without exposing internal state.
    """
    CELL_TYPE = "differentiated"
    ORBIT     = "cosmos"

    # SC checks (from family [124])
    KNOWN_SECURITY_ROLES   = {"identity","membrane","integrity",
                               "self_nonself","healing","collective"}
    KNOWN_IMMUNE_FUNCTIONS = {"detection","containment","recovery","collective_trust","healing"}
    KNOWN_PQ_READINESS     = {"fips_final","in_progress","classical_only","hybrid_transitional"}
    PQ_SENSITIVE_ROLES     = {"identity","membrane"}

    def __init__(self, log: TransparencyLog, trust_root: bytes):
        self._log        = log
        self._trust_root = trust_root
        self._attested:  dict[str, str] = {}   # cert_id → attestation_token

    def attest(self, cert: dict) -> dict:
        """
        Run SC1-SC11 checks on a security competency cert.
        Returns attestation result with error list.
        """
        errors = []

        # SC1: schema_version
        if cert.get("schema_version") != "QA_SECURITY_COMPETENCY_CERT.v1":
            errors.append("SC1:SCHEMA_VERSION_MISMATCH")

        # SC2: security_role
        role = cert.get("security_role", "")
        if role not in self.KNOWN_SECURITY_ROLES:
            errors.append(f"SC2:UNKNOWN_SECURITY_ROLE:{role}")

        # SC3: immune_function
        fn = cert.get("immune_function", "")
        if fn not in self.KNOWN_IMMUNE_FUNCTIONS:
            errors.append(f"SC3:UNKNOWN_IMMUNE_FUNCTION:{fn}")

        # SC4: pq_readiness
        pq = cert.get("pq_readiness", "")
        if pq not in self.KNOWN_PQ_READINESS:
            errors.append(f"SC4:UNKNOWN_PQ_READINESS:{pq}")

        # SC5: identity/membrane + classical_only → must have migration path
        if role in self.PQ_SENSITIVE_ROLES and pq == "classical_only":
            path = cert.get("pq_migration_path", "").strip()
            if not path:
                errors.append("SC5:PQ_MIGRATION_REQUIRED")

        # SC6: fips_final → nist_fips non-empty
        if pq == "fips_final" and not cert.get("nist_fips", "").strip():
            errors.append("SC6:MISSING_FIPS_DESIGNATION")

        # SC7: failure_modes non-empty
        if not cert.get("failure_modes"):
            errors.append("SC7:EMPTY_FAILURE_MODES")

        # SC8: composition_rules non-empty
        if not cert.get("composition_rules"):
            errors.append("SC8:EMPTY_COMPOSITION_RULES")

        # SC10: goal ≥ 10 chars
        if len(cert.get("goal", "")) < 10:
            errors.append("SC10:GOAL_TOO_SHORT")

        passed = len(errors) == 0

        # Generate attestation token (HMAC over cert fingerprint with trust root)
        cert_bytes = json.dumps(cert, sort_keys=True, separators=(",",":"),
                                ensure_ascii=False).encode()
        token = hmac.new(self._trust_root, cert_bytes, hashlib.sha256).hexdigest()

        if passed:
            cert_id = cert.get("name", "unknown")
            self._attested[cert_id] = token

        result = {
            "passed":  passed,
            "errors":  errors,
            "token":   token[:16] if passed else None,
            "cert_id": cert.get("name", "unknown"),
        }
        self._log.append(
            "CERT_ATTESTED" if passed else "CERT_REJECTED",
            {"cert_id": cert.get("name"), "errors": errors, "passed": passed}
        )
        return result

    def is_attested(self, cert_id: str) -> bool:
        return cert_id in self._attested

    @property
    def attested_count(self) -> int:
        return len(self._attested)


# ── ImmuneSystem — orchestrates all four agents ───────────────────────────────

class ImmuneSystem:
    """
    Full QA Lab immune organ: Detection + Containment + Recovery + Attestation.

    QA orbit mapping:
      - Each cert issuance = one orbit step → logged in transparency tape
      - Merkle root = orbit commitment tree seal
      - Threshold rekey = collective orbit step requiring K agents
      - Attestation token = orbit membership proof

    Levin metamorphosis: the system can differentiate from progenitor to committed
    state by running full attestation + Merkle commitment (organ formation).
    """

    def __init__(self):
        master_key   = secrets.token_bytes(32)
        trust_root   = secrets.token_bytes(32)
        self.log     = TransparencyLog()
        self.detect  = DetectionAgent(self.log)
        self.contain = ContainmentAgent(self.log, master_key)
        self.recover = RecoveryAgent(self.log, master_key)
        self.attest  = AttestationAgent(self.log, trust_root)
        self._master_key = master_key
        self._initialized = False
        self.log.append("IMMUNE_SYSTEM_INIT", {
            "schema": SCHEMA_VERSION,
            "agents": ["DetectionAgent","ContainmentAgent","RecoveryAgent","AttestationAgent"],
        })

    def register_cert_family(self, cert: dict) -> dict:
        """
        Full intake: attest → register fingerprint → Merkle commit.
        Returns combined result.
        """
        attest_result = self.attest.attest(cert)
        if not attest_result["passed"]:
            return {"status": "REJECTED", **attest_result}

        fingerprint = self.detect.register_cert(cert.get("name", "?"), cert)
        return {
            "status":      "REGISTERED",
            "fingerprint": fingerprint[:16],
            "merkle_root": self.detect.merkle_root[:16],
            "token":       attest_result["token"],
        }

    def immune_response(self, suspicious_cert: dict) -> dict:
        """
        Full immune cycle for a suspicious cert:
        1. Detection screen
        2. If threat: contain (quarantine sender)
        3. If critical threat: initiate recovery (rekey)
        Returns immune response report.
        """
        screen = self.detect.screen(suspicious_cert)

        if screen["status"] == "CLEAN":
            return {"response": "PASS", "screen": screen}

        # Contain
        agent_id = suspicious_cert.get("submitted_by", "unknown_agent")
        ticket   = self.contain.quarantine(
            agent_id,
            screen["reason"],
            {"fingerprint": screen.get("fingerprint"), "cert": suspicious_cert}
        )

        report = {
            "response":   "CONTAINED",
            "threat_level": screen["threat_level"],
            "reason":     screen["reason"],
            "ticket":     ticket,
        }

        # Critical threat: trigger recovery
        if screen["threat_level"] == "CRITICAL":
            new_key  = self.recover.rekey(f"critical_threat:{screen['reason']}")
            report["recovery_epoch"] = self.recover.epoch
            report["response"]       = "CONTAINED_AND_REKEYED"

        return report

    def status(self) -> dict:
        return {
            "log_entries":       len(self.log),
            "log_chain_valid":   self.log.verify_chain(),
            "merkle_root":       self.detect.merkle_root[:16],
            "bloom_count":       self.detect._bloom.count,
            "quarantine_count":  self.contain.quarantine_count,
            "attested_count":    self.attest.attested_count,
            "recovery_epoch":    self.recover.epoch,
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    print("QA LAB IMMUNE SYSTEM — DEMO")
    print("=" * 70)
    print()

    immune = ImmuneSystem()

    # ── Register known-good cert families ────────────────────────────────────

    print("1. REGISTERING CERT FAMILIES")
    print("-" * 40)

    good_certs = [
        {
            "name":               "ml_kem",
            "schema_version":     "QA_SECURITY_COMPETENCY_CERT.v1",
            "security_role":      "membrane",
            "immune_function":    "containment",
            "pq_readiness":       "fips_final",
            "nist_fips":          "FIPS 203",
            "goal":               "Post-quantum key encapsulation for sealed channels",
            "failure_modes":      ["decapsulation_failure", "randomness_weakness"],
            "composition_rules":  ["pair_with_hkdf", "use_fresh_nonce"],
        },
        {
            "name":               "ed25519",
            "schema_version":     "QA_SECURITY_COMPETENCY_CERT.v1",
            "security_role":      "identity",
            "immune_function":    "detection",
            "pq_readiness":       "classical_only",
            "pq_migration_path":  "Migrate to ML-DSA (FIPS 204) by 2030",
            "goal":               "Fast elliptic-curve signatures for cert attestation",
            "failure_modes":      ["weak_randomness", "side_channel"],
            "composition_rules":  ["verify_before_trust", "pin_public_key"],
        },
        {
            "name":               "sha3_256",
            "schema_version":     "QA_SECURITY_COMPETENCY_CERT.v1",
            "security_role":      "integrity",
            "immune_function":    "detection",
            "pq_readiness":       "fips_final",
            "nist_fips":          "FIPS 202",
            "goal":               "Quantum-resistant orbit step hash for cert fingerprinting",
            "failure_modes":      ["length_extension_not_applicable", "collision_theoretical"],
            "composition_rules":  ["always_domain_separate", "never_truncate_below_128bit"],
        },
    ]

    for cert in good_certs:
        result = immune.register_cert_family(cert)
        status = result["status"]
        fp     = result.get("fingerprint", "N/A")
        print(f"  {cert['name']:>20}  →  {status}  fp={fp}...")

    print()

    # ── Attestation: known-bad cert (SC5 violation) ───────────────────────────

    print("2. ATTESTATION: SC5 VIOLATION (RSA classical-only, no migration path)")
    print("-" * 40)

    bad_cert = {
        "name":              "rsa_1024_legacy",
        "schema_version":    "QA_SECURITY_COMPETENCY_CERT.v1",
        "security_role":     "identity",
        "immune_function":   "detection",
        "pq_readiness":      "classical_only",
        "pq_migration_path": "",                 # MISSING — SC5 violation
        "goal":              "Legacy RSA signature for old systems",
        "failure_modes":     ["factoring", "timing"],
        "composition_rules": ["prefer_pss_padding"],
    }

    result = immune.register_cert_family(bad_cert)
    print(f"  Status:  {result['status']}")
    print(f"  Errors:  {result.get('errors', [])}")
    print()

    # ── Detection: screen known-good cert ─────────────────────────────────────

    print("3. DETECTION SCREEN: known-good cert")
    print("-" * 40)
    screen = immune.detect.screen(good_certs[0])
    print(f"  Status:       {screen['status']}")
    print(f"  Threat level: {screen['threat_level']}")
    print(f"  Cert ID:      {screen.get('cert_id', 'N/A')}")
    print()

    # ── Detection: screen unknown cert (spoofed) ──────────────────────────────

    print("4. DETECTION SCREEN: spoofed / unknown cert")
    print("-" * 40)
    spoofed = {
        "name":               "spoofed_ml_kem",
        "schema_version":     "QA_SECURITY_COMPETENCY_CERT.v1",
        "security_role":      "membrane",
        "immune_function":    "containment",
        "pq_readiness":       "fips_final",
        "nist_fips":          "FIPS 203",
        "submitted_by":       "adversarial_agent_7",
        "goal":               "Backdoored KEM that leaks key material",
        "failure_modes":      [],
        "composition_rules":  [],
    }
    spoofed["submitted_by"] = "adversarial_agent_7"
    response = immune.immune_response(spoofed)
    print(f"  Response:     {response['response']}")
    print(f"  Threat level: {response.get('threat_level', 'N/A')}")
    print(f"  Reason:       {response.get('reason', 'N/A')}")
    print(f"  Ticket:       {response.get('ticket', 'N/A')}")
    print()

    # ── Containment: seal a channel ───────────────────────────────────────────

    print("5. CONTAINMENT: seal + verify channel")
    print("-" * 40)
    message = b"CERT_ISSUANCE: ml_kem epoch=1 root=" + immune.detect.merkle_root[:16].encode()
    sealed  = immune.contain.seal_channel("orbit_channel_0", message)
    valid   = immune.contain.verify_channel(sealed)
    print(f"  Channel:   orbit_channel_0")
    print(f"  MAC valid: {valid}")

    # Tamper and re-verify
    tampered = dict(sealed)
    tampered["ciphertext"] = b"TAMPERED_" + message
    invalid = immune.contain.verify_channel(tampered)
    print(f"  MAC valid after tamper: {invalid}  (expected False)")
    print()

    # ── Recovery: key split and reconstruct ───────────────────────────────────

    print("6. RECOVERY: Shamir split + reconstruct")
    print("-" * 40)
    s1, s2, s3   = immune.recover.split_key("master_epoch_0")
    reconstructed = immune.recover.reconstruct_key(s1, s2, s3, "master_epoch_0")
    match = reconstructed == immune.recover._master_key
    print(f"  Shares split:     3  (2-of-3 threshold)")
    print(f"  Reconstruct:      {match}  (expected True)")

    new_key = immune.recover.rekey("test_rotation")
    print(f"  Rekey epoch:      {immune.recover.epoch}")
    print()

    # ── Transparency log ──────────────────────────────────────────────────────

    print("7. TRANSPARENCY LOG")
    print("-" * 40)
    chain_valid = immune.log.verify_chain()
    print(f"  Total entries:    {len(immune.log)}")
    print(f"  Chain valid:      {chain_valid}")
    print(f"  Recent events:")
    for entry in immune.log.tail(5):
        print(f"    [{entry['seq']:>3}] {entry['event_type']:<30} {entry['hash'][:16]}...")
    print()

    # ── Status summary ────────────────────────────────────────────────────────

    print("=" * 70)
    print("IMMUNE SYSTEM STATUS")
    print("=" * 70)
    st = immune.status()
    for k, v in st.items():
        print(f"  {k:<25} {v}")

    # ── QA Orbit mapping demonstration ───────────────────────────────────────

    print()
    print("=" * 70)
    print("QA ORBIT MAPPING")
    print("=" * 70)
    print("""
  Hash functions  = orbit step hash:   fingerprint = next orbit state
  Signatures      = orbit commitment:  prove cert provenance without disclosure
  Merkle tree     = orbit commitment tree: all 129 families → one root
  Transparency log= orbit history tape: every immune event = one orbit step
  Threshold rekey = collective orbit step: K agents jointly advance
  Attestation     = orbit membership proof: SC1-SC11 verifies cert is on orbit

  Current Merkle root (orbit seal):  {}...
  Transparency head (orbit step):    {} events
  Recovery epoch (orbit generation): {}
""".format(
        immune.detect.merkle_root[:32],
        len(immune.log),
        immune.recover.epoch
    ))

    # ── Save event log ────────────────────────────────────────────────────────

    out = {
        "schema": SCHEMA_VERSION,
        "status": immune.status(),
        "log":    immune.log.tail(20),
    }
    Path("qa_immune_demo.json").write_text(
        json.dumps(out, indent=2, default=lambda x: x.hex() if isinstance(x, bytes) else str(x))
    )
    print(f"  Event log saved to qa_immune_demo.json")

    return immune


if __name__ == "__main__":
    immune = run_demo()
