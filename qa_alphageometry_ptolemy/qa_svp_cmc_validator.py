#!/usr/bin/env python3
"""
qa_svp_cmc_validator.py

Strict validator for QA SVP-CMC family:
  - QA_SVP_CMC_SEMANTICS_CERT.v1
  - QA_SVP_CMC_WITNESS_PACK.v1
  - QA_SVP_CMC_COUNTEREXAMPLES_PACK.v1

Validates against SVP-CMC cause-first semantics and radionics obstruction ledger.

Usage:
    python qa_svp_cmc_validator.py --demo
    python qa_svp_cmc_validator.py --cert path/to/cert.json --ledger qa_ledger__radionics_obstructions.v1.yaml
    python qa_svp_cmc_validator.py --rehash   # Recompute and update manifest hashes
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# --- Canonical JSON ---
def canonical_json_compact(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    """SHA256 of canonical JSON representation."""
    return hashlib.sha256(canonical_json_compact(obj).encode("utf-8")).hexdigest()


# --- YAML-lite parser (stdlib only) ---
RE_OBS_ID = re.compile(
    r'^\s*-?\s*obstruction_id:\s*"([^"]+)"\s*$|'
    r"^\s*-?\s*obstruction_id:\s*'([^']+)'\s*$|"
    r'^\s*-?\s*obstruction_id:\s*([A-Za-z0-9_\-\.]+)\s*$'
)


def parse_ledger_obstruction_ids(text: str) -> Set[str]:
    """Extract obstruction_id values from YAML ledger (line-based, no deps)."""
    ids: Set[str] = set()
    for line in text.splitlines():
        m = RE_OBS_ID.match(line)
        if m:
            for i in range(1, 4):
                g = m.group(i)
                if g:
                    ids.add(g.strip())
                    break
    return ids


# --- Validation result types ---
@dataclass
class Issue:
    code: str
    message: str
    path: str = ""
    severity: str = "error"  # error | warning


@dataclass
class ValidationResult:
    ok: bool
    issues: List[Issue] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        for i in self.issues:
            prefix = "ERROR" if i.severity == "error" else "WARN"
            loc = f" @ {i.path}" if i.path else ""
            lines.append(f"  [{prefix}] {i.code}{loc}: {i.message}")
        return "\n".join(lines)


# --- Forbidden causal language patterns ---
FORBIDDEN_CAUSAL_PATTERNS = [
    (re.compile(r'\benergy\s+(caus|driv|generat|creat)', re.I), "energy as cause"),
    (re.compile(r'\bforce\s+(caus|driv|generat|creat)', re.I), "force as cause"),
    (re.compile(r'\btransmit', re.I), "transmission language"),
    (re.compile(r'\bsignal\s+(sent|transmit|propagat)', re.I), "signal transmission"),
    (re.compile(r'\binstantaneous', re.I), "instantaneous action"),
    (re.compile(r'\bimmediate\s+effect', re.I), "immediate effect"),
]


# --- Manifest helpers ---
def _manifest_hashable_copy(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return copy with manifest.canonical_json_sha256 set to placeholder."""
    out = copy.deepcopy(obj)
    if "manifest" in out and isinstance(out["manifest"], dict):
        out["manifest"]["canonical_json_sha256"] = "placeholder"
    return out


def _enforce_manifest(obj: Dict[str, Any], label: str) -> None:
    """Verify manifest hash matches canonical JSON of content."""
    manifest = obj.get("manifest", {})
    claimed = manifest.get("canonical_json_sha256", "")
    hashable = _manifest_hashable_copy(obj)
    computed = sha256_canonical(hashable)
    if claimed != computed:
        raise ValueError(f"{label}: manifest hash mismatch (claimed={claimed[:16]}..., computed={computed[:16]}...)")


def _update_manifest(obj: Dict[str, Any]) -> str:
    """Update manifest hash in place and return the new hash."""
    hashable = _manifest_hashable_copy(obj)
    computed = sha256_canonical(hashable)
    if "manifest" not in obj:
        obj["manifest"] = {"hash_alg": "sha256"}
    obj["manifest"]["canonical_json_sha256"] = computed
    return computed


# --- Core certificate validator ---
def validate_cert(
    cert: Dict[str, Any],
    ledger_obs_ids: Optional[Set[str]] = None,
) -> ValidationResult:
    """
    Validate a QA_CERT__SVP_CMC_ANALYSIS.v1 certificate.

    Levels:
      1. Schema (required fields, types)
      2. Consistency (cross-field checks)
      3. Policy (SVP-CMC cause-first constraints)
      4. Obstruction (referenced IDs exist in ledger)
    """
    issues: List[Issue] = []

    # --- Level 1: Schema checks ---
    required_top = [
        "schema_version", "cert_type", "cert_id", "subject",
        "scalar", "disturbance", "latency", "kinetics",
        "impossibles", "trace", "sha256_manifest"
    ]
    for key in required_top:
        if key not in cert:
            issues.append(Issue("MISSING_FIELD", f"Missing required field: {key}", path=key))

    if cert.get("cert_type") != "QA_CERT__SVP_CMC_ANALYSIS.v1":
        issues.append(Issue("WRONG_CERT_TYPE", f"Expected cert_type QA_CERT__SVP_CMC_ANALYSIS.v1"))

    # Subject checks
    subject = cert.get("subject", {})
    for key in ["subject_id", "subject_type", "prompt_text"]:
        if key not in subject:
            issues.append(Issue("MISSING_FIELD", f"Missing subject.{key}", path=f"subject.{key}"))

    valid_subject_types = {"device", "organism", "phenomenon", "state", "question", "protocol"}
    if subject.get("subject_type") and subject["subject_type"] not in valid_subject_types:
        issues.append(Issue("INVALID_SUBJECT_TYPE", f"Invalid subject_type: {subject['subject_type']}"))

    # Scalar checks
    scalar = cert.get("scalar", {})
    for key in ["architecture", "neutral_centers", "symmetry", "persistence"]:
        if key not in scalar:
            issues.append(Issue("MISSING_FIELD", f"Missing scalar.{key}", path=f"scalar.{key}"))

    arch = scalar.get("architecture", {})
    for key in ["geometry", "materials", "polarity", "boundary_conditions"]:
        if key not in arch:
            issues.append(Issue("MISSING_FIELD", f"Missing scalar.architecture.{key}", path=f"scalar.architecture.{key}"))

    # Disturbance checks
    dist = cert.get("disturbance", {})
    if "generator_id" not in dist:
        issues.append(Issue("MISSING_FIELD", "Missing disturbance.generator_id", path="disturbance.generator_id"))
    if "parameters" not in dist:
        issues.append(Issue("MISSING_FIELD", "Missing disturbance.parameters", path="disturbance.parameters"))
    else:
        for key in ["type", "rate", "direction", "magnitude"]:
            if key not in dist["parameters"]:
                issues.append(Issue("MISSING_FIELD", f"Missing disturbance.parameters.{key}", path=f"disturbance.parameters.{key}"))

    # Latency checks
    latency = cert.get("latency", {})
    for key in ["response_order", "lags_and_causes"]:
        if key not in latency:
            issues.append(Issue("MISSING_FIELD", f"Missing latency.{key}", path=f"latency.{key}"))

    # Kinetics checks
    kinetics = cert.get("kinetics", {})
    for key in ["permitted", "observed"]:
        if key not in kinetics:
            issues.append(Issue("MISSING_FIELD", f"Missing kinetics.{key}", path=f"kinetics.{key}"))

    # Trace checks
    trace = cert.get("trace", {})
    if "moves" not in trace:
        issues.append(Issue("MISSING_FIELD", "Missing trace.moves", path="trace.moves"))
    else:
        for i, move in enumerate(trace.get("moves", [])):
            if "move" not in move:
                issues.append(Issue("MISSING_TRACE_KEY", f"Move {i} missing 'move'", path=f"trace.moves[{i}].move"))
            if "invariant_diff" not in move:
                issues.append(Issue("MISSING_TRACE_KEY", f"Move {i} missing 'invariant_diff'", path=f"trace.moves[{i}].invariant_diff"))

    # --- Level 2: Consistency checks ---
    permitted = set(kinetics.get("permitted", []))
    observed = set(kinetics.get("observed", []))
    if not observed.issubset(permitted):
        extra = observed - permitted
        issues.append(Issue("KINETICS_MISMATCH", f"Observed kinetics not in permitted: {extra}", path="kinetics"))

    # --- Level 3: Policy checks (SVP-CMC cause-first) ---
    def scan_for_forbidden(obj: Any, path: str = ""):
        if isinstance(obj, str):
            for pattern, desc in FORBIDDEN_CAUSAL_PATTERNS:
                if pattern.search(obj):
                    issues.append(Issue("SVP_POLICY_VIOLATION", f"Forbidden causal language ({desc}): '{obj[:50]}...'", path=path))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                scan_for_forbidden(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan_for_forbidden(v, f"{path}[{i}]")

    scan_for_forbidden(cert)

    # --- Level 4: Obstruction ledger checks ---
    impossibles = cert.get("impossibles", [])
    if ledger_obs_ids is not None:
        for i, imp in enumerate(impossibles):
            obs_id = imp.get("obstruction_id", "")
            if obs_id and obs_id not in ledger_obs_ids:
                issues.append(Issue("UNKNOWN_OBSTRUCTION_ID", f"obstruction_id '{obs_id}' not in ledger", path=f"impossibles[{i}].obstruction_id"))

    errors = [i for i in issues if i.severity == "error"]
    return ValidationResult(ok=len(errors) == 0, issues=issues)


# --- Demo certificate ---
def make_demo_cert() -> Dict[str, Any]:
    """Generate a minimal valid demo certificate."""
    cert = {
        "schema_version": "qa_cert.v1",
        "cert_type": "QA_CERT__SVP_CMC_ANALYSIS.v1",
        "cert_id": "DEMO_SVP_CMC_001",
        "subject": {
            "subject_id": "tuning_fork_resonance",
            "subject_type": "device",
            "prompt_text": "Analyze sympathetic resonance between two tuning forks."
        },
        "scalar": {
            "architecture": {
                "geometry": "two_forks_parallel_5cm",
                "materials": "steel_440Hz",
                "polarity": "neutral",
                "boundary_conditions": "open_air_room"
            },
            "neutral_centers": ["fork_1_base", "fork_2_base"],
            "symmetry": "bilateral_mirror",
            "persistence": {
                "resists": ["frequency_shift", "damping_increase"],
                "invariants": ["fundamental_frequency", "phase_lock"]
            },
            "parameters": {
                "subdivision_frequency_band": "audible_400-450Hz",
                "phase_relationships": "locked_0deg",
                "amplitude_degree": "medium"
            }
        },
        "disturbance": {
            "generator_id": "strike_fork_1",
            "parameters": {
                "type": "impulse",
                "rate": "single_strike",
                "direction": "perpendicular_to_tine",
                "magnitude": "standard_mallet"
            }
        },
        "latency": {
            "response_order": ["fork_1_oscillation", "air_coupling", "fork_2_oscillation"],
            "lags_and_causes": [
                "fork_1 responds immediately to strike",
                "air_coupling requires ~15ms for wavefront propagation",
                "fork_2 builds amplitude over ~200ms via sympathetic resonance"
            ],
            "path_metrics": {
                "min_steps": 3,
                "return_in_k": 0
            }
        },
        "kinetics": {
            "permitted": ["audible_tone", "mechanical_oscillation", "air_pressure_wave"],
            "observed": ["audible_tone", "mechanical_oscillation"]
        },
        "impossibles": [
            {
                "effect": "electromagnetic_emission",
                "obstruction_id": "RADIONICS_OBS__WRONG_SUBDIVISION_TARGETED__v1",
                "reason": "Subdivision band is audible/mechanical; EM emission requires different configuration."
            }
        ],
        "trace": {
            "moves": [
                {
                    "move": "strike_fork_1",
                    "invariant_diff": {
                        "before": {"fork_1_state": "rest", "fork_2_state": "rest"},
                        "after": {"fork_1_state": "oscillating", "fork_2_state": "rest"}
                    }
                },
                {
                    "move": "sympathetic_coupling",
                    "invariant_diff": {
                        "before": {"fork_2_state": "rest"},
                        "after": {"fork_2_state": "oscillating"}
                    }
                }
            ]
        },
        "sha256_manifest": ""
    }
    cert_json = json.dumps(cert, sort_keys=True, separators=(',', ':'))
    cert["sha256_manifest"] = hashlib.sha256(cert_json.encode()).hexdigest()
    return cert


# --- Triplet validation ---
def validate_all(
    semantics_path: str,
    witness_path: str,
    counterexamples_path: str,
    ledger_path: Optional[str] = None,
) -> None:
    """
    Validate the full SVP-CMC triplet (semantics + witness + counterexamples).
    Raises on validation failure.
    """
    base_dir = Path(__file__).parent

    # Load ledger
    if ledger_path is None:
        ledger_path = str(base_dir / "qa_ledger__radionics_obstructions.v1.yaml")
    ledger_text = Path(ledger_path).read_text(encoding="utf-8")
    ledger_ids = parse_ledger_obstruction_ids(ledger_text)
    if not ledger_ids:
        raise ValueError("Ledger contains no obstruction IDs")

    # Load and validate semantics
    with open(semantics_path, "r", encoding="utf-8") as f:
        semantics = json.load(f)
    _enforce_manifest(semantics, "semantics")
    if semantics.get("schema_id") != "QA_SVP_CMC_SEMANTICS_CERT.v1":
        raise ValueError(f"Unexpected semantics schema_id: {semantics.get('schema_id')}")

    # Note: fail_types in semantics cert should align with ledger obstruction classes
    # We don't enforce strict 1:1 mapping - semantics may define additional fail types
    sem_fail_types = set(semantics.get("fail_types", []))

    # Load and validate witness
    with open(witness_path, "r", encoding="utf-8") as f:
        witness = json.load(f)
    _enforce_manifest(witness, "witness")
    if witness.get("schema_id") != "QA_SVP_CMC_WITNESS_PACK.v1":
        raise ValueError(f"Unexpected witness schema_id: {witness.get('schema_id')}")
    if not witness.get("claimed_success"):
        raise ValueError("Witness pack must have claimed_success=true")
    if not witness.get("ledger_sanity_passed"):
        raise ValueError("Witness pack ledger_sanity_passed must be true")
    if not witness.get("validator_demo_passed"):
        raise ValueError("Witness pack validator_demo_passed must be true")

    # Validate the embedded demo cert
    demo_cert = witness.get("demo_cert", {})
    res = validate_cert(demo_cert, ledger_obs_ids=ledger_ids)
    if not res.ok:
        raise ValueError(f"Witness demo cert validation failed:\n{res.summary()}")

    # Load and validate counterexamples
    with open(counterexamples_path, "r", encoding="utf-8") as f:
        counterexamples = json.load(f)
    _enforce_manifest(counterexamples, "counterexamples")
    if counterexamples.get("schema_id") != "QA_SVP_CMC_COUNTEREXAMPLES_PACK.v1":
        raise ValueError(f"Unexpected counterexamples schema_id: {counterexamples.get('schema_id')}")

    # Validate each counterexample case
    cases = counterexamples.get("cases", [])
    if len(cases) == 0:
        raise ValueError("Counterexamples pack must have at least one case")

    for case in cases:
        case_id = case.get("case_id")
        expected_fail = case.get("expected_fail_type")
        if not case_id or not expected_fail:
            raise ValueError(f"Counterexample case missing case_id or expected_fail_type")
        if expected_fail not in sem_fail_types:
            raise ValueError(f"Case {case_id}: expected_fail_type '{expected_fail}' not in semantics fail_types")

    print(f"[SVP-CMC] Validated triplet: semantics + witness + {len(cases)} counterexamples")


# --- Rehash function ---
def rehash_all(base_dir: Path) -> None:
    """Recompute and update manifest hashes for all SVP-CMC family certs."""
    paths = [
        base_dir / "certs" / "QA_SVP_CMC_SEMANTICS_CERT.v1.json",
        base_dir / "certs" / "witness" / "QA_SVP_CMC_WITNESS_PACK.v1.json",
        base_dir / "certs" / "counterexamples" / "QA_SVP_CMC_COUNTEREXAMPLES_PACK.v1.json",
    ]

    for p in paths:
        if not p.exists():
            print(f"[SKIP] {p.name} not found")
            continue

        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

        new_hash = _update_manifest(obj)

        with open(p, "w", encoding="utf-8") as f:
            f.write(canonical_json_compact(obj))
            f.write("\n")

        print(f"[REHASH] {p.name}: {new_hash[:16]}...")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate SVP-CMC family certificates")
    ap.add_argument("--cert", help="Path to certificate JSON")
    ap.add_argument("--ledger", help="Path to obstruction ledger YAML")
    ap.add_argument("--demo", action="store_true", help="Run demo validation")
    ap.add_argument("--rehash", action="store_true", help="Recompute manifest hashes")
    ap.add_argument("--validate-triplet", action="store_true", help="Validate full triplet")
    args = ap.parse_args(argv)

    base_dir = Path(__file__).parent

    if args.rehash:
        print("[SVP-CMC] Rehashing manifest hashes...")
        rehash_all(base_dir)
        return 0

    if args.validate_triplet:
        print("[SVP-CMC] Validating full triplet...")
        try:
            validate_all(
                semantics_path=str(base_dir / "certs" / "QA_SVP_CMC_SEMANTICS_CERT.v1.json"),
                witness_path=str(base_dir / "certs" / "witness" / "QA_SVP_CMC_WITNESS_PACK.v1.json"),
                counterexamples_path=str(base_dir / "certs" / "counterexamples" / "QA_SVP_CMC_COUNTEREXAMPLES_PACK.v1.json"),
            )
            print("[SVP-CMC] PASS: Full triplet validation")
            return 0
        except Exception as e:
            print(f"[SVP-CMC] FAIL: {e}")
            return 1

    if args.demo:
        print("[SVP-CMC] Running demo validation...")
        cert = make_demo_cert()
        default_ledger = base_dir / "qa_ledger__radionics_obstructions.v1.yaml"
        ledger_ids = None
        if default_ledger.exists():
            ledger_ids = parse_ledger_obstruction_ids(default_ledger.read_text())
            print(f"[SVP-CMC] Loaded {len(ledger_ids)} obstruction IDs from ledger")

        res = validate_cert(cert, ledger_obs_ids=ledger_ids)
        if res.ok:
            print("[SVP-CMC] PASS: Demo certificate valid")
            print(f"[SVP-CMC] cert_id: {cert['cert_id']}")
            print(f"[SVP-CMC] subject: {cert['subject']['subject_id']}")
            return 0
        else:
            print("[SVP-CMC] FAIL: Demo certificate invalid")
            print(res.summary())
            return 1

    if not args.cert:
        print("ERROR: --cert required (or use --demo, --rehash, --validate-triplet)")
        return 2

    try:
        with open(args.cert, "r", encoding="utf-8") as f:
            cert = json.load(f)
    except Exception as e:
        print(f"ERROR: Cannot load certificate: {e}")
        return 2

    ledger_ids = None
    if args.ledger:
        try:
            with open(args.ledger, "r", encoding="utf-8") as f:
                ledger_ids = parse_ledger_obstruction_ids(f.read())
            print(f"[SVP-CMC] Loaded {len(ledger_ids)} obstruction IDs from ledger")
        except Exception as e:
            print(f"ERROR: Cannot load ledger: {e}")
            return 2

    res = validate_cert(cert, ledger_obs_ids=ledger_ids)
    if res.ok:
        print(f"[SVP-CMC] PASS: {args.cert}")
        return 0
    else:
        print(f"[SVP-CMC] FAIL: {args.cert}")
        print(res.summary())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
