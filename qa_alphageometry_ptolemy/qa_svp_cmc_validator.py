#!/usr/bin/env python3
"""
qa_svp_cmc_validator.py

Strict validator for QA_CERT__SVP_CMC_ANALYSIS.v1 certificates.
Validates against SVP-CMC cause-first semantics and radionics obstruction ledger.

Usage:
    python qa_svp_cmc_validator.py --cert path/to/cert.json --ledger qa_ledger__radionics_obstructions.v1.yaml
    python qa_svp_cmc_validator.py --demo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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


# --- Core validator ---
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
    # Observed kinetics must be subset of permitted
    permitted = set(kinetics.get("permitted", []))
    observed = set(kinetics.get("observed", []))
    if not observed.issubset(permitted):
        extra = observed - permitted
        issues.append(Issue("KINETICS_MISMATCH", f"Observed kinetics not in permitted: {extra}", path="kinetics"))

    # --- Level 3: Policy checks (SVP-CMC cause-first) ---
    # Scan all string fields for forbidden causal language
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

    # Determine overall result
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
    # Compute manifest hash
    cert_json = json.dumps(cert, sort_keys=True, separators=(',', ':'))
    cert["sha256_manifest"] = hashlib.sha256(cert_json.encode()).hexdigest()
    return cert


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate SVP-CMC analysis certificates")
    ap.add_argument("--cert", help="Path to certificate JSON")
    ap.add_argument("--ledger", help="Path to obstruction ledger YAML")
    ap.add_argument("--demo", action="store_true", help="Run demo validation")
    args = ap.parse_args(argv)

    if args.demo:
        print("[SVP-CMC] Running demo validation...")
        cert = make_demo_cert()
        # Load default ledger if exists
        default_ledger = Path(__file__).parent / "qa_ledger__radionics_obstructions.v1.yaml"
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
        print("ERROR: --cert required (or use --demo)")
        return 2

    # Load certificate
    try:
        with open(args.cert, "r", encoding="utf-8") as f:
            cert = json.load(f)
    except Exception as e:
        print(f"ERROR: Cannot load certificate: {e}")
        return 2

    # Load ledger if provided
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
