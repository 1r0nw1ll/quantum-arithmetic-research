#!/usr/bin/env python3
"""
qa_competency_intake_runner.py

CLI entry point for the QA Competency intake pipeline.

Pipeline: RAW (jsonl) -> PARSED TRACE -> GRAPH + METRICS -> CERT -> VALIDATE -> HASH

Usage:
    python qa_competency/intake/qa_competency_intake_runner.py \
        --input logs/agent.jsonl \
        --adapter llm_tool_agent \
        --out out/

Produces:
    out/<stem>.bundle.json    (QA_COMPETENCY_CERT_BUNDLE.v1)
    out/<stem>.intake.json    (QA_COMPETENCY_INTAKE.v0)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical JSON helpers (self-contained)
# ---------------------------------------------------------------------------

def canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    return hashlib.sha256(
        canonical_json_compact(obj).encode("utf-8")
    ).hexdigest()


HEX64_ZERO = "0" * 64


# ---------------------------------------------------------------------------
# Import adapter + validator with try/except for flexibility
# ---------------------------------------------------------------------------

def _import_validator():
    """Import validate_bundle and _update_manifest from qa_competency_validator."""
    try:
        from qa_competency.qa_competency_validator import (
            validate_bundle, _update_manifest,
        )
        return validate_bundle, _update_manifest
    except ImportError:
        # Fall back to direct import when running from qa_competency dir
        parent = str(Path(__file__).resolve().parent.parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from qa_competency.qa_competency_validator import (
            validate_bundle, _update_manifest,
        )
        return validate_bundle, _update_manifest


def _import_adapter(name: str):
    """Import the named adapter module and return its adapt() function."""
    if name == "llm_tool_agent":
        try:
            from qa_competency.intake.adapters.llm_tool_agent import adapt, IntakeError
            return adapt, IntakeError
        except ImportError:
            parent = str(Path(__file__).resolve().parent.parent.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            from qa_competency.intake.adapters.llm_tool_agent import adapt, IntakeError
            return adapt, IntakeError
    else:
        raise ValueError(f"Unknown adapter: {name}")


# ---------------------------------------------------------------------------
# Manifest helper (mirrors validator's _update_manifest for intake records)
# ---------------------------------------------------------------------------

def _update_manifest_local(obj: Dict[str, Any]) -> str:
    """Update manifest hash in place and return the new hash."""
    import copy
    if "manifest" not in obj:
        obj["manifest"] = {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        }
    out = copy.deepcopy(obj)
    out["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    computed = sha256_canonical(out)
    obj["manifest"]["canonical_json_sha256"] = computed
    return computed


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_intake(
    input_path: str,
    adapter_name: str,
    out_dir: str,
    *,
    domain: str = "software_engineering",
    substrate: str = "llm_agent",
    description: str = "LLM tool agent competency intake",
) -> Dict[str, str]:
    """Execute the intake pipeline.

    Returns:
        Dict with 'bundle_path' and 'intake_path' keys.

    Raises:
        Various IntakeError or RuntimeError on failure.
    """
    adapt_fn, IntakeError = _import_adapter(adapter_name)
    validate_bundle, _update_manifest = _import_validator()

    input_p = Path(input_path)
    if not input_p.is_file():
        raise IntakeError(
            fail_type="INGEST_INPUT_NOT_FOUND",
            invariant_diff={"path": str(input_p)},
        )

    # ---- 1. Read + hash raw input file --------------------------------
    raw_bytes = input_p.read_bytes()
    input_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # ---- 2. Parse JSONL -----------------------------------------------
    events: List[dict] = []
    for line_no, line in enumerate(raw_bytes.decode("utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            raise IntakeError(
                fail_type="INGEST_PARSE_ERROR",
                invariant_diff={"line": line_no, "error": str(e)},
            )
        events.append(ev)

    # ---- 3. Call adapter to produce cert dict -------------------------
    cert = adapt_fn(
        events,
        domain=domain,
        substrate=substrate,
        description=description,
    )

    # ---- 4. Wrap cert in bundle, set manifests ------------------------
    _update_manifest(cert)

    bundle: Dict[str, Any] = {
        "schema_id": "QA_COMPETENCY_CERT_BUNDLE.v1",
        "manifest": {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
        "certs": [cert],
    }
    _update_manifest(bundle)

    # ---- 5. Validate produced bundle ----------------------------------
    result = validate_bundle(bundle)
    if not result.ok:
        raise IntakeError(
            fail_type="CERT_VALIDATION_FAILED",
            invariant_diff={
                "validator_fail_type": result.fail_type,
                "details": result.invariant_diff,
            },
        )

    # ---- 6. Write bundle + intake record ------------------------------
    os.makedirs(out_dir, exist_ok=True)
    stem = input_p.stem

    bundle_path = Path(out_dir) / f"{stem}.bundle.json"
    bundle_bytes = canonical_json_compact(bundle).encode("utf-8")
    bundle_path.write_bytes(bundle_bytes + b"\n")
    output_bundle_sha256 = hashlib.sha256(bundle_bytes + b"\n").hexdigest()

    intake_record: Dict[str, Any] = {
        "schema_id": "QA_COMPETENCY_INTAKE.v0",
        "input_file": str(input_p.name),
        "input_sha256": input_sha256,
        "adapter": adapter_name,
        "event_count": len(events),
        "output_bundle_sha256": output_bundle_sha256,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }
    _update_manifest_local(intake_record)

    intake_path = Path(out_dir) / f"{stem}.intake.json"
    intake_bytes = canonical_json_compact(intake_record).encode("utf-8")
    intake_path.write_bytes(intake_bytes + b"\n")

    return {
        "bundle_path": str(bundle_path),
        "intake_path": str(intake_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="QA Competency Intake Pipeline v0",
    )
    ap.add_argument("--input", required=True, help="Path to JSONL input file")
    ap.add_argument("--adapter", default="llm_tool_agent",
                    help="Adapter name (default: llm_tool_agent)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--domain", default="software_engineering",
                    help="Domain label (default: software_engineering)")
    ap.add_argument("--substrate", default="llm_agent",
                    help="Substrate label (default: llm_agent)")
    ap.add_argument("--description", default="LLM tool agent competency intake",
                    help="Description for cert metadata")
    args = ap.parse_args(argv)

    try:
        result = run_intake(
            input_path=args.input,
            adapter_name=args.adapter,
            out_dir=args.out,
            domain=args.domain,
            substrate=args.substrate,
            description=args.description,
        )
        print(f"[Intake] Bundle:  {result['bundle_path']}")
        print(f"[Intake] Intake:  {result['intake_path']}")

        # Print summary from bundle
        bundle = json.loads(Path(result["bundle_path"]).read_text("utf-8"))
        cert = bundle["certs"][0]
        mi = cert["metric_inputs"]
        cm = cert["competency_metrics"]
        print(f"[Intake] Events:  {mi['reachable_states']} states, "
              f"{mi['total_states']} total")
        print(f"[Intake] Metrics: agency={cm['agency_index']:.4f} "
              f"plasticity={cm['plasticity_index']:.4f} "
              f"goal_density={cm['goal_density']:.4f} "
              f"entropy={cm['control_entropy']:.4f}")
        print(f"[Intake] Generators: {[g['id'] for g in cert['generators']]}")
        print("[Intake] OK")
        return 0

    except Exception as e:
        print(f"[Intake] FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
