#!/usr/bin/env python3
"""
qa_formal_conjecture_resolution_cert_validate.py  [family 248]

QA_COMPLIANCE: {
    'signal_injection': 'none (cert record validation only)',
    'dynamics': 'JSON schema + semantic rule checks; no QA state evolution',
    'float_state': false,
    'observer_projection': 'none — cert records are discrete metadata',
    'time': 'none (static validation; T1 N/A)'
}

Validates QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1 records per the design
spec at docs/theory/QA_AUTOMATED_CONJECTURE_RESOLUTION.md §6. Primary
source: Ju, Gao, Jiang, Wu, Sun, Chen, Wang, Wang, Wang, He, Wu, Xiao,
Liu, Dai, Dong (2026). "Automated Conjecture Resolution with Formal
Verification." arXiv:2604.03789 (https://arxiv.org/abs/2604.03789).

Seven gates FCR_1..FCR_7 enforce schema integrity, typed failure labels,
generator-set declaration, NT compliance, verdict vocabulary, witness
path, and Lean-stub open-question acknowledgment. See theory doc for the
operator mapping and the paper's Rethlas/Archon/Lean-4 pipeline.
"""

QA_COMPLIANCE = (
    "cert_validator - formal conjecture resolution records; "
    "no QA state, int/str only, no pow, no float state, no stochastic inputs"
)

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1"
FAMILY_NAME = "qa_formal_conjecture_resolution_cert"

ALLOWED_CANDIDATE_KIND = {
    "orbit_invariant",
    "diagonal_class",
    "generator_chain",
    "bridge",
    "external",
}
ALLOWED_FORMAL_BACKEND = {
    "qa_symbolic",
    "qa_exhaustive",
    "lean4_stub",
    "python_proof",
}
ALLOWED_PROOF_STATUS = {
    "proved",
    "formal_gap",
    "qa_obstruction",
    "generator_insufficient",
    "inconclusive",
}
ALLOWED_VERDICT = {"CONSISTENT", "PARTIAL", "CONTRADICTS", "INCONCLUSIVE"}

_REQUIRED_STRING_FIELDS = (
    "conjecture_id",
    "source_claim",
    "qa_translation",
    "candidate_kind",
    "formal_backend",
    "proof_status",
    "witness_path",
    "verdict",
)


def _is_nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


def check_fcr_1_schema(record):
    if record.get("schema_version") != SCHEMA_VERSION:
        return (
            "FCR_1: schema_version must be "
            f"{SCHEMA_VERSION!r}, got {record.get('schema_version')!r}"
        )
    if record.get("family") != FAMILY_NAME:
        return (
            "FCR_1: family must be "
            f"{FAMILY_NAME!r}, got {record.get('family')!r}"
        )
    for field in _REQUIRED_STRING_FIELDS:
        if not _is_nonempty_str(record.get(field)):
            return f"FCR_1: field {field!r} missing or not a non-empty string"
    if record["candidate_kind"] not in ALLOWED_CANDIDATE_KIND:
        return (
            "FCR_1: candidate_kind must be one of "
            f"{sorted(ALLOWED_CANDIDATE_KIND)}"
        )
    if record["formal_backend"] not in ALLOWED_FORMAL_BACKEND:
        return (
            "FCR_1: formal_backend must be one of "
            f"{sorted(ALLOWED_FORMAL_BACKEND)}"
        )
    if record["proof_status"] not in ALLOWED_PROOF_STATUS:
        return (
            "FCR_1: proof_status must be one of "
            f"{sorted(ALLOWED_PROOF_STATUS)}"
        )
    return None


def check_fcr_2_generator_set(record):
    gen = record.get("generator_set")
    if not isinstance(gen, list) or not gen:
        return "FCR_2: generator_set must be a non-empty list"
    for i, g in enumerate(gen):
        if not _is_nonempty_str(g):
            return f"FCR_2: generator_set[{i}] must be a non-empty string"
    return None


def check_fcr_3_failure_mode(record):
    status = record.get("proof_status")
    failure_mode = record.get("failure_mode")
    if status == "proved":
        if failure_mode is not None:
            return "FCR_3: failure_mode must be null when proof_status=proved"
        return None
    if not _is_nonempty_str(failure_mode):
        return (
            "FCR_3: failure_mode must be a non-empty string when "
            f"proof_status={status!r}"
        )
    return None


def check_fcr_4_nt_compliance(record):
    nt = record.get("nt_compliance")
    if not isinstance(nt, bool):
        return "FCR_4: nt_compliance must be a boolean"
    if nt is False:
        failure_mode = record.get("failure_mode") or ""
        lower = failure_mode.lower()
        if "nt" not in lower and "firewall" not in lower and "observer projection" not in lower:
            return (
                "FCR_4: if nt_compliance=false, failure_mode must name the NT "
                "violation (Theorem NT / observer projection / firewall)"
            )
    return None


def check_fcr_5_verdict(record):
    if record.get("verdict") not in ALLOWED_VERDICT:
        return f"FCR_5: verdict must be one of {sorted(ALLOWED_VERDICT)}"
    return None


def check_fcr_6_witness(record, base_dir):
    witness_path = record.get("witness_path")
    if not _is_nonempty_str(witness_path):
        return "FCR_6: witness_path must be a non-empty string"
    if witness_path == "symbolic_inline":
        return None
    candidate = Path(base_dir) / witness_path
    if not candidate.exists():
        return (
            "FCR_6: witness_path "
            f"{witness_path!r} does not exist and is not 'symbolic_inline'"
        )
    return None


def check_fcr_7_lean4_stub(record):
    if record.get("formal_backend") != "lean4_stub":
        return None
    open_questions = record.get("open_questions")
    if not isinstance(open_questions, list) or not open_questions:
        return (
            "FCR_7: formal_backend=lean4_stub requires non-empty "
            "open_questions array naming the Lean gap"
        )
    for i, question in enumerate(open_questions):
        if not _is_nonempty_str(question):
            return f"FCR_7: open_questions[{i}] must be a non-empty string"
    return None


def validate(record, base_dir):
    errors = []
    for check in (
        check_fcr_1_schema,
        check_fcr_2_generator_set,
        check_fcr_3_failure_mode,
        check_fcr_4_nt_compliance,
        check_fcr_5_verdict,
        check_fcr_7_lean4_stub,
    ):
        err = check(record)
        if err:
            errors.append(err)
    err = check_fcr_6_witness(record, base_dir)
    if err:
        errors.append(err)
    return errors


def self_test():
    here = Path(__file__).resolve().parent
    fixtures_dir = here / "fixtures"
    pass_files = sorted(fixtures_dir.glob("fcr_pass_*.json"))
    fail_files = sorted(fixtures_dir.glob("fcr_fail_*.json"))
    if not pass_files:
        return {"ok": False, "reason": "no fcr_pass_*.json fixtures"}
    if not fail_files:
        return {"ok": False, "reason": "no fcr_fail_*.json fixtures"}
    pass_results = []
    for path in pass_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        errors = validate(record, fixtures_dir)
        if errors:
            return {"ok": False, "fixture": path.name, "errors": errors}
        pass_results.append(
            {
                "name": path.name,
                "verdict": record["verdict"],
                "status": record["proof_status"],
            }
        )
    fail_results = []
    for path in fail_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        errors = validate(record, fixtures_dir)
        if not errors:
            return {
                "ok": False,
                "fixture": path.name,
                "reason": "expected rejection, got clean pass",
            }
        fail_results.append({"name": path.name, "errors": errors})
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY_NAME,
        "pass_count": len(pass_results),
        "fail_count": len(fail_results),
        "pass_fixtures": pass_results,
        "fail_fixtures": fail_results,
    }


def main(argv):
    if len(argv) >= 2 and argv[1] == "--self-test":
        result = self_test()
        print(
            json.dumps(
                result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        )
        return 0 if result.get("ok") else 1
    print(
        json.dumps(
            {"ok": False, "reason": "pass --self-test"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
