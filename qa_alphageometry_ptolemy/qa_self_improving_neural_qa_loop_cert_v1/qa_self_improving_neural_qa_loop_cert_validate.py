#!/usr/bin/env python3
"""Validator for QA_SELF_IMPROVING_NEURAL_QA_LOOP_CERT.v1."""

from __future__ import annotations

QA_COMPLIANCE = (
    "cert_validator — validates multi-round self-improving neural QA loop "
    "transcripts; rejected rounds cannot mutate promoted state"
)

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def load_tool_function(tool_name: str, function_name: str) -> Any:
    path = TOOLS / tool_name
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": f"_loaded_{path.stem}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace[function_name]


def validate_transcript_file(path: Path) -> dict[str, Any]:
    validate_transcript = load_tool_function(
        "qa_self_improving_neural_qa_transcript_validate.py",
        "validate_transcript",
    )
    return validate_transcript(path)


def self_test() -> dict[str, Any]:
    fixtures = Path(__file__).resolve().parent / "fixtures"
    cases = [
        ("pass_live_transcript.jsonl", True),
        ("fail_tampered_rejected_mutates.jsonl", False),
    ]
    results = []
    ok = True
    for name, should_pass in cases:
        result = validate_transcript_file(fixtures / name)
        passed = bool(result["ok"])
        case_ok = passed is should_pass
        ok = ok and case_ok
        results.append({"fixture": name, "ok": case_ok, "passed": passed, "result": result})
    pass_result = validate_transcript_file(fixtures / "pass_live_transcript.jsonl")
    if pass_result.get("accepted", 0) < 1 or pass_result.get("rejected", 0) < 1:
        ok = False
        results.append({
            "fixture": "pass_live_transcript.jsonl",
            "ok": False,
            "error": "PASS fixture must include accepted and rejected rounds",
        })
    return {"ok": ok, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1

    paths = args.paths or sorted((Path(__file__).resolve().parent / "fixtures").glob("*.jsonl"))
    all_ok = True
    for path in paths:
        result = validate_transcript_file(path)
        all_ok = all_ok and bool(result["ok"])
        print(json.dumps({"path": str(path), **result}, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
