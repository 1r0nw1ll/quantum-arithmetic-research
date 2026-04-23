#!/usr/bin/env python3
# noqa: DECL-1 (eval scaffold — not empirical QA code)
"""
Small runner for the blind TLA+/formal-methods eval harness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SCORECARD_SCHEMA = ROOT / "rubrics" / "scorecard_schema.json"
VISIBLE_FILES = {
    "generation": ("prompt.md",),
    "review": ("case.md",),
    "repair": ("case.md",),
}
CASE_ROOTS = {
    "generation": ROOT / "tasks" / "generation",
    "review": ROOT / "review_corpus",
    "repair": ROOT / "repair_cases",
}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _case_dir(layer: str, case_id: str) -> Path:
    return CASE_ROOTS[layer] / case_id


def _list_cases() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for layer, root in CASE_ROOTS.items():
        if not root.exists():
            continue
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest_name = "task.json" if layer == "generation" else "case.json"
            manifest = _load_json(case_dir / manifest_name)
            rows.append(
                {
                    "layer": layer,
                    "case_id": case_dir.name,
                    "title": str(manifest.get("title", "")),
                    "difficulty": str(manifest.get("difficulty", "")),
                }
            )
    return rows


def _show_visible_prompt(layer: str, case_id: str) -> str:
    case_dir = _case_dir(layer, case_id)
    parts: list[str] = []
    manifest_name = "task.json" if layer == "generation" else "case.json"
    manifest = _load_json(case_dir / manifest_name)
    parts.append(f"# {manifest.get('title', case_id)}")
    parts.append("")
    parts.append(f"Layer: {layer}")
    parts.append(f"Case ID: {case_id}")
    parts.append("")
    for rel_name in VISIBLE_FILES[layer]:
        path = case_dir / rel_name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8").rstrip())
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _blank_scorecard(layer: str, case_id: str) -> dict[str, Any]:
    schema = _load_json(SCORECARD_SCHEMA)
    fields = schema["score_fields"]
    return {
        "layer": layer,
        "case_id": case_id,
        "decision": "",
        "rationale": "",
        "scores": {field["name"]: None for field in fields},
        "notes": [],
    }


def _self_test() -> dict[str, Any]:
    cases = _list_cases()
    required = {"generation", "review", "repair"}
    present = {row["layer"] for row in cases}
    ok = required.issubset(present)
    return {
        "ok": ok,
        "layer_counts": {
            layer: sum(1 for row in cases if row["layer"] == layer)
            for layer in sorted(CASE_ROOTS)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", parents=[common])
    show = sub.add_parser("show", parents=[common])
    show.add_argument("layer", choices=sorted(CASE_ROOTS))
    show.add_argument("case_id")

    init = sub.add_parser("init-scorecard", parents=[common])
    init.add_argument("layer", choices=sorted(CASE_ROOTS))
    init.add_argument("case_id")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    json_mode = bool(getattr(args, "json", False))
    if args.self_test:
        payload = _self_test()
        print(_json_dumps(payload) if json_mode else json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload["ok"] else 1
    if args.command == "list":
        payload = {"cases": _list_cases()}
        print(_json_dumps(payload) if json_mode else json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.command == "show":
        print(_show_visible_prompt(args.layer, args.case_id))
        return 0
    if args.command == "init-scorecard":
        payload = _blank_scorecard(args.layer, args.case_id)
        print(_json_dumps(payload) if json_mode else json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
