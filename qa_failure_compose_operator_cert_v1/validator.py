#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema


FAIL_SCHEMA = "SCHEMA_INVALID"
FAIL_DUPLICATE_COMPOSE_ENTRY = "DUPLICATE_COMPOSE_ENTRY"
FAIL_COMPOSE_TABLE_INCOMPLETE = "COMPOSE_TABLE_INCOMPLETE"
FAIL_CLOSURE_VIOLATION = "CLOSURE_VIOLATION"
FAIL_COMPOSE_ASSOCIATIVITY_VIOLATION = "COMPOSE_ASSOCIATIVITY_VIOLATION"
FAIL_RECOMPUTE_MISMATCH = "RECOMPUTE_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gate_1_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
        return None
    except jsonschema.ValidationError as exc:
        return {
            "ok": False,
            "fail_type": FAIL_SCHEMA,
            "invariant_diff": {},
            "details": {"error": str(exc)},
        }


def _compose_rows_sorted(compose_map: Dict[Tuple[str, str, str], str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for (form, a, b), comp in sorted(compose_map.items()):
        out.append({"form": form, "a": a, "b": b, "comp": comp})
    return out


def _build_compose_map(cert: Dict[str, Any]) -> Tuple[Optional[Dict[Tuple[str, str, str], str]], Optional[Dict[str, Any]]]:
    forms = set(cert["forms"])
    carrier = set(cert["carrier"])
    compose_map: Dict[Tuple[str, str, str], str] = {}
    for i, row in enumerate(cert["compose_table"]):
        form = str(row["form"])
        a = str(row["a"])
        b = str(row["b"])
        comp = str(row["comp"])
        if form not in forms:
            return None, {
                "ok": False,
                "fail_type": FAIL_SCHEMA,
                "invariant_diff": {"compose_table.form": {"index": i, "value": form}},
                "details": {"reason": "form not present in forms"},
            }
        if a not in carrier or b not in carrier:
            return None, {
                "ok": False,
                "fail_type": FAIL_SCHEMA,
                "invariant_diff": {"compose_table.args": {"index": i, "a": a, "b": b}},
                "details": {"reason": "a/b not in carrier"},
            }
        key = (form, a, b)
        if key in compose_map:
            return None, {
                "ok": False,
                "fail_type": FAIL_DUPLICATE_COMPOSE_ENTRY,
                "invariant_diff": {"duplicate_key": {"form": form, "a": a, "b": b}},
                "details": {"index": i},
            }
        compose_map[key] = comp
    return compose_map, None


def gate_2_closure(cert: Dict[str, Any], compose_map: Dict[Tuple[str, str, str], str]) -> Optional[Dict[str, Any]]:
    carrier = list(cert["carrier"])
    carrier_set = set(carrier)
    forms = list(cert["forms"])

    for form in forms:
        for a in carrier:
            for b in carrier:
                key = (form, a, b)
                if key not in compose_map:
                    return {
                        "ok": False,
                        "fail_type": FAIL_COMPOSE_TABLE_INCOMPLETE,
                        "invariant_diff": {"missing_entry": {"form": form, "a": a, "b": b}},
                        "details": {},
                    }
                comp = compose_map[key]
                if comp not in carrier_set:
                    return {
                        "ok": False,
                        "fail_type": FAIL_CLOSURE_VIOLATION,
                        "invariant_diff": {
                            "closure": {
                                "form": form,
                                "a": a,
                                "b": b,
                                "comp": comp,
                            }
                        },
                        "details": {},
                    }
    return None


def gate_3_associativity(cert: Dict[str, Any], compose_map: Dict[Tuple[str, str, str], str]) -> Optional[Dict[str, Any]]:
    carrier = list(cert["carrier"])
    forms = list(cert["forms"])

    for form in forms:
        for a in carrier:
            for b in carrier:
                for c in carrier:
                    ab = compose_map[(form, a, b)]
                    bc = compose_map[(form, b, c)]
                    left = compose_map[(form, ab, c)]
                    right = compose_map[(form, a, bc)]
                    if left != right:
                        return {
                            "ok": False,
                            "fail_type": FAIL_COMPOSE_ASSOCIATIVITY_VIOLATION,
                            "invariant_diff": {
                                "associativity": {
                                    "form": form,
                                    "a": a,
                                    "b": b,
                                    "c": c,
                                    "left": left,
                                    "right": right,
                                }
                            },
                            "details": {},
                        }
    return None


def gate_4_claims(cert: Dict[str, Any], recomputed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    claimed = cert["claimed"]
    invariant_diff: Dict[str, Any] = {}

    if bool(claimed["closure_holds"]) != bool(recomputed["closure_holds"]):
        invariant_diff["closure_holds"] = {
            "claimed": bool(claimed["closure_holds"]),
            "recomputed": bool(recomputed["closure_holds"]),
        }

    if bool(claimed["associativity_holds"]) != bool(recomputed["associativity_holds"]):
        invariant_diff["associativity_holds"] = {
            "claimed": bool(claimed["associativity_holds"]),
            "recomputed": bool(recomputed["associativity_holds"]),
        }

    if invariant_diff:
        return {
            "ok": False,
            "fail_type": FAIL_RECOMPUTE_MISMATCH,
            "invariant_diff": invariant_diff,
            "details": {"recomputed": recomputed},
        }
    return None


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    g1 = gate_1_schema(cert, schema)
    if g1 is not None:
        return g1

    compose_map, map_err = _build_compose_map(cert)
    if map_err is not None:
        return map_err

    assert compose_map is not None

    g2 = gate_2_closure(cert, compose_map)
    if g2 is not None:
        return g2

    g3 = gate_3_associativity(cert, compose_map)
    if g3 is not None:
        return g3

    recomputed = {
        "closure_holds": True,
        "associativity_holds": True,
    }

    g4 = gate_4_claims(cert, recomputed)
    if g4 is not None:
        return g4

    compose_digest = sha256_hex_bytes(canonical_json_bytes(_compose_rows_sorted(compose_map)))
    cert_sha256 = sha256_hex_bytes(canonical_json_bytes(cert))

    return {
        "ok": True,
        "value": {
            "cert_id": cert["cert_id"],
            "carrier_size": len(cert["carrier"]),
            "forms": list(cert["forms"]),
            "compose_row_count": len(compose_map),
            "recomputed": recomputed,
            "compose_digest": compose_digest,
            "cert_sha256": cert_sha256,
        },
    }


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures = [
        ("pass_feedback_escalation.json", True, None),
        ("fail_closure_incomplete_table.json", False, FAIL_COMPOSE_TABLE_INCOMPLETE),
        ("fail_associativity_feedback_violation.json", False, FAIL_COMPOSE_ASSOCIATIVITY_VIOLATION),
    ]

    results: List[Dict[str, Any]] = []
    failed: List[str] = []

    for name, expected_ok, expected_fail_type in fixtures:
        cert = load_json(base / "fixtures" / name)
        out = validate_cert(cert, schema)
        got_ok = bool(out.get("ok"))
        got_fail_type = out.get("fail_type")
        ok_match = got_ok == expected_ok
        fail_match = expected_fail_type is None or got_fail_type == expected_fail_type
        if not ok_match or not fail_match:
            failed.append(name)
        results.append(
            {
                "fixture": name,
                "expected_ok": expected_ok,
                "expected_fail_type": expected_fail_type,
                "got_ok": got_ok,
                "got_fail_type": got_fail_type,
                "result": out,
            }
        )

    payload = {
        "ok": len(failed) == 0,
        "failed_fixtures": failed,
        "fixtures": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if len(failed) == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA Failure Compose Operator Cert v1")
    parser.add_argument("--schema", default="schema.json")
    parser.add_argument("--cert", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    if args.cert is None:
        parser.error("provide --cert or use --self-test")

    schema = load_json(Path(args.schema))
    cert = load_json(Path(args.cert))
    out = validate_cert(cert, schema)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
