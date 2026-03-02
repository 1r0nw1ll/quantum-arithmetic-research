#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema


FAIL_SCHEMA = "SCHEMA_INVALID"
FAIL_FAMILY87_REF_MISSING = "FAMILY87_REF_MISSING"
FAIL_FAMILY87_REF_HASH_MISMATCH = "FAMILY87_REF_HASH_MISMATCH"
FAIL_DUPLICATE_COMPOSE_ENTRY = "DUPLICATE_COMPOSE_ENTRY"
FAIL_COMPOSE_TABLE_NONCANONICAL_ORDER = "COMPOSE_TABLE_NONCANONICAL_ORDER"
FAIL_COMPOSE_TABLE_INCOMPLETE = "COMPOSE_TABLE_INCOMPLETE"
FAIL_CLOSURE_VIOLATION = "CLOSURE_VIOLATION"
FAIL_COMPOSE_ASSOCIATIVITY_VIOLATION = "COMPOSE_ASSOCIATIVITY_VIOLATION"
FAIL_IDENTITY_CLAIM_MISMATCH = "IDENTITY_CLAIM_MISMATCH"
FAIL_ABSORBER_CLAIM_MISMATCH = "ABSORBER_CLAIM_MISMATCH"
FAIL_COMMUTATIVITY_CLAIM_MISMATCH = "COMMUTATIVITY_CLAIM_MISMATCH"
FAIL_PREORDER_INVALID = "PREORDER_INVALID"
FAIL_MONOTONICITY_VIOLATION = "MONOTONICITY_VIOLATION"
FAIL_MONOTONICITY_CLAIM_MISMATCH = "MONOTONICITY_CLAIM_MISMATCH"
FAIL_RECOMPUTE_MISMATCH = "RECOMPUTE_MISMATCH"


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _key_rank(forms: List[str], carrier: List[str], key: Tuple[str, str, str]) -> Tuple[int, int, int]:
    form, a, b = key
    return (forms.index(form), carrier.index(a), carrier.index(b))


def _claims_map(cert: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in cert["claimed"]["per_form_claims"]:
        out[str(row["form"])] = row
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
        if form not in forms or a not in carrier or b not in carrier:
            return None, {
                "ok": False,
                "fail_type": FAIL_SCHEMA,
                "invariant_diff": {"compose_table.args": {"index": i, "row": row}},
                "details": {"reason": "form/a/b not in forms/carrier"},
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


def gate_1_schema_and_pins(cert: Dict[str, Any], schema: Dict[str, Any], repo_root: Path) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {
            "ok": False,
            "fail_type": FAIL_SCHEMA,
            "invariant_diff": {},
            "details": {"error": str(exc)},
        }

    forms = list(cert["forms"])
    carrier = list(cert["carrier"])
    claims = cert["claimed"]["per_form_claims"]
    claim_forms = [str(x["form"]) for x in claims]
    if set(claim_forms) != set(forms) or len(claim_forms) != len(forms):
        return {
            "ok": False,
            "fail_type": FAIL_SCHEMA,
            "invariant_diff": {
                "per_form_claims": {
                    "forms": forms,
                    "claim_forms": claim_forms,
                }
            },
            "details": {"reason": "per_form_claims must contain exactly one entry per form"},
        }

    ref_path_rel = cert["family87_ref"]["path"]
    ref_path = (repo_root / ref_path_rel).resolve()
    if not ref_path.exists():
        return {
            "ok": False,
            "fail_type": FAIL_FAMILY87_REF_MISSING,
            "invariant_diff": {"family87_ref.path": {"claimed": ref_path_rel}},
            "details": {},
        }

    ref_obj = load_json(ref_path)
    ref_hash = sha256_hex_bytes(canonical_json_bytes(ref_obj))
    if ref_hash != cert["family87_ref"]["cert_sha256"]:
        return {
            "ok": False,
            "fail_type": FAIL_FAMILY87_REF_HASH_MISMATCH,
            "invariant_diff": {
                "family87_ref.cert_sha256": {
                    "claimed": cert["family87_ref"]["cert_sha256"],
                    "recomputed": ref_hash,
                }
            },
            "details": {"resolved_path": str(ref_path)},
        }

    compose_map, map_err = _build_compose_map(cert)
    if map_err is not None:
        return map_err

    assert compose_map is not None

    prev_rank: Optional[Tuple[int, int, int]] = None
    for i, row in enumerate(cert["compose_table"]):
        key = (str(row["form"]), str(row["a"]), str(row["b"]))
        rank = _key_rank(forms, carrier, key)
        if prev_rank is not None and rank < prev_rank:
            return {
                "ok": False,
                "fail_type": FAIL_COMPOSE_TABLE_NONCANONICAL_ORDER,
                "invariant_diff": {
                    "compose_table.order": {
                        "index": i,
                        "prev_rank": prev_rank,
                        "rank": rank,
                        "row": row,
                    }
                },
                "details": {},
            }
        prev_rank = rank

    return None


def gate_2_closure_associativity(cert: Dict[str, Any], compose_map: Dict[Tuple[str, str, str], str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    forms = list(cert["forms"])
    carrier = list(cert["carrier"])
    carrier_set = set(carrier)

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
                    }, {}
                comp = compose_map[key]
                if comp not in carrier_set:
                    return {
                        "ok": False,
                        "fail_type": FAIL_CLOSURE_VIOLATION,
                        "invariant_diff": {
                            "closure": {"form": form, "a": a, "b": b, "comp": comp}
                        },
                        "details": {},
                    }, {}

    for form in forms:
        for a in carrier:
            for b in carrier:
                for c in carrier:
                    left = compose_map[(form, compose_map[(form, a, b)], c)]
                    right = compose_map[(form, a, compose_map[(form, b, c)])]
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
                        }, {}

    return None, {
        "closure_holds": True,
        "associativity_holds": True,
        "per_form": {form: {"semigroup": True} for form in forms},
    }


def _identity_candidates(form: str, carrier: List[str], compose_map: Dict[Tuple[str, str, str], str]) -> List[str]:
    out: List[str] = []
    for e in carrier:
        ok = True
        for x in carrier:
            if compose_map[(form, e, x)] != x or compose_map[(form, x, e)] != x:
                ok = False
                break
        if ok:
            out.append(e)
    return out


def _absorber_candidates(form: str, carrier: List[str], compose_map: Dict[Tuple[str, str, str], str]) -> List[str]:
    out: List[str] = []
    for z in carrier:
        ok = True
        for x in carrier:
            if compose_map[(form, z, x)] != z or compose_map[(form, x, z)] != z:
                ok = False
                break
        if ok:
            out.append(z)
    return out


def _identity_witness(form: str, e: str, carrier: List[str], compose_map: Dict[Tuple[str, str, str], str]) -> Optional[Dict[str, Any]]:
    for x in carrier:
        left = compose_map[(form, e, x)]
        right = compose_map[(form, x, e)]
        if left != x or right != x:
            return {"x": x, "left": left, "right": right}
    return None


def _absorber_witness(form: str, z: str, carrier: List[str], compose_map: Dict[Tuple[str, str, str], str]) -> Optional[Dict[str, Any]]:
    for x in carrier:
        left = compose_map[(form, z, x)]
        right = compose_map[(form, x, z)]
        if left != z or right != z:
            return {"x": x, "left": left, "right": right}
    return None


def gate_3_identity_absorber(cert: Dict[str, Any], compose_map: Dict[Tuple[str, str, str], str], recomputed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    forms = list(cert["forms"])
    carrier = list(cert["carrier"])
    claims = _claims_map(cert)

    for form in forms:
        id_cands = _identity_candidates(form, carrier, compose_map)
        abs_cands = _absorber_candidates(form, carrier, compose_map)
        identity = id_cands[0] if len(id_cands) == 1 else None
        absorber = abs_cands[0] if len(abs_cands) == 1 else None

        recomputed["per_form"][form].update(
            {
                "identity_candidates": id_cands,
                "absorber_candidates": abs_cands,
                "identity": identity,
                "absorber": absorber,
                "monoid": identity is not None,
            }
        )

        claim = claims[form]
        claimed_identity = claim["identity"]
        claimed_absorber = claim["absorber"]

        if claimed_identity != identity:
            witness = None
            if claimed_identity is not None and claimed_identity in set(carrier):
                witness = _identity_witness(form, claimed_identity, carrier, compose_map)
            return {
                "ok": False,
                "fail_type": FAIL_IDENTITY_CLAIM_MISMATCH,
                "invariant_diff": {
                    "identity": {
                        "form": form,
                        "claimed": claimed_identity,
                        "recomputed": identity,
                        "identity_candidates": id_cands,
                        "witness": witness,
                    }
                },
                "details": {},
            }

        if claimed_absorber != absorber:
            witness = None
            if claimed_absorber is not None and claimed_absorber in set(carrier):
                witness = _absorber_witness(form, claimed_absorber, carrier, compose_map)
            return {
                "ok": False,
                "fail_type": FAIL_ABSORBER_CLAIM_MISMATCH,
                "invariant_diff": {
                    "absorber": {
                        "form": form,
                        "claimed": claimed_absorber,
                        "recomputed": absorber,
                        "absorber_candidates": abs_cands,
                        "witness": witness,
                    }
                },
                "details": {},
            }

    return None


def gate_4_commutativity(cert: Dict[str, Any], compose_map: Dict[Tuple[str, str, str], str], recomputed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    claims = _claims_map(cert)
    forms = list(cert["forms"])
    carrier = list(cert["carrier"])

    if bool(cert["claimed"]["closure_holds"]) != bool(recomputed["closure_holds"]):
        return {
            "ok": False,
            "fail_type": FAIL_RECOMPUTE_MISMATCH,
            "invariant_diff": {
                "closure_holds": {
                    "claimed": bool(cert["claimed"]["closure_holds"]),
                    "recomputed": bool(recomputed["closure_holds"]),
                }
            },
            "details": {},
        }
    if bool(cert["claimed"]["associativity_holds"]) != bool(recomputed["associativity_holds"]):
        return {
            "ok": False,
            "fail_type": FAIL_RECOMPUTE_MISMATCH,
            "invariant_diff": {
                "associativity_holds": {
                    "claimed": bool(cert["claimed"]["associativity_holds"]),
                    "recomputed": bool(recomputed["associativity_holds"]),
                }
            },
            "details": {},
        }

    for form in forms:
        commutative = True
        witness: Optional[Dict[str, Any]] = None
        for a in carrier:
            for b in carrier:
                ab = compose_map[(form, a, b)]
                ba = compose_map[(form, b, a)]
                if ab != ba:
                    commutative = False
                    witness = {"a": a, "b": b, "ab": ab, "ba": ba}
                    break
            if witness is not None:
                break

        recomputed["per_form"][form].update({"commutative": commutative})

        if bool(claims[form]["commutative"]) != commutative:
            return {
                "ok": False,
                "fail_type": FAIL_COMMUTATIVITY_CLAIM_MISMATCH,
                "invariant_diff": {
                    "commutative": {
                        "form": form,
                        "claimed": bool(claims[form]["commutative"]),
                        "recomputed": commutative,
                        "witness": witness,
                    }
                },
                "details": {},
            }

    return None


def _leq_from_preorder(cert: Dict[str, Any], carrier: List[str]) -> Tuple[Optional[Set[Tuple[str, str]]], Optional[Dict[str, Any]]]:
    preorder = cert.get("preorder")
    if preorder is None:
        return None, None

    carrier_set = set(carrier)
    leq_pairs: Set[Tuple[str, str]] = set()
    for i, row in enumerate(preorder["leq"]):
        a = str(row["a"])
        b = str(row["b"])
        if a not in carrier_set or b not in carrier_set:
            return None, {
                "ok": False,
                "fail_type": FAIL_PREORDER_INVALID,
                "invariant_diff": {"preorder.leq": {"index": i, "row": row}},
                "details": {"reason": "pair element not in carrier"},
            }
        leq_pairs.add((a, b))

    for x in carrier:
        if (x, x) not in leq_pairs:
            return None, {
                "ok": False,
                "fail_type": FAIL_PREORDER_INVALID,
                "invariant_diff": {"preorder.reflexive": {"missing": x}},
                "details": {},
            }

    for a in carrier:
        for b in carrier:
            if (a, b) not in leq_pairs:
                continue
            for c in carrier:
                if (b, c) in leq_pairs and (a, c) not in leq_pairs:
                    return None, {
                        "ok": False,
                        "fail_type": FAIL_PREORDER_INVALID,
                        "invariant_diff": {
                            "preorder.transitive": {
                                "a": a,
                                "b": b,
                                "c": c,
                            }
                        },
                        "details": {},
                    }

    return leq_pairs, None


def gate_5_monotonicity(cert: Dict[str, Any], compose_map: Dict[Tuple[str, str, str], str], recomputed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    forms = list(cert["forms"])
    carrier = list(cert["carrier"])
    claims = _claims_map(cert)

    leq_pairs, preorder_err = _leq_from_preorder(cert, carrier)
    if preorder_err is not None:
        return preorder_err

    if leq_pairs is None:
        for form in forms:
            recomputed["per_form"][form]["monotone"] = None
            if claims[form]["monotone"] is not None:
                return {
                    "ok": False,
                    "fail_type": FAIL_MONOTONICITY_CLAIM_MISMATCH,
                    "invariant_diff": {
                        "monotone": {
                            "form": form,
                            "claimed": claims[form]["monotone"],
                            "recomputed": None,
                            "reason": "preorder not provided",
                        }
                    },
                    "details": {},
                }
        recomputed["monotonicity"] = {"provided": False}
        return None

    for form in forms:
        for a in carrier:
            for a2 in carrier:
                if (a, a2) not in leq_pairs:
                    continue
                for b in carrier:
                    for b2 in carrier:
                        if (b, b2) not in leq_pairs:
                            continue
                        lhs = compose_map[(form, a, b)]
                        rhs = compose_map[(form, a2, b2)]
                        if (lhs, rhs) not in leq_pairs:
                            return {
                                "ok": False,
                                "fail_type": FAIL_MONOTONICITY_VIOLATION,
                                "invariant_diff": {
                                    "monotonicity": {
                                        "form": form,
                                        "a": a,
                                        "a_prime": a2,
                                        "b": b,
                                        "b_prime": b2,
                                        "compose_ab": lhs,
                                        "compose_a_prime_b_prime": rhs,
                                    }
                                },
                                "details": {},
                            }
        recomputed["per_form"][form]["monotone"] = True
        if claims[form]["monotone"] is not True:
            return {
                "ok": False,
                "fail_type": FAIL_MONOTONICITY_CLAIM_MISMATCH,
                "invariant_diff": {
                    "monotone": {
                        "form": form,
                        "claimed": claims[form]["monotone"],
                        "recomputed": True,
                    }
                },
                "details": {},
            }

    recomputed["monotonicity"] = {
        "provided": True,
        "pair_count": len(leq_pairs),
    }
    return None


def _classification_payload(forms: List[str], recomputed: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for form in forms:
        row = recomputed["per_form"][form]
        typ = "monoid" if row["monoid"] else "semigroup"
        out[form] = {
            "type": typ,
            "commutative": row["commutative"],
            "identity": row["identity"],
            "absorber": row["absorber"],
            "monotone": row.get("monotone"),
        }
    return out


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    g1 = gate_1_schema_and_pins(cert, schema, repo_root)
    if g1 is not None:
        return g1

    compose_map, map_err = _build_compose_map(cert)
    if map_err is not None:
        return map_err
    assert compose_map is not None

    g2, recomputed = gate_2_closure_associativity(cert, compose_map)
    if g2 is not None:
        return g2

    g3 = gate_3_identity_absorber(cert, compose_map, recomputed)
    if g3 is not None:
        return g3

    g4 = gate_4_commutativity(cert, compose_map, recomputed)
    if g4 is not None:
        return g4

    g5 = gate_5_monotonicity(cert, compose_map, recomputed)
    if g5 is not None:
        return g5

    forms = list(cert["forms"])
    classification = _classification_payload(forms, recomputed)

    compose_digest = sha256_hex_bytes(
        canonical_json_bytes(
            [
                {"form": f, "a": a, "b": b, "comp": compose_map[(f, a, b)]}
                for f in forms
                for a in cert["carrier"]
                for b in cert["carrier"]
            ]
        )
    )

    return {
        "ok": True,
        "family": 88,
        "carrier_size": len(cert["carrier"]),
        "forms": forms,
        "recomputed": recomputed,
        "classification": classification,
        "compose_digest": compose_digest,
        "cert_sha256": sha256_hex_bytes(canonical_json_bytes(cert)),
        "witnesses": [],
    }


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures = [
        ("pass_classify_from_family87_tables.json", True, None),
        ("fail_identity_claim_wrong.json", False, FAIL_IDENTITY_CLAIM_MISMATCH),
        ("fail_absorber_claim_wrong.json", False, FAIL_ABSORBER_CLAIM_MISMATCH),
        ("fail_commutative_claim_wrong.json", False, FAIL_COMMUTATIVITY_CLAIM_MISMATCH),
        ("fail_monotonicity_violation.json", False, FAIL_MONOTONICITY_VIOLATION),
    ]

    failed: List[str] = []
    results: List[Dict[str, Any]] = []

    repo_root = base.parent

    for name, expected_ok, expected_fail_type in fixtures:
        cert = load_json(base / "fixtures" / name)
        out = validate_cert(cert, schema, repo_root)
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
    parser = argparse.ArgumentParser(description="Validate QA Failure Algebra Structure Classification Cert v1")
    parser.add_argument("--schema", default="schema.json")
    parser.add_argument("--cert")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    if not args.cert:
        parser.error("provide --cert or use --self-test")

    base = Path(__file__).resolve().parent
    repo_root = base.parent
    schema = load_json(Path(args.schema))
    cert = load_json(Path(args.cert))
    out = validate_cert(cert, schema, repo_root)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
