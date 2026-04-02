#!/usr/bin/env python3
# Entry point: python qa_modal_reachability.py --emit --outdir .

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required to load qa_modal_reachability.yaml. "
        "Install with: pip install pyyaml"
    ) from exc


SPEC_PATH = str(Path(__file__).resolve().with_name("qa_modal_reachability.yaml"))


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_domain(domain: Any) -> List[str]:
    if domain is None:
        return []
    if isinstance(domain, list):
        return domain
    return [domain]


def _resolve_state_id(name: str, state_ids: Dict[str, str]) -> str:
    if name in state_ids:
        return state_ids[name]
    return name


def build_toy_graph(spec: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    states = spec.get("states", {})
    state_ids = {name: info.get("id", name) for name, info in states.items()}
    graph: Dict[str, List[Dict[str, Any]]] = {}

    for state_id in state_ids.values():
        graph[state_id] = []
    graph.setdefault("REDUCED", [])

    generators = spec.get("generators", {})
    for gen in generators.values():
        gen_id = gen.get("id", "unknown_generator")
        domain_list = _normalize_domain(gen.get("domain"))
        codomain = gen.get("codomain")

        for domain in domain_list:
            source_id = _resolve_state_id(domain, state_ids)
            if source_id not in graph:
                graph[source_id] = []

            if codomain == "same":
                target_id = source_id
            elif codomain == "reduced_state":
                target_id = "REDUCED"
            elif isinstance(codomain, str):
                target_id = _resolve_state_id(codomain, state_ids)
            else:
                target_id = "UNKNOWN"

            edge: Dict[str, Any] = {"generator": gen_id, "target": target_id}
            if gen.get("may_fail"):
                edge["may_fail"] = list(gen.get("may_fail", []))
            graph[source_id].append(edge)

    return graph


def _collect_ids(spec: Dict[str, Any]) -> Tuple[set, set]:
    generator_ids = {
        gen_id for gen_id in (gen.get("id") for gen in spec.get("generators", {}).values()) if gen_id
    }
    failure_ids = set(spec.get("failures", {}).keys())
    return generator_ids, failure_ids


def _infer_certificate_type(cert: Dict[str, Any]) -> str:
    if "fail_type" in cert:
        return "CYCLE_IMPOSSIBLE"
    if "path" in cert:
        return "RETURN_CONSTRUCTED"
    raise ValueError("Unable to infer certificate type (missing path/fail_type).")


def _required_fields_for(schema: Dict[str, Any]) -> Tuple[set, set]:
    required: set = set()
    optional: set = set()
    for field, rule in schema.get("fields", {}).items():
        if rule == "optional":
            optional.add(field)
        elif isinstance(rule, dict) and rule.get("type") == "optional":
            optional.add(field)
        else:
            required.add(field)
    return required, optional


def validate_certificate(cert: Dict[str, Any], spec: Dict[str, Any]) -> None:
    generator_ids, failure_ids = _collect_ids(spec)
    states = spec.get("states", {})
    state_ids = {info.get("id", name) for name, info in states.items()}
    cert_type = _infer_certificate_type(cert)
    schema = spec.get("certificates", {}).get(cert_type, {})
    required_fields, optional_fields = _required_fields_for(schema)

    missing = required_fields.difference(cert.keys())
    if missing:
        raise ValueError(
            f"{cert_type} certificate missing required fields: {sorted(missing)}"
        )

    allowed_fields = required_fields.union(optional_fields)
    extra = set(cert.keys()).difference(allowed_fields)
    if extra:
        raise ValueError(f"{cert_type} certificate has unknown fields: {sorted(extra)}")

    if "path" in cert:
        for gen_id in cert.get("path", []):
            if gen_id not in generator_ids:
                raise ValueError(f"Unknown generator id in certificate path: {gen_id}")

    if "fail_type" in cert:
        fail_type = cert.get("fail_type")
        if fail_type not in failure_ids:
            raise ValueError(f"Unknown failure id in certificate: {fail_type}")

    for key in ("source_state", "target_state"):
        value = cert.get(key)
        if value is not None and value not in state_ids:
            raise ValueError(f"Unknown state id in certificate: {value}")


def _state_capabilities(spec: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    capabilities: Dict[str, Dict[str, bool]] = {}
    for name, info in spec.get("states", {}).items():
        state_id = info.get("id", name)
        state_caps = info.get("capabilities", {}) or {}
        capabilities[state_id] = {
            key: bool(value) for key, value in state_caps.items()
        }
    return capabilities


def _generator_map(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for gen in spec.get("generators", {}).values():
        gen_id = gen.get("id")
        if gen_id:
            mapping[gen_id] = gen
    return mapping


def _iter_requires_capabilities(gen: Dict[str, Any]) -> Iterable[str]:
    return gen.get("requires_capabilities", []) or []


def _target_reachable_without_reducing(
    spec: Dict[str, Any],
    source: str,
    target: str,
    avoid_may_fail: bool,
) -> bool:
    graph = build_toy_graph(spec)
    generator_map = _generator_map(spec)
    capabilities = _state_capabilities(spec)

    visited = {source}
    queue = [source]

    while queue:
        state = queue.pop(0)
        if state == target:
            return True

        for edge in graph.get(state, []):
            gen_id = edge.get("generator")
            gen = generator_map.get(gen_id, {})
            if avoid_may_fail and edge.get("may_fail"):
                continue

            next_state = edge.get("target")
            is_reducing = bool(gen.get("irreversible")) or next_state == "REDUCED"
            if is_reducing:
                continue

            required_caps = list(_iter_requires_capabilities(gen))
            state_caps = capabilities.get(state, {})
            missing = [cap for cap in required_caps if not state_caps.get(cap, False)]
            if missing:
                continue

            if next_state in visited:
                continue
            visited.add(next_state)
            queue.append(next_state)

    return False


def find_reachability_certificate(
    spec: Dict[str, Any],
    source: str,
    target: str,
    avoid_may_fail: bool,
) -> Dict[str, Any]:
    graph = build_toy_graph(spec)
    generator_map = _generator_map(spec)
    capabilities = _state_capabilities(spec)

    if source not in graph:
        raise ValueError(f"Unknown source state id: {source}")
    if target not in graph:
        raise ValueError(f"Unknown target state id: {target}")

    visited = {source}
    queue: List[Tuple[str, List[str], set, set]] = [(source, [], set(), set())]
    missing_caps: set = set()
    reduced_reachable = False

    while queue:
        state, path, preconditions, invariants = queue.pop(0)
        if state == target:
            cert = {
                "source_state": source,
                "target_state": target,
                "path": path,
                "preconditions_met": sorted(preconditions),
                "invariants_preserved": sorted(invariants),
                "error_bounds": 0.0,
                "notes": "Shortest path via BFS.",
            }
            validate_certificate(cert, spec)
            return cert

        for edge in graph.get(state, []):
            gen_id = edge.get("generator")
            gen = generator_map.get(gen_id, {})

            if avoid_may_fail and edge.get("may_fail"):
                continue

            required_caps = list(_iter_requires_capabilities(gen))
            state_caps = capabilities.get(state, {})
            missing = [cap for cap in required_caps if not state_caps.get(cap, False)]
            if missing:
                missing_caps.update(missing)
                continue

            next_state = edge.get("target")
            if next_state == "REDUCED":
                reduced_reachable = True
            if next_state in visited:
                continue

            next_preconditions = set(preconditions)
            next_preconditions.update(required_caps)
            next_invariants = set(invariants)
            next_invariants.update(gen.get("preserves", []) or [])

            visited.add(next_state)
            queue.append((next_state, path + [gen_id], next_preconditions, next_invariants))

    if missing_caps:
        fail_type = "GENERATOR_INSUFFICIENT"
    elif reduced_reachable and not _target_reachable_without_reducing(
        spec, source, target, avoid_may_fail
    ):
        fail_type = "NON_IDENTIFIABLE"
    else:
        fail_type = "UNREACHABLE"

    cert = {
        "attempted_goal": f"{source}_to_{target}",
        "required_missing_information": sorted(missing_caps),
        "fail_type": fail_type,
        "invariant_difference": [],
        "notes": "No admissible path found under current constraints.",
    }
    validate_certificate(cert, spec)
    return cert


def build_sample_certificates() -> List[Tuple[str, Dict[str, Any]]]:
    return [
        (
            "return_constructed_hsi_lidar.json",
            {
                "source_state": "S_L",
                "target_state": "S_H",
                "path": [
                    "gen_pose_transform",
                    "gen_repr_swap",
                    "gen_coregister_hsi_lidar",
                    "gen_refine",
                ],
                "preconditions_met": [
                    "has_pose",
                    "has_extrinsics",
                    "supports_coregistration",
                ],
                "invariants_preserved": ["geometry_connectivity", "spatial_adjacency"],
                "error_bounds": 0.05,
                "notes": "Toy fusion path for geometry-to-spectral alignment.",
            },
        ),
        (
            "return_constructed_hsi_ir.json",
            {
                "source_state": "S_H",
                "target_state": "S_IR",
                "path": ["gen_thermal_forward"],
                "preconditions_met": [
                    "has_temperature_model",
                    "has_emissivity_model",
                    "has_radiometric_model",
                ],
                "invariants_preserved": ["spatial_adjacency", "thermal_contrast"],
                "error_bounds": 0.1,
                "notes": "Toy forward model from spectra to IR radiance.",
            },
        ),
        (
            "cycle_impossible_rgb_hsi.json",
            {
                "attempted_goal": "RGB_to_HSI",
                "required_missing_information": [
                    "spectral_response",
                    "illumination_model",
                    "priors",
                ],
                "fail_type": "NON_IDENTIFIABLE",
                "invariant_difference": ["spectral_shape", "band_order"],
                "notes": "Projection removes bands; inversion is underdetermined.",
            },
        ),
        (
            "cycle_impossible_ir_hsi.json",
            {
                "attempted_goal": "IR_to_HSI_full_spectrum",
                "required_missing_information": [
                    "emissivity_model",
                    "temperature_model",
                ],
                "fail_type": "EMISSIVITY_TEMPERATURE_AMBIGUITY",
                "invariant_difference": ["spectral_shape", "band_order"],
                "notes": "Temperature-emissivity coupling blocks unique recovery.",
            },
        ),
    ]


def emit_certificates(
    certs: List[Tuple[str, Dict[str, Any]]],
    spec: Dict[str, Any],
    outdir: str,
) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []

    for filename, payload in certs:
        validate_certificate(payload, spec)
        output_path = os.path.join(outdir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        written.append(output_path)

    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Toy reachability graph builder and certificate emitter."
    )
    parser.add_argument(
        "--spec",
        default=SPEC_PATH,
        help="Path to qa_modal_reachability.yaml",
    )
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Emit example JSON certificates",
    )
    parser.add_argument(
        "--source",
        help="Source state id for reachability query (e.g., S_L)",
    )
    parser.add_argument(
        "--target",
        help="Target state id for reachability query (e.g., S_H)",
    )
    parser.add_argument(
        "--avoid-may-fail",
        action="store_true",
        help="Avoid edges annotated with may_fail",
    )
    parser.add_argument(
        "--emit-certificate",
        help="Output path for reachability query certificate (JSON)",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Output directory for emitted JSON certificates",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    spec = load_spec(args.spec)
    graph = build_toy_graph(spec)
    emit_to_stdout = bool(args.source and args.target and not args.emit_certificate)

    if not emit_to_stdout:
        edge_count = sum(len(edges) for edges in graph.values())
        print(f"Loaded {len(graph)} nodes with {edge_count} toy edges.")

    if args.emit:
        certs = build_sample_certificates()
        written = emit_certificates(certs, spec, args.outdir)
        if not emit_to_stdout:
            print(f"Emitted {len(written)} JSON certificates to {args.outdir}.")

    if args.source and args.target:
        cert = find_reachability_certificate(
            spec,
            source=args.source,
            target=args.target,
            avoid_may_fail=args.avoid_may_fail,
        )
        if args.emit_certificate:
            output_path = args.emit_certificate
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(cert, handle, indent=2, sort_keys=True)
                handle.write("\n")
            if not emit_to_stdout:
                print(f"Wrote reachability certificate to {output_path}.")
        else:
            print(json.dumps(cert, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
