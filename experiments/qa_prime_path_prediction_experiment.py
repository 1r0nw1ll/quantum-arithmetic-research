#!/usr/bin/env python3
QA_COMPLIANCE = "observer=experiment_script, state_alphabet=mod24"
"""
Measure whether QA reachability/path-length profiles separate primes from non-primes.

This is a falsification-oriented experiment. It does not assume that QA path
length predicts primes; it measures whether prime integers occupy profile
signatures that are exclusive relative to semiprimes and other composites.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Deque

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_lab.qa_core import (
    canonical_json,
    classify_state,
    domain_sha256,
    is_composite,
    is_prime,
    is_semiprime,
    prime_residues,
    qa_norm_mod,
    semiprime_residues,
    state_successors,
    structural_obstruction,
)


def _parse_moduli(raw: str) -> list[int]:
    moduli = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value <= 1:
            raise ValueError("All moduli must be > 1.")
        if value not in moduli:
            moduli.append(value)
    if not moduli:
        raise ValueError("At least one modulus is required.")
    return moduli


def _parse_generator_sets(raw: str) -> list[tuple[str, ...]]:
    sets = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        generator_set = tuple(piece.strip().upper() for piece in chunk.split(",") if piece.strip())
        if not generator_set:
            continue
        if generator_set not in sets:
            sets.append(generator_set)
    if not sets:
        raise ValueError("At least one generator set is required.")
    return sets


def discover_family_sources(modulus: int) -> tuple[dict[str, tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    family_sources: dict[str, tuple[int, int]] = {}
    states_by_family: dict[str, list[tuple[int, int]]] = {}
    for b in range(modulus):
        for e in range(modulus):
            family = classify_state(b, e, modulus)
            states_by_family.setdefault(family, []).append((b, e))
            if family not in family_sources:
                family_sources[family] = (b, e)
    expected = {"singularity", "satellite", "cosmos"}
    missing = expected.difference(family_sources)
    if missing:
        raise ValueError(f"Could not discover canonical sources for families: {sorted(missing)}")
    return family_sources, states_by_family


def bfs_profile(
    sources: list[tuple[int, int]],
    modulus: int,
    generators: tuple[str, ...],
) -> dict[tuple[int, int], int]:
    queue: Deque[tuple[int, int]] = deque(sources)
    distances = {source: 0 for source in sources}
    while queue:
        state = queue.popleft()
        for _generator_id, next_state in state_successors(
            state[0], state[1], modulus, generators=generators
        ):
            if next_state in distances:
                continue
            distances[next_state] = distances[state] + 1
            queue.append(next_state)
    return distances


def residue_profiles(
    modulus: int,
    generators: tuple[str, ...],
) -> dict[int, dict[str, object]]:
    family_sources, states_by_family = discover_family_sources(modulus)
    distances_by_family = {
        family: bfs_profile(source_states, modulus, generators)
        for family, source_states in states_by_family.items()
    }
    states_by_residue: dict[int, list[tuple[int, int]]] = {residue: [] for residue in range(modulus)}
    for b in range(modulus):
        for e in range(modulus):
            states_by_residue[qa_norm_mod(b, e, modulus)].append((b, e))

    profiles: dict[int, dict[str, object]] = {}
    for residue, states in states_by_residue.items():
        family_min_steps: dict[str, int | None] = {}
        family_min_positive_steps: dict[str, int | None] = {}
        family_reachable_counts: dict[str, int] = {}
        family_target_orbits: dict[str, list[str]] = {}
        family_state_support: dict[str, int] = {}
        for family, distances in distances_by_family.items():
            reachable = [state for state in states if state in distances]
            family_reachable_counts[family] = len(reachable)
            family_state_support[family] = sum(
                1 for state in states if classify_state(state[0], state[1], modulus) == family
            )
            if reachable:
                all_steps = [distances[state] for state in reachable]
                positive_steps = [step for step in all_steps if step > 0]
                family_min_steps[family] = min(all_steps)
                family_min_positive_steps[family] = None if not positive_steps else min(positive_steps)
                family_target_orbits[family] = sorted({classify_state(state[0], state[1], modulus) for state in reachable})
            else:
                family_min_steps[family] = None
                family_min_positive_steps[family] = None
                family_target_orbits[family] = []

        profiles[residue] = {
            "residue": residue,
            "obstructed": structural_obstruction(residue, modulus),
            "prime_residue_candidate": residue in prime_residues(modulus),
            "semiprime_residue_candidate": residue in semiprime_residues(modulus),
            "state_count": len(states),
            "state_count_by_orbit_family": family_state_support,
            "min_steps_by_family": family_min_steps,
            "min_positive_steps_by_family": family_min_positive_steps,
            "reachable_state_count_by_family": family_reachable_counts,
            "target_orbit_families_by_source_family": family_target_orbits,
            "canonical_source_examples": {family: list(source) for family, source in family_sources.items()},
        }
    return profiles


def integer_class(n: int) -> str:
    if is_prime(n):
        return "prime"
    if is_semiprime(n):
        return "semiprime"
    if is_composite(n):
        return "composite"
    return "other"


def signature_for_profile(profile: dict[str, object]) -> tuple:
    min_steps = profile["min_steps_by_family"]
    min_positive_steps = profile["min_positive_steps_by_family"]
    reachable_counts = profile["reachable_state_count_by_family"]
    family_support = profile["state_count_by_orbit_family"]
    return (
        profile["obstructed"],
        tuple((family, family_support[family]) for family in sorted(family_support)),
        tuple((family, min_steps[family]) for family in sorted(min_steps)),
        tuple((family, min_positive_steps[family]) for family in sorted(min_positive_steps)),
        tuple((family, reachable_counts[family]) for family in sorted(reachable_counts)),
    )


def evaluate_configuration(
    start: int,
    end: int,
    modulus: int,
    generators: tuple[str, ...],
) -> dict[str, object]:
    profiles = residue_profiles(modulus, generators)
    rows = []
    signatures_to_classes: dict[tuple, set[str]] = {}
    prime_signatures: set[tuple] = set()
    nonprime_signatures: set[tuple] = set()

    for n in range(start, end + 1):
        label = integer_class(n)
        if label == "other":
            continue
        residue = n % modulus
        profile = profiles[residue]
        signature = signature_for_profile(profile)
        rows.append(
            {
                "n": n,
                "class": label,
                "residue": residue,
                "signature": signature,
                "profile": profile,
            }
        )
        signatures_to_classes.setdefault(signature, set()).add(label)
        if label == "prime":
            prime_signatures.add(signature)
        else:
            nonprime_signatures.add(signature)

    prime_exclusive_signatures = prime_signatures.difference(nonprime_signatures)
    prime_rows = [row for row in rows if row["class"] == "prime"]
    semiprime_rows = [row for row in rows if row["class"] == "semiprime"]
    composite_rows = [row for row in rows if row["class"] == "composite"]

    primes_covered = [row["n"] for row in prime_rows if row["signature"] in prime_exclusive_signatures]
    prime_coverage_ratio = 0.0 if not prime_rows else len(primes_covered) / len(prime_rows)

    family_medians = {}
    for family in ("cosmos", "satellite", "singularity"):
        family_medians[family] = {}
        for label, label_rows in (
            ("prime", prime_rows),
            ("semiprime", semiprime_rows),
            ("composite", composite_rows),
        ):
            values = [
                row["profile"]["min_steps_by_family"][family]
                for row in label_rows
                if row["profile"]["min_steps_by_family"][family] is not None
            ]
            family_medians[family][label] = None if not values else statistics.median(values)

    if prime_coverage_ratio == 1.0 and prime_rows:
        verdict = "PASS"
    elif prime_coverage_ratio > 0.0:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    signature_examples = []
    for signature, classes in signatures_to_classes.items():
        if len(signature_examples) >= 10:
            break
        members = [row["n"] for row in rows if row["signature"] == signature][:6]
        signature_examples.append(
            {
                "classes": sorted(classes),
                "members": members,
                "signature": repr(signature),
            }
        )

    return {
        "modulus": modulus,
        "generator_set": list(generators),
        "verdict": verdict,
        "prime_count": len(prime_rows),
        "nonprime_count": len(semiprime_rows) + len(composite_rows),
        "prime_exclusive_signature_count": len(prime_exclusive_signatures),
        "prime_coverage_ratio": prime_coverage_ratio,
        "primes_covered_by_exclusive_signatures": primes_covered,
        "family_median_min_steps": family_medians,
        "signature_examples": signature_examples,
        "honest_interpretation": (
            "Prime prediction is supported only if prime-exclusive QA path signatures cover all interval primes. "
            "Otherwise QA path length is acting as a structural descriptor, not an exact primality law."
        ),
    }


def run_experiment(
    start: int,
    end: int,
    moduli: list[int],
    generator_sets: list[tuple[str, ...]],
) -> dict[str, object]:
    configurations = []
    for modulus in moduli:
        for generators in generator_sets:
            configurations.append(
                evaluate_configuration(start, end, modulus, generators)
            )

    verdicts = [config["verdict"] for config in configurations]
    if "PASS" in verdicts:
        overall = "PASS"
    elif "PARTIAL" in verdicts:
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    best_configuration = max(
        configurations,
        key=lambda item: (item["prime_coverage_ratio"], item["prime_exclusive_signature_count"]),
    )

    payload = {
        "experiment_id": f"qa_prime_path_prediction_experiment_{start}_{end}",
        "hypothesis": (
            "If QA reachability/path-length carries exact prime structure rather than only residue-class structure, "
            "then at least one tested modulus/generator configuration should assign all primes in the interval to "
            "QA path signatures not shared by semiprimes or other composites."
        ),
        "success_criteria": (
            "PASS if any tested configuration yields prime_coverage_ratio = 1.0 with zero non-prime collisions; "
            "PARTIAL if some but not all primes occupy exclusive signatures; FAIL if every prime signature is shared "
            "with non-primes."
        ),
        "interval": {"start": start, "end": end},
        "tested_moduli": moduli,
        "tested_generator_sets": [list(item) for item in generator_sets],
        "result": overall,
        "best_configuration": best_configuration,
        "configurations": configurations,
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_PRIME_PATH_PREDICTION_EXPERIMENT.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Test whether QA path profiles separate primes from non-primes.")
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--moduli", default="24,72")
    parser.add_argument("--generator-sets", default="Q;T;Q,T")
    parser.add_argument(
        "--out",
        default="results/qa_prime_path_prediction_experiment.json",
        help="Where to write the JSON artifact.",
    )
    args = parser.parse_args()

    payload = run_experiment(
        start=int(args.start),
        end=int(args.end),
        moduli=_parse_moduli(args.moduli),
        generator_sets=_parse_generator_sets(args.generator_sets),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"[qa_prime_path_prediction_experiment] Wrote {out_path}")
    print(f"[qa_prime_path_prediction_experiment] Overall result: {payload['result']}")
    best = payload["best_configuration"]
    print(
        "[qa_prime_path_prediction_experiment] Best configuration: "
        f"mod={best['modulus']} generators={best['generator_set']} "
        f"coverage={best['prime_coverage_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
