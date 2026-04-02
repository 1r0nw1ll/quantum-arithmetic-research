#!/usr/bin/env python3
"""
Annotate a single integer with exact arithmetic and QA-overlay metadata.

Examples
--------
python tools/qa_integer_annotate.py --n 221
python tools/qa_integer_annotate.py --n 997 --moduli 24,72 --pretty
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_lab.qa_core import (
    canonical_json,
    domain_sha256,
    factor_integer,
    is_composite,
    is_prime,
    is_prime_power,
    is_semiprime,
    is_squarefree,
    largest_prime_factor,
    mobius,
    omega,
    prime_residues,
    semiprime_residues,
    sigma,
    smallest_prime_factor,
    structural_obstruction,
    tau,
    totient,
    big_omega,
)


SCHEMA_PATH = "schemas/qa_integer_annotation.schema.json"


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


def _previous_prime(n: int) -> int | None:
    candidate = n - 1
    while candidate >= 2:
        if is_prime(candidate):
            return candidate
        candidate -= 1
    return None


def _next_prime(n: int) -> int:
    candidate = max(2, n + 1)
    while not is_prime(candidate):
        candidate += 1
    return candidate


def _partners(n: int, gap: int) -> list[int]:
    if not is_prime(n):
        return []
    partners = []
    for other in (n - gap, n + gap):
        if other >= 2 and is_prime(other):
            partners.append(other)
    return sorted(partners)


def build_annotation(n: int, moduli: list[int]) -> dict:
    factors = factor_integer(n)
    previous_prime = _previous_prime(n)
    next_prime = _next_prime(n)
    payload = {
        "schema": SCHEMA_PATH,
        "n": n,
        "classification": {
            "is_prime": is_prime(n),
            "is_composite": is_composite(n),
            "is_semiprime": is_semiprime(n),
            "is_prime_power": is_prime_power(n),
            "is_squarefree": is_squarefree(n),
            "is_twin_prime_member": bool(_partners(n, 2)),
            "is_cousin_prime_member": bool(_partners(n, 4)),
            "is_sexy_prime_member": bool(_partners(n, 6)),
        },
        "factorization": {
            "prime_powers": [{"p": prime, "e": exponent} for prime, exponent in factors],
            "omega": omega(n),
            "Omega": big_omega(n),
            "largest_prime_factor": largest_prime_factor(n),
            "smallest_prime_factor": smallest_prime_factor(n),
        },
        "arithmetic": {
            "tau": tau(n),
            "sigma": sigma(n),
            "mobius": mobius(n),
            "totient": totient(n),
        },
        "neighbors": {
            "previous_prime": previous_prime,
            "next_prime": next_prime,
            "twin_prime_partners": _partners(n, 2),
            "cousin_prime_partners": _partners(n, 4),
            "sexy_prime_partners": _partners(n, 6),
        },
        "residues": {f"mod_{modulus}": n % modulus for modulus in moduli},
        "qa_overlay": {
            f"mod_{modulus}": {
                "residue": n % modulus,
                "prime_residue_candidate": (n % modulus) in prime_residues(modulus),
                "semiprime_residue_candidate": (n % modulus) in semiprime_residues(modulus),
                "obstructed": structural_obstruction(n % modulus, modulus),
            }
            for modulus in moduli
        },
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_INTEGER_ANNOTATION.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate one integer with exact number-theory metadata.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--moduli", default="24,72")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    moduli = _parse_moduli(args.moduli)
    payload = build_annotation(int(args.n), moduli)
    if args.pretty:
        print(__import__("json").dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
