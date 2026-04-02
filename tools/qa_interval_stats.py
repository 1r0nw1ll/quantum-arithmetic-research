#!/usr/bin/env python3
"""
Exact interval statistics for primes and related arithmetic families.

Examples
--------
python tools/qa_interval_stats.py --start 1 --end 1000
python tools/qa_interval_stats.py --start 100 --end 500 --moduli 24,72 --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_lab.qa_core import (
    canonical_json,
    domain_sha256,
    obstructed_prime_residues,
    obstructed_semiprime_residues,
    prime_residues,
    semiprime_residues,
)


SCHEMA_PATH = "schemas/qa_interval_stats.schema.json"


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


def _smallest_prime_factors(limit: int) -> list[int]:
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    divisor = 2
    while divisor * divisor <= limit:
        if spf[divisor] == divisor:
            multiple = divisor * divisor
            while multiple <= limit:
                if spf[multiple] == multiple:
                    spf[multiple] = divisor
                multiple += divisor
        divisor += 1
    return spf


def _factor_from_spf(n: int, spf: list[int]) -> list[tuple[int, int]]:
    if n < 2:
        return []
    factors = []
    while n > 1:
        prime = spf[n]
        exponent = 0
        while n % prime == 0:
            exponent += 1
            n //= prime
        factors.append((prime, exponent))
    return factors


def _is_prime_from_spf(n: int, spf: list[int]) -> bool:
    return n >= 2 and spf[n] == n


def _is_semiprime_from_spf(n: int, spf: list[int]) -> bool:
    return sum(exponent for _prime, exponent in _factor_from_spf(n, spf)) == 2


def _is_prime_power_from_spf(n: int, spf: list[int]) -> bool:
    factors = _factor_from_spf(n, spf)
    return len(factors) == 1 and factors[0][1] >= 2


def _is_squarefree_from_spf(n: int, spf: list[int]) -> bool:
    return all(exponent == 1 for _prime, exponent in _factor_from_spf(n, spf))


def _pair_list(primes: list[int], gap: int) -> list[list[int]]:
    pairs = []
    prime_set = set(primes)
    for prime in primes:
        partner = prime + gap
        if partner in prime_set:
            pairs.append([prime, partner])
    return pairs


def build_interval_stats(start: int, end: int, moduli: list[int]) -> dict:
    if start < 1:
        raise ValueError("Interval start must be >= 1.")
    if end < start:
        raise ValueError("Interval end must be >= start.")

    spf = _smallest_prime_factors(end)
    primes_upto_end = [n for n in range(2, end + 1) if _is_prime_from_spf(n, spf)]
    interval_values = list(range(start, end + 1))
    interval_primes = [n for n in interval_values if _is_prime_from_spf(n, spf)]
    interval_semiprimes = [n for n in interval_values if _is_semiprime_from_spf(n, spf)]

    gaps = [
        interval_primes[index + 1] - interval_primes[index]
        for index in range(len(interval_primes) - 1)
    ]
    residue_histograms = {}
    for modulus in moduli:
        residue_histograms[f"primes_mod_{modulus}"] = {}
        residue_histograms[f"semiprimes_mod_{modulus}"] = {}
        for value in interval_primes:
            key = str(value % modulus)
            residue_histograms[f"primes_mod_{modulus}"][key] = residue_histograms[f"primes_mod_{modulus}"].get(key, 0) + 1
        for value in interval_semiprimes:
            key = str(value % modulus)
            residue_histograms[f"semiprimes_mod_{modulus}"][key] = residue_histograms[f"semiprimes_mod_{modulus}"].get(key, 0) + 1

    payload = {
        "schema": SCHEMA_PATH,
        "range": {"start": start, "end": end},
        "counts": {
            "total": end - start + 1,
            "units": 1 if start <= 1 <= end else 0,
            "primes": len(interval_primes),
            "composites": sum(1 for n in interval_values if n > 1 and not _is_prime_from_spf(n, spf)),
            "semiprimes": len(interval_semiprimes),
            "prime_powers": sum(1 for n in interval_values if _is_prime_power_from_spf(n, spf)),
            "squarefree": sum(1 for n in interval_values if _is_squarefree_from_spf(n, spf)),
        },
        "prime_counting": {
            "pi_end": len(primes_upto_end),
            "pi_before_start": sum(1 for value in primes_upto_end if value < start),
            "pi_interval": len(interval_primes),
        },
        "gaps": {
            "max_prime_gap": None if not gaps else max(gaps),
            "mean_prime_gap": None if not gaps else sum(gaps) / len(gaps),
        },
        "constellations": {
            "twin_prime_pairs": _pair_list(interval_primes, 2),
            "cousin_prime_pairs": _pair_list(interval_primes, 4),
            "sexy_prime_pairs": _pair_list(interval_primes, 6),
        },
        "residue_histograms": residue_histograms,
        "qa_overlay": {
            f"mod_{modulus}": {
                "candidate_prime_residues": prime_residues(modulus),
                "obstructed_prime_residues": obstructed_prime_residues(modulus),
                "reachable_prime_residues": sorted(
                    set(prime_residues(modulus)).difference(obstructed_prime_residues(modulus))
                ),
                "candidate_semiprime_residues": semiprime_residues(modulus),
                "obstructed_semiprime_residues": obstructed_semiprime_residues(modulus),
                "reachable_semiprime_residues": sorted(
                    set(semiprime_residues(modulus)).difference(obstructed_semiprime_residues(modulus))
                ),
            }
            for modulus in moduli
        },
    }
    payload["canonical_hash"] = domain_sha256(
        "QA_INTERVAL_STATS.v1",
        canonical_json(payload),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute exact interval statistics for primes and related families.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--moduli", default="24,72")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    payload = build_interval_stats(int(args.start), int(args.end), _parse_moduli(args.moduli))
    if args.pretty:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
