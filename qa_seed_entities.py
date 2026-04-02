#!/usr/bin/env python3
"""
qa_seed_entities.py

Emit a small seed catalog of high-value domain entities to enrich the graph
with classic network datasets and a domain-specific experiment.

Outputs: qa_entities_seed.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


@dataclass
class Entity:
    name: str
    definition: str
    symbols: List[str]
    source_section: str
    type: str = "domain"


@dataclass
class Relationship:
    source: str
    target: str
    relationship: str = "RELATED_TO"


def main() -> int:
    p = argparse.ArgumentParser(description="Write seed entities catalog")
    p.add_argument("--output", default="qa_entities_seed.json")
    args = p.parse_args()

    entities = [
        Entity(
            name="football network experiment",
            definition="Experiment analyzing football team/college network structure; classic social network benchmark.",
            symbols=[],
            source_section="Seeds",
        ),
        Entity(
            name="Zachary's Karate Club",
            definition="Classic social network of a university karate club used to study community detection.",
            symbols=[],
            source_section="Seeds",
        ),
        Entity(
            name="Les Misérables",
            definition="Character co-occurrence network from Victor Hugo's Les Misérables used as a graph benchmark.",
            symbols=[],
            source_section="Seeds",
        ),
    ]

    # Light connections to existing core concepts to ensure the nodes integrate into the graph
    rels = [
        Relationship("football network experiment", "QA‑Markovian system", "RELATED_TO"),
        Relationship("football network experiment", "Harmonic Index", "RELATED_TO"),
        Relationship("Zachary's Karate Club", "QA modular correlator", "RELATED_TO"),
        Relationship("Les Misérables", "QA modular correlator", "RELATED_TO"),
    ]

    payload = {
        "source": "seed",
        "counts": {"entities": len(entities), "relationships": len(rels)},
        "entities": [asdict(e) for e in entities],
        "relationships": [asdict(r) for r in rels],
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Wrote seeds: {len(entities)} entities, {len(rels)} relationships → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

