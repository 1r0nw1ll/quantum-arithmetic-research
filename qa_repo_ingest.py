#!/usr/bin/env python3
"""
qa_repo_ingest.py

Entry: python qa_repo_ingest.py

Parses Bell test documents in the repo to add canonical entities:
- CHSH Bell Test, I₃₃₂₂ Bell Test, Platonic Solid Bell Tests
- Tsirelson bound, 8 | N Theorem, 6 | N Theorem
- Bell test experiments (hub entity)

Outputs: qa_entities_repo.json with entities and relationships.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


TARGET_FILES = [
    Path("BELL_TESTS_FINAL_SUMMARY.md"),
    Path("I3322_FINAL_VALIDATION.md"),
    Path("BELL_TEST_IMPLEMENTATIONS_SUMMARY.md"),
    Path("I3322_COEFFICIENT_FINDINGS.md"),
]


@dataclass
class Entity:
    name: str
    definition: str
    symbols: List[str]
    source_section: str
    type: str = "concept"


@dataclass
class Relationship:
    source: str
    target: str
    relationship: str = "RELATED_TO"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def extract_section(md: str, heading: str) -> str:
    # Get content under a Markdown heading starting with '### <heading>'
    pat = re.compile(r"^###\s+\d*\.?\s*" + re.escape(heading) + r"\s*$", re.M)
    m = pat.search(md)
    if not m:
        return ""
    start = m.end()
    # until next '### '
    next_m = re.search(r"^###\s+", md[start:], flags=re.M)
    end = start + (next_m.start() if next_m else len(md))
    return md[start:end].strip()


def ingest_repo() -> Tuple[List[Entity], List[Relationship]]:
    entities: Dict[str, Entity] = {}
    rels: List[Relationship] = []

    bell_hub = Entity(
        name="Bell test experiments",
        definition="Collection of Bell inequality tests reconstructed in the QA framework (CHSH, I₃₃₂₂, Platonic).",
        symbols=[],
        source_section="Repo",
    )
    entities[bell_hub.name] = bell_hub

    # Parse main bell summary
    bell_md = read_text(TARGET_FILES[0])
    if bell_md:
        # CHSH
        chsh_sec = extract_section(bell_md, "CHSH Bell Test (`qa_chsh_bell_test.py`)") or extract_section(bell_md, "CHSH Bell Test")
        if chsh_sec:
            defn = "Reconstructed CHSH Bell test; reproduces Tsirelson bound; verifies \"8 | N\" theorem."
            entities["CHSH Bell Test"] = Entity("CHSH Bell Test", defn, [], "Repo")
            rels.append(Relationship(source=bell_hub.name, target="CHSH Bell Test", relationship="INCLUDES"))

        # Platonic
        plat_sec = extract_section(bell_md, "Platonic Solid Bell Tests (`qa_platonic_bell_tests.py`)") or extract_section(bell_md, "Platonic Solid Bell Tests")
        if plat_sec:
            defn = "Bell tests using Platonic solids (octahedron, icosahedron, dodecahedron); kernel limitations observed."
            entities["Platonic Solid Bell Tests"] = Entity("Platonic Solid Bell Tests", defn, [], "Repo")
            rels.append(Relationship(source=bell_hub.name, target="Platonic Solid Bell Tests", relationship="INCLUDES"))

            # Include individual solids as entities
            entities.setdefault("Octahedron", Entity("Octahedron", "Platonic solid with 6 vertices used for Bell tests.", [], "Repo"))
            entities.setdefault("Icosahedron", Entity("Icosahedron", "Platonic solid with 12 vertices used for Bell tests.", [], "Repo"))
            entities.setdefault("Dodecahedron", Entity("Dodecahedron", "Platonic solid with 20 vertices used for Bell tests.", [], "Repo"))
            rels.append(Relationship(source="Platonic Solid Bell Tests", target="Octahedron", relationship="INCLUDES"))
            rels.append(Relationship(source="Platonic Solid Bell Tests", target="Icosahedron", relationship="INCLUDES"))
            rels.append(Relationship(source="Platonic Solid Bell Tests", target="Dodecahedron", relationship="INCLUDES"))

        # Tsirelson bound mention
        if re.search(r"Tsirelson", bell_md, re.I):
            entities["Tsirelson bound"] = Entity("Tsirelson bound", "Quantum bound S = 2√2 for CHSH.", [], "Repo")
            rels.append(Relationship(source="CHSH Bell Test", target="Tsirelson bound", relationship="ACHIEVES"))

        # Divisibility
        if re.search(r"8\s*\|\s*N", bell_md):
            entities["8 | N Theorem"] = Entity("8 | N Theorem", "CHSH achieves S = 2√2 exactly when N divisible by 8.", [], "Repo")
            rels.append(Relationship(source="CHSH Bell Test", target="8 | N Theorem", relationship="SATISFIES"))

        # Mathematical framework
        if re.search(r"Core QA Correlator|E_N\(s, t, N\)", bell_md):
            entities["QA modular correlator"] = Entity("QA modular correlator", "E_N(s,t) = cos(2π(s-t)/N) — universal QA correlator.", [], "Repo")
            # Link to tests as COMPUTES (tests compute Bell expressions from correlator)
            if "CHSH Bell Test" in entities:
                rels.append(Relationship(source="CHSH Bell Test", target="QA modular correlator", relationship="COMPUTES"))
            if "I₃₃₂₂ Bell Test" in entities:
                rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="QA modular correlator", relationship="COMPUTES"))
            if "Platonic Solid Bell Tests" in entities:
                rels.append(Relationship(source="Platonic Solid Bell Tests", target="QA modular correlator", relationship="COMPUTES"))

        # Classical/Quantum bounds
        if re.search(r"Classical bound", bell_md, re.I):
            entities["Classical bound"] = Entity("Classical bound", "Local realist (classical) limit for Bell expressions.", [], "Repo")
        if re.search(r"Quantum bound|Tsirelson", bell_md, re.I):
            entities.setdefault("Quantum bound", Entity("Quantum bound", "Maximum quantum value (e.g., Tsirelson bound).", [], "Repo"))
            if "CHSH Bell Test" in entities:
                rels.append(Relationship(source="CHSH Bell Test", target="Quantum bound", relationship="BOUNDED_BY"))
            if "I₃₃₂₂ Bell Test" in entities:
                rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="Quantum bound", relationship="BOUNDED_BY"))

        # Divisibility + N=24 universal
        if re.search(r"N = 24 is Universal|Why N=24 is Universal|LCM\(8,\s*6\)\s*=\s*24", bell_md):
            entities["N=24 Universal"] = Entity("N=24 Universal", "LCM(8,6)=24 satisfies CHSH and I₃₃₂₂ simultaneously; 15° resolution.", [], "Repo")
            if "CHSH Bell Test" in entities:
                rels.append(Relationship(source="N=24 Universal", target="CHSH Bell Test", relationship="ENABLED_FOR"))
            if "I₃₃₂₂ Bell Test" in entities:
                rels.append(Relationship(source="N=24 Universal", target="I₃₃₂₂ Bell Test", relationship="ENABLED_FOR"))
            if "Platonic Solid Bell Tests" in entities:
                rels.append(Relationship(source="N=24 Universal", target="Platonic Solid Bell Tests", relationship="ENABLED_FOR"))

    # I3322 validation
    i3322_md = read_text(TARGET_FILES[1])
    if i3322_md:
        if re.search(r"I[ _]?₃₃₂₂|I3322", i3322_md):
            defn = "QA I₃₃₂₂ Bell inequality implementation; vault convention confirms I = 5.0 when 6 | N."
            entities["I₃₃₂₂ Bell Test"] = Entity("I₃₃₂₂ Bell Test", defn, [], "Repo")
            rels.append(Relationship(source=bell_hub.name, target="I₃₃₂₂ Bell Test", relationship="INCLUDES"))
        if re.search(r"6\s*\|\s*N", i3322_md):
            entities["6 | N Theorem"] = Entity("6 | N Theorem", "I₃₃₂₂ achieves I = 5.0 exactly when N divisible by 6 (non-degenerate settings).", [], "Repo")
            rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="6 | N Theorem", relationship="SATISFIES"))

        # Strategies (degenerate/non-degenerate)
        if re.search(r"Degenerate|Repeated Angles", i3322_md):
            entities["Degenerate strategy"] = Entity("Degenerate strategy", "Repeated angles (e.g., {0°,0°,180°}) achieving I=5.0 in I₃₃₂₂.", [], "Repo")
            rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="Degenerate strategy", relationship="USES"))
        if re.search(r"Non-Degenerate|Distinct Angles|Trisymmetric", i3322_md):
            entities["Non-degenerate strategy"] = Entity("Non-degenerate strategy", "Distinct angles (e.g., {0°,120°,240°}) requiring 6|N.", [], "Repo")
            rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="Non-degenerate strategy", relationship="USES"))

        # Measurement settings
        entities.setdefault("Measurement settings", Entity("Measurement settings", "Pairs of measurement bases (A,A',B,B') for CHSH; 3×3 for I₃₃₂₂.", [], "Repo"))
        if "CHSH Bell Test" in entities:
            rels.append(Relationship(source="CHSH Bell Test", target="Measurement settings", relationship="USES"))
        if "I₃₃₂₂ Bell Test" in entities:
            rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="Measurement settings", relationship="USES"))

    # Simple cross-links
    if "CHSH Bell Test" in entities and "I₃₃₂₂ Bell Test" in entities:
        rels.append(Relationship(source="CHSH Bell Test", target="I₃₃₂₂ Bell Test", relationship="RELATED_TO"))
        rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="CHSH Bell Test", relationship="RELATED_TO"))

    # Implementations summary: additional kernels and families
    impl_md = read_text(TARGET_FILES[2])
    if impl_md:
        # Kernels
        if re.search(r"Multi-Harmonic Kernel", impl_md, re.I):
            entities["Multi-Harmonic Kernel"] = Entity("Multi-Harmonic Kernel", "E_multi(s,t)=Σ α_k cos(2πk(s-t)/N) — multi-frequency kernel.", [], "Repo")
        if re.search(r"Duo-Fibonacci Spectral Kernel", impl_md, re.I):
            entities["Duo-Fibonacci Spectral Kernel"] = Entity("Duo-Fibonacci Spectral Kernel", "Kernel integrating two Fibonacci modes into correlation structure.", [], "Repo")
        if re.search(r"Toroidal-Spherical Kernel", impl_md, re.I):
            entities["Toroidal-Spherical Kernel"] = Entity("Toroidal-Spherical Kernel", "Kernel mapping tuple states onto toroidal/spherical phase spaces.", [], "Repo")
        # Tuple engine & families
        if re.search(r"Modular QA Tuple Engine", impl_md, re.I):
            entities["Modular QA Tuple Engine"] = Entity("Modular QA Tuple Engine", "Tracks digital-root(9) and mod-24 residues; 24-step harmonic cycle.", [], "Repo")
        for fam in ["Fibonacci family", "Lucas family", "Phibonacci family", "Tribonacci family", "Ninbonacci family"]:
            entities[fam] = Entity(fam, f"QA digital-root family: {fam}.", [], "Repo")
            # Link family to engine
            rels.append(Relationship(source=fam, target="Modular QA Tuple Engine", relationship="RELATED_TO"))
        # Platonic comprehensive list
        for solid in ["Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron"]:
            if solid not in entities:
                entities[solid] = Entity(solid, f"Platonic solid: {solid}.", [], "Repo")
            if "Platonic Solid Bell Tests" in entities:
                rels.append(Relationship(source="Platonic Solid Bell Tests", target=solid, relationship="INCLUDES"))
        # CHSH details
        if re.search(r"Win probability|P_win", impl_md):
            entities["CHSH win probability"] = Entity("CHSH win probability", "P_win = cos²(π/8) ≈ 0.8536 for optimal CHSH.", [], "Repo")
            if "CHSH Bell Test" in entities:
                rels.append(Relationship(source="CHSH Bell Test", target="CHSH win probability", relationship="COMPUTES"))
        if re.search(r"Optimal Settings|Alice: 0° and 90°", impl_md):
            entities["CHSH optimal settings"] = Entity("CHSH optimal settings", "Alice: 0°,90°; Bob: ±45° at N=24.", [], "Repo")
            rels.append(Relationship(source="CHSH Bell Test", target="CHSH optimal settings", relationship="USES"))

    # I3322 coefficient findings: scaling and optimal settings
    i3322_coeff_md = read_text(TARGET_FILES[3])
    if i3322_coeff_md:
        if re.search(r"20\s*×|20x|20\s*\*", i3322_coeff_md, re.I):
            entities["I₃₃₂₂ scaling factor"] = Entity("I₃₃₂₂ scaling factor", "Vault uses 20× scaling vs literature (0.25 → 5.0).", [], "Repo")
            if "I₃₃₂₂ Bell Test" in entities:
                rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="I₃₃₂₂ scaling factor", relationship="USES"))
        if re.search(r"Optimal Settings|Alice's angles|Bob's angles", i3322_coeff_md, re.I):
            entities["I₃₃₂₂ optimal settings"] = Entity("I₃₃₂₂ optimal settings", "Alice: {0,8,16}; Bob: {2,10,18} on 24-gon.", [], "Repo")
            if "I₃₃₂₂ Bell Test" in entities:
                rels.append(Relationship(source="I₃₃₂₂ Bell Test", target="I₃₃₂₂ optimal settings", relationship="USES"))
        if re.search(r"Pisano|π\(9\)\s*=\s*24|Pisano period", i3322_coeff_md, re.I):
            entities["Pisano period π(9)=24"] = Entity("Pisano period π(9)=24", "Pisano period for mod-9 equals 24; connects to N=24 cycles.", [], "Repo")
            rels.append(Relationship(source="N=24 Universal", target="Pisano period π(9)=24", relationship="RELATED_TO"))

    return list(entities.values()), rels


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest repo Bell-test documents into entity catalog")
    parser.add_argument("--output", default="qa_entities_repo.json", help="Output JSON path")
    args = parser.parse_args()

    ents, rels = ingest_repo()
    payload = {
        "source": "repo",
        "counts": {"entities": len(ents), "relationships": len(rels)},
        "entities": [asdict(e) for e in ents],
        "relationships": [asdict(r) for r in rels],
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Repo ingest: {len(ents)} entities, {len(rels)} relationships → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
