#!/usr/bin/env python3
"""
qa_repo_health.py — QA Repo Health Scanner

Scans all cert family roots and produces a machine-readable health vector
with ranked weakness report. Goes beyond pass/fail (which the meta-validator
already does) to measure *coverage depth* per family.

Health axes per family:
  1. schema        — Does a JSON schema exist?
  2. validator     — Does a validator script exist?
  3. self_test     — Does the validator support --self-test?
  4. pos_fixtures  — Count of valid_* / *_valid* fixture files
  5. neg_fixtures  — Count of invalid_* / *_invalid* fixture files
  6. invariant_diff — Does the validator source reference invariant_diff?
  7. mapping_proto — mapping_protocol.json or mapping_protocol_ref.json present?
  8. human_doc     — Does docs/families/<slug>.md exist?
  9. readme        — Does a README.md exist in the family root?
  10. dedicated_root — Does the family have its own directory (not shared ".")?

Output:
  - Human-readable table to stdout
  - repo_health.json (machine-readable) with per-family scores + weakness ranking

Usage:
  python tools/qa_repo_health.py              # scan + report
  python tools/qa_repo_health.py --json       # JSON only to stdout
  python tools/qa_repo_health.py --weakness   # weakness ranking only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Family registry — mirrors FAMILY_SWEEPS from qa_meta_validator.py
# Format: (id, label, doc_slug, family_root_rel, dedicated_root, sub_root)
#
# family_root_rel is relative to qa_alphageometry_ptolemy/
# sub_root is the actual subdirectory within family_root where this family's
# artifacts (schema, validator, fixtures) live. None means root-level.
# For shared-root legacy families, sub_root disambiguates which subdir
# belongs to which family.
# ---------------------------------------------------------------------------

FAMILY_REGISTRY: List[Tuple[int, str, str, str, bool, Optional[str]]] = [
    # Legacy families [18]-[24]: validators at root, certs in certs/
    (18, "QA Datastore family", "18_datastore", ".", False, None),
    (19, "QA Topology Resonance bundle", "19_topology_resonance", ".", False, None),
    (20, "QA Datastore view family", "20_datastore_view", ".", False, None),
    (21, "QA A-RAG interface family", "21_arag_interface", ".", False, None),
    (22, "QA ingest->view bridge family", "22_ingest_view_bridge", ".", False, None),
    (23, "QA ingestion family", "23_ingestion", ".", False, None),
    (24, "QA SVP-CMC family", "24_svp_cmc", ".", False, None),
    # Bundle families [26]-[28]: validators at root
    (26, "QA Competency Detection family", "26_competency_detection", ".", False, None),
    (27, "QA Elliptic Correspondence bundle", "27_elliptic_correspondence", ".", False, None),
    (28, "QA Graph Structure bundle", "28_graph_structure", ".", False, None),
    # Families [29]-[34]: have dedicated subdirs within qa_alphageometry_ptolemy/
    (29, "QA Agent Trace family", "29_agent_traces", ".", False, "qa_agent_traces"),
    (30, "QA Agent Trace Competency Cert", "30_agent_trace_competency_cert", ".", False, "qa_agent_traces"),
    (31, "QA Math Compiler Stack", "31_math_compiler_stack", ".", False, "qa_math_compiler"),
    (32, "QA Conjecture-Prove Loop", "32_conjecture_prove_loop", ".", False, "qa_conjecture_prove"),
    (33, "QA Discovery Pipeline", "33_discovery_pipeline", ".", False, "qa_discovery_pipeline"),
    (34, "QA Rule 30 Certified Discovery", "34_rule30_cert", ".", False, "qa_rule30"),
    # New dedicated-root families [35]-[39]
    (35, "QA Mapping Protocol", "35_mapping_protocol", "../qa_mapping_protocol", True, None),
    (36, "QA Mapping Protocol REF", "36_mapping_protocol_ref", "../qa_mapping_protocol_ref", True, None),
    (37, "QA EBM Navigation Cert", "37_ebm_navigation_cert", "../qa_ebm_navigation_cert", True, None),
    (38, "QA Energy-Capability Separation", "38_energy_capability_separation", "../qa_energy_capability_separation_cert", True, None),
    (39, "QA EBM Verifier Bridge Cert", "39_ebm_verifier_bridge_cert", "../qa_ebm_verifier_bridge_cert", True, None),
]


# ---------------------------------------------------------------------------
# Health axes
# ---------------------------------------------------------------------------

AXES = [
    "schema", "validator", "self_test", "pos_fixtures", "neg_fixtures",
    "invariant_diff", "mapping_proto", "human_doc", "readme", "dedicated_root",
]

# Weights for weakness ranking (higher = more important gap)
AXIS_WEIGHTS: Dict[str, float] = {
    "schema": 3.0,
    "validator": 5.0,
    "self_test": 2.0,
    "pos_fixtures": 4.0,
    "neg_fixtures": 4.0,
    "invariant_diff": 3.0,
    "mapping_proto": 3.0,
    "human_doc": 2.0,
    "readme": 1.0,
    "dedicated_root": 1.0,
}


@dataclass
class FamilyHealth:
    family_id: int
    label: str
    doc_slug: str
    family_root_abs: str
    dedicated_root: bool

    # Boolean axes
    has_schema: bool = False
    has_validator: bool = False
    has_self_test: bool = False
    has_invariant_diff: bool = False
    has_mapping_proto: bool = False
    has_human_doc: bool = False
    has_readme: bool = False
    is_dedicated_root: bool = False

    # Count axes
    pos_fixture_count: int = 0
    neg_fixture_count: int = 0

    # Detail
    schema_paths: List[str] = field(default_factory=list)
    validator_paths: List[str] = field(default_factory=list)
    pos_fixture_paths: List[str] = field(default_factory=list)
    neg_fixture_paths: List[str] = field(default_factory=list)
    mapping_proto_mode: str = ""  # "inline" | "ref" | ""

    @property
    def score(self) -> float:
        """0.0 to 1.0 health score (weighted)."""
        total_weight = sum(AXIS_WEIGHTS.values())
        earned = 0.0
        earned += AXIS_WEIGHTS["schema"] * (1.0 if self.has_schema else 0.0)
        earned += AXIS_WEIGHTS["validator"] * (1.0 if self.has_validator else 0.0)
        earned += AXIS_WEIGHTS["self_test"] * (1.0 if self.has_self_test else 0.0)
        earned += AXIS_WEIGHTS["pos_fixtures"] * min(1.0, self.pos_fixture_count / 1.0)
        earned += AXIS_WEIGHTS["neg_fixtures"] * min(1.0, self.neg_fixture_count / 2.0)
        earned += AXIS_WEIGHTS["invariant_diff"] * (1.0 if self.has_invariant_diff else 0.0)
        earned += AXIS_WEIGHTS["mapping_proto"] * (1.0 if self.has_mapping_proto else 0.0)
        earned += AXIS_WEIGHTS["human_doc"] * (1.0 if self.has_human_doc else 0.0)
        earned += AXIS_WEIGHTS["readme"] * (1.0 if self.has_readme else 0.0)
        earned += AXIS_WEIGHTS["dedicated_root"] * (1.0 if self.is_dedicated_root else 0.0)
        return round(earned / total_weight, 3)

    @property
    def gaps(self) -> List[str]:
        """Return list of missing axes."""
        g = []
        if not self.has_schema:
            g.append("schema")
        if not self.has_validator:
            g.append("validator")
        if not self.has_self_test:
            g.append("self_test")
        if self.pos_fixture_count == 0:
            g.append("pos_fixtures")
        if self.neg_fixture_count < 2:
            g.append(f"neg_fixtures({self.neg_fixture_count}/2)")
        if not self.has_invariant_diff:
            g.append("invariant_diff")
        if not self.has_mapping_proto:
            g.append("mapping_proto")
        if not self.has_human_doc:
            g.append("human_doc")
        if not self.has_readme:
            g.append("readme")
        if not self.is_dedicated_root:
            g.append("dedicated_root")
        return g

    @property
    def weighted_gap(self) -> float:
        """Sum of weights for missing axes. Higher = worse."""
        gap = 0.0
        if not self.has_schema:
            gap += AXIS_WEIGHTS["schema"]
        if not self.has_validator:
            gap += AXIS_WEIGHTS["validator"]
        if not self.has_self_test:
            gap += AXIS_WEIGHTS["self_test"]
        if self.pos_fixture_count == 0:
            gap += AXIS_WEIGHTS["pos_fixtures"]
        if self.neg_fixture_count < 2:
            gap += AXIS_WEIGHTS["neg_fixtures"] * (1.0 - self.neg_fixture_count / 2.0)
        if not self.has_invariant_diff:
            gap += AXIS_WEIGHTS["invariant_diff"]
        if not self.has_mapping_proto:
            gap += AXIS_WEIGHTS["mapping_proto"]
        if not self.has_human_doc:
            gap += AXIS_WEIGHTS["human_doc"]
        if not self.has_readme:
            gap += AXIS_WEIGHTS["readme"]
        if not self.is_dedicated_root:
            gap += AXIS_WEIGHTS["dedicated_root"]
        return round(gap, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "label": self.label,
            "doc_slug": self.doc_slug,
            "family_root": self.family_root_abs,
            "score": self.score,
            "weighted_gap": self.weighted_gap,
            "axes": {
                "schema": self.has_schema,
                "validator": self.has_validator,
                "self_test": self.has_self_test,
                "pos_fixtures": self.pos_fixture_count,
                "neg_fixtures": self.neg_fixture_count,
                "invariant_diff": self.has_invariant_diff,
                "mapping_proto": self.has_mapping_proto,
                "human_doc": self.has_human_doc,
                "readme": self.has_readme,
                "dedicated_root": self.is_dedicated_root,
            },
            "mapping_proto_mode": self.mapping_proto_mode,
            "gaps": self.gaps,
        }


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _find_schemas(root: str) -> List[str]:
    """Find schema JSON files in root or immediate subdirectories."""
    found = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.lower() == "schema.json" and os.path.isfile(path):
            found.append(path)
    # Also check schemas/ subdir
    schemas_dir = os.path.join(root, "schemas")
    if os.path.isdir(schemas_dir):
        for name in os.listdir(schemas_dir):
            if name.endswith(".json") and os.path.isfile(os.path.join(schemas_dir, name)):
                found.append(os.path.join(schemas_dir, name))
    return found


def _find_validators(root: str) -> List[str]:
    """Find validator Python files."""
    found = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.endswith(".py") and "validator" in name.lower():
            found.append(path)
    # Check subdirectories one level deep
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path) and not sub.startswith("."):
            for name in os.listdir(sub_path):
                path = os.path.join(sub_path, name)
                if os.path.isfile(path) and name.endswith(".py") and "validator" in name.lower():
                    found.append(path)
    return found


def _check_self_test(validator_paths: List[str]) -> bool:
    """Check if any validator file contains --self-test support."""
    for vp in validator_paths:
        try:
            with open(vp, "r", encoding="utf-8") as f:
                src = f.read()
            if "--self-test" in src or "self_test" in src:
                return True
        except Exception:
            pass
    return False


def _check_invariant_diff(validator_paths: List[str]) -> bool:
    """Check if any validator references invariant_diff."""
    for vp in validator_paths:
        try:
            with open(vp, "r", encoding="utf-8") as f:
                src = f.read()
            if "invariant_diff" in src:
                return True
        except Exception:
            pass
    return False


def _find_fixtures(root: str) -> Tuple[List[str], List[str]]:
    """Find positive and negative fixture files.

    Convention: valid_* or *_valid* = positive, invalid_* or *_invalid* = negative.
    Searches fixtures/ subdirs recursively.
    """
    pos, neg = [], []

    def _scan_dir(d: str) -> None:
        if not os.path.isdir(d):
            return
        for entry in os.listdir(d):
            full = os.path.join(d, entry)
            if os.path.isfile(full) and entry.endswith(".json"):
                lower = entry.lower()
                if lower.startswith("valid") or "_valid" in lower:
                    if "invalid" not in lower:
                        pos.append(full)
                if lower.startswith("invalid") or "_invalid" in lower:
                    neg.append(full)
                # cert_neg_ pattern (used by rule30)
                if lower.startswith("cert_neg"):
                    neg.append(full)
            elif os.path.isdir(full) and entry not in ("__pycache__", ".git"):
                # Recurse into subdirs for negative fixture bundles
                if "neg" in entry.lower() or "invalid" in entry.lower():
                    neg.append(full)  # count the dir as one negative case

    # Check root/fixtures/
    fixtures_dir = os.path.join(root, "fixtures")
    _scan_dir(fixtures_dir)

    # Also check subdirectories that might have their own fixtures/
    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path) and not sub.startswith("."):
            sub_fixtures = os.path.join(sub_path, "fixtures")
            _scan_dir(sub_fixtures)

    return pos, neg


def _check_mapping_protocol(root: str) -> Tuple[bool, str]:
    """Check for mapping_protocol.json or mapping_protocol_ref.json."""
    inline = os.path.join(root, "mapping_protocol.json")
    ref = os.path.join(root, "mapping_protocol_ref.json")
    if os.path.isfile(inline):
        return True, "inline"
    if os.path.isfile(ref):
        return True, "ref"
    return False, ""


def scan_family(
    fam_id: int,
    label: str,
    doc_slug: str,
    family_root_abs: str,
    dedicated_root: bool,
    docs_dir: str,
    sub_root: Optional[str] = None,
) -> FamilyHealth:
    """Scan a single family and return its health.

    If sub_root is set, schema/validator/fixture scanning targets that
    subdirectory within family_root_abs instead of the root itself.
    Mapping protocol is always checked at family_root_abs (Gate 0 rule).
    """
    h = FamilyHealth(
        family_id=fam_id,
        label=label,
        doc_slug=doc_slug,
        family_root_abs=family_root_abs,
        dedicated_root=dedicated_root,
    )

    if not os.path.isdir(family_root_abs):
        return h

    # Artifact scan dir: sub_root if specified, else family_root
    artifact_dir = family_root_abs
    if sub_root:
        artifact_dir = os.path.join(family_root_abs, sub_root)
        if not os.path.isdir(artifact_dir):
            # sub_root doesn't exist — family has no local artifacts
            # Still check mapping proto and docs at root level
            h.has_mapping_proto, h.mapping_proto_mode = _check_mapping_protocol(family_root_abs)
            doc_file = os.path.join(docs_dir, f"{doc_slug}.md")
            h.has_human_doc = os.path.isfile(doc_file)
            h.is_dedicated_root = dedicated_root
            return h

    # Schema
    h.schema_paths = _find_schemas(artifact_dir)
    h.has_schema = len(h.schema_paths) > 0

    # Validator
    h.validator_paths = _find_validators(artifact_dir)
    # For legacy root-level families (no sub_root), also check root for
    # validators that match the family name pattern
    if not sub_root and not h.validator_paths:
        # These are the [18]-[28] families whose validators live at root
        h.validator_paths = _find_validators(family_root_abs)
    h.has_validator = len(h.validator_paths) > 0

    # Self-test
    h.has_self_test = _check_self_test(h.validator_paths)

    # Invariant diff
    h.has_invariant_diff = _check_invariant_diff(h.validator_paths)

    # Fixtures
    h.pos_fixture_paths, h.neg_fixture_paths = _find_fixtures(artifact_dir)
    h.pos_fixture_count = len(h.pos_fixture_paths)
    h.neg_fixture_count = len(h.neg_fixture_paths)

    # Mapping protocol (always at family_root level, per Gate 0)
    h.has_mapping_proto, h.mapping_proto_mode = _check_mapping_protocol(family_root_abs)

    # Human doc
    doc_file = os.path.join(docs_dir, f"{doc_slug}.md")
    h.has_human_doc = os.path.isfile(doc_file)

    # README
    readme_dir = artifact_dir if sub_root else family_root_abs
    h.has_readme = os.path.isfile(os.path.join(readme_dir, "README.md"))

    # Dedicated root
    h.is_dedicated_root = dedicated_root

    return h


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def scan_all(repo_root: str) -> List[FamilyHealth]:
    """Scan all registered families."""
    ptolemy_dir = os.path.join(repo_root, "qa_alphageometry_ptolemy")
    docs_dir = os.path.join(repo_root, "docs", "families")
    results = []

    for fam_id, label, doc_slug, family_root_rel, dedicated, sub_root in FAMILY_REGISTRY:
        family_root_abs = os.path.normpath(
            os.path.join(ptolemy_dir, family_root_rel)
        )
        h = scan_family(
            fam_id, label, doc_slug, family_root_abs,
            dedicated, docs_dir, sub_root=sub_root,
        )
        results.append(h)

    return results


def weakness_ranking(families: List[FamilyHealth]) -> List[Dict[str, Any]]:
    """Produce a weakness vector: which axes are weakest across all families."""
    axis_gaps: Dict[str, List[int]] = {a: [] for a in AXES}

    for h in families:
        if not h.has_schema:
            axis_gaps["schema"].append(h.family_id)
        if not h.has_validator:
            axis_gaps["validator"].append(h.family_id)
        if not h.has_self_test:
            axis_gaps["self_test"].append(h.family_id)
        if h.pos_fixture_count == 0:
            axis_gaps["pos_fixtures"].append(h.family_id)
        if h.neg_fixture_count < 2:
            axis_gaps["neg_fixtures"].append(h.family_id)
        if not h.has_invariant_diff:
            axis_gaps["invariant_diff"].append(h.family_id)
        if not h.has_mapping_proto:
            axis_gaps["mapping_proto"].append(h.family_id)
        if not h.has_human_doc:
            axis_gaps["human_doc"].append(h.family_id)
        if not h.has_readme:
            axis_gaps["readme"].append(h.family_id)
        if not h.is_dedicated_root:
            axis_gaps["dedicated_root"].append(h.family_id)

    total = len(families)
    ranking = []
    for axis in AXES:
        gap_count = len(axis_gaps[axis])
        pct = round(100.0 * gap_count / total, 1) if total else 0.0
        ranking.append({
            "axis": axis,
            "weight": AXIS_WEIGHTS[axis],
            "gap_count": gap_count,
            "gap_pct": pct,
            "impact": round(AXIS_WEIGHTS[axis] * pct, 1),
            "families_missing": axis_gaps[axis],
        })

    # Sort by impact descending
    ranking.sort(key=lambda r: r["impact"], reverse=True)
    return ranking


def print_table(families: List[FamilyHealth]) -> None:
    """Print human-readable health table."""
    # Sort by score ascending (worst first)
    by_score = sorted(families, key=lambda h: h.score)

    print("=" * 100)
    print("QA REPO HEALTH REPORT")
    print("=" * 100)
    print()

    # Summary
    scores = [h.score for h in families]
    avg = sum(scores) / len(scores) if scores else 0
    perfect = sum(1 for s in scores if s >= 1.0)
    print(f"Families scanned: {len(families)}")
    print(f"Average health:   {avg:.1%}")
    print(f"Perfect score:    {perfect}/{len(families)}")
    print()

    # Per-family table
    hdr = f"{'ID':>4}  {'Score':>6}  {'Sch':>3} {'Val':>3} {'ST':>3} {'P+':>3} {'N-':>3} {'ID':>3} {'MP':>3} {'Doc':>3} {'RM':>3} {'DR':>3}  Label"
    print(hdr)
    print("-" * len(hdr) + "-" * 20)

    for h in by_score:
        yn = lambda b: " Y " if b else " - "
        cnt = lambda n, thresh: f"{n:>3}" if n >= thresh else f" {n} "
        print(
            f"[{h.family_id:>2}]  {h.score:>5.0%}"
            f"  {yn(h.has_schema)}{yn(h.has_validator)}{yn(h.has_self_test)}"
            f"{cnt(h.pos_fixture_count, 1)}{cnt(h.neg_fixture_count, 2)}"
            f"{yn(h.has_invariant_diff)}{yn(h.has_mapping_proto)}"
            f"{yn(h.has_human_doc)}{yn(h.has_readme)}{yn(h.is_dedicated_root)}"
            f"  {h.label}"
        )

    print()
    print("Legend: Sch=schema Val=validator ST=self-test P+=positive_fixtures")
    print("        N-=negative_fixtures ID=invariant_diff MP=mapping_protocol")
    print("        Doc=human_doc RM=readme DR=dedicated_root")


def print_weakness(ranking: List[Dict[str, Any]]) -> None:
    """Print weakness ranking."""
    print()
    print("=" * 80)
    print("WEAKNESS RANKING (sorted by impact = weight x gap%)")
    print("=" * 80)
    print()
    print(f"{'Axis':<20} {'Weight':>6} {'Gap':>5} {'Gap%':>6} {'Impact':>7}  Missing in families")
    print("-" * 80)
    for r in ranking:
        fams = ", ".join(f"[{f}]" for f in r["families_missing"])
        if not fams:
            fams = "(none)"
        print(
            f"{r['axis']:<20} {r['weight']:>6.1f} {r['gap_count']:>5}"
            f" {r['gap_pct']:>5.1f}% {r['impact']:>7.1f}  {fams}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="QA Repo Health Scanner")
    ap.add_argument("--json", action="store_true", help="JSON output only")
    ap.add_argument("--weakness", action="store_true", help="Weakness ranking only")
    ap.add_argument("--out", default=None, help="Write JSON to file (default: repo_health.json)")
    args = ap.parse_args()

    # Resolve repo root (this script lives in tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, ".."))

    families = scan_all(repo_root)
    ranking = weakness_ranking(families)

    report = {
        "repo_root": repo_root,
        "families_scanned": len(families),
        "average_score": round(
            sum(h.score for h in families) / len(families), 3
        ) if families else 0,
        "families": [h.to_dict() for h in families],
        "weakness_ranking": ranking,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    elif args.weakness:
        print_weakness(ranking)
    else:
        print_table(families)
        print_weakness(ranking)

    # Write JSON output only if --out is explicitly given
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        if not args.json:
            print(f"\nJSON report written to: {args.out}")


if __name__ == "__main__":
    main()
