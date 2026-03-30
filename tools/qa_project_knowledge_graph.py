"""
qa_project_knowledge_graph.py

Builds a combined project knowledge graph (structural + concept layers)
from the QA certificate ecosystem, tools, skills, demos, experiments,
and documentation.

Entry point:
  python tools/qa_project_knowledge_graph.py [--output-dir DIR]

Outputs:
  - <output-dir>/project_knowledge_graph.graphml
  - <output-dir>/project_knowledge_graph_summary.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    print("ERROR: networkx is required. Install with: pip install networkx", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache",
    "node_modules", "target", "qa_venv", "venv",
}

TEXT_EXTS = {
    ".md", ".py", ".rs", ".toml", ".yaml", ".yml",
    ".json", ".txt", ".sh", ".tex", ".cff", ".tla",
}

MAX_FILE_SIZE = 1_000_000  # 1 MB

# ---------------------------------------------------------------------------
# Concept clusters
# ---------------------------------------------------------------------------

CONCEPT_CLUSTERS: Dict[str, List[str]] = {
    "E8_alignment":      ["e8", "alignment", "root system", "cosine"],
    "rule_30":           ["rule 30", "nonperiodicity", "wolfram", "cellular automaton"],
    "harmonic_index":    ["harmonic index", "harmonic coherence"],
    "hash_chain":        ["hash chain", "sha256", "canonical json", "domain-separated"],
    "mapping_protocol":  ["mapping protocol", "gate 0", "intake constitution"],
    "agent_trace":       ["agent trace", "hash-chain", "event trace"],
    "bell_test":         ["bell test", "chsh", "tsirelson", "i3322"],
    "EBM":               ["energy-based", "ebm", "navigation cert", "verifier bridge"],
    "competency":        ["competency", "dominance", "baseline", "separation"],
    "math_compiler":     ["math compiler", "human-formal", "binding cert"],
}

CONCEPT_THRESHOLD = 2  # must match >= 2 distinct keywords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _safe_read(path: Path) -> Optional[str]:
    """Read a text file, returning None on failure or binary."""
    if not path.is_file():
        return None
    if path.stat().st_size > MAX_FILE_SIZE:
        return None
    if path.suffix not in TEXT_EXTS:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _iter_tree(root: Path):
    """Iterate files under root, skipping SKIP_DIRS."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            yield Path(dirpath) / fn


# ---------------------------------------------------------------------------
# FAMILY_SWEEPS parser
# ---------------------------------------------------------------------------

_FAMILY_SWEEP_RE = re.compile(
    r"""\(\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*\w+\s*,\s*"[^"]*"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*(?:True|False)\s*\)""",
)


def parse_family_sweeps() -> List[Dict[str, Any]]:
    """Parse FAMILY_SWEEPS from qa_meta_validator.py source."""
    meta_path = REPO_ROOT / "qa_alphageometry_ptolemy" / "qa_meta_validator.py"
    if not meta_path.is_file():
        _log(f"  WARNING: meta-validator not found at {meta_path}")
        return []
    text = meta_path.read_text(encoding="utf-8", errors="replace")
    # Find the FAMILY_SWEEPS block
    start = text.find("FAMILY_SWEEPS = [")
    if start < 0:
        _log("  WARNING: FAMILY_SWEEPS not found in meta-validator")
        return []
    block = text[start:]
    end = block.find("\n]")
    if end > 0:
        block = block[: end + 2]

    families = []
    for m in _FAMILY_SWEEP_RE.finditer(block):
        fam_id = int(m.group(1))
        label = m.group(2)
        doc_slug = m.group(3)
        family_root_rel = m.group(4)
        families.append({
            "id": fam_id,
            "label": label,
            "doc_slug": doc_slug,
            "family_root_rel": family_root_rel,
        })
    _log(f"  Parsed {len(families)} families from FAMILY_SWEEPS")
    return families


# ---------------------------------------------------------------------------
# Structural node collectors
# ---------------------------------------------------------------------------

def collect_families(G: nx.DiGraph) -> None:
    _log("Collecting family nodes...")
    for fam in parse_family_sweeps():
        node_id = f"family:{fam['id']}"
        G.add_node(node_id, type="family", label=fam["label"],
                   family_id=fam["id"], doc_slug=fam["doc_slug"],
                   family_root_rel=fam["family_root_rel"])


def _glob_add(G: nx.DiGraph, node_type: str, patterns: List[str],
              base: Optional[Path] = None) -> List[Path]:
    """Glob patterns from base (default REPO_ROOT), add nodes, return paths."""
    base = base or REPO_ROOT
    found: List[Path] = []
    for pat in patterns:
        for p in sorted(base.glob(pat)):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            rel = _relative(p)
            node_id = f"{node_type}:{rel}"
            if node_id not in G:
                G.add_node(node_id, type=node_type, path=rel, label=p.name)
                found.append(p)
    return found


def collect_validators(G: nx.DiGraph) -> List[Path]:
    _log("Collecting validator nodes...")
    return _glob_add(G, "validator", [
        "**/validator*.py",
        "**/qa_*_validator*.py",
    ])


def collect_inline_validators(G: nx.DiGraph) -> None:
    """Detect families whose validator is defined inline in qa_meta_validator.py.

    Any family that has no standalone validator file after build_validator_edges()
    gets an inline_validator node + validates edge, reflecting the fact that its
    validation logic lives inside the meta-validator itself.
    """
    _log("Collecting inline validator nodes...")
    meta_path = "qa_alphageometry_ptolemy/qa_meta_validator.py"
    meta_node = f"validator:{meta_path}"

    # Find families with no inbound validates edge
    validated = set()
    for u, v, edata in G.edges(data=True):
        if edata.get("edge_type") == "validates" and G.nodes.get(v, {}).get("type") == "family":
            validated.add(v)

    all_families = {n for n, d in G.nodes(data=True) if d.get("type") == "family"}
    unvalidated = all_families - validated

    for fam_node in sorted(unvalidated):
        fam_data = G.nodes[fam_node]
        fid = fam_data.get("family_id", "?")
        inline_id = f"inline_validator:{meta_path}#family_{fid}"
        G.add_node(inline_id, type="inline_validator", path=meta_path,
                   label=f"inline@meta_validator#family_{fid}")
        G.add_edge(inline_id, fam_node, edge_type="validates")
        if meta_node in G:
            G.add_edge(meta_node, inline_id, edge_type="contains")
        _log(f"  Added inline validator for family [{fid}]")


def collect_schemas(G: nx.DiGraph) -> List[Path]:
    _log("Collecting schema nodes...")
    return _glob_add(G, "schema", [
        "**/*.schema.json",
        "**/schema.json",
    ])


def collect_tools(G: nx.DiGraph) -> List[Path]:
    _log("Collecting tool nodes...")
    return _glob_add(G, "tool", ["tools/*.py"])


def collect_skills(G: nx.DiGraph) -> List[Path]:
    _log("Collecting skill nodes...")
    found: List[Path] = []
    for skill_md in sorted(REPO_ROOT.glob("codex_skills/*/SKILL.md")):
        skill_dir = skill_md.parent
        rel = _relative(skill_dir)
        node_id = f"skill:{rel}"
        G.add_node(node_id, type="skill", path=rel, label=skill_dir.name)
        found.append(skill_dir)
        # Add contains edges for child files
        for child in sorted(skill_dir.iterdir()):
            if child.is_file():
                crel = _relative(child)
                child_id = f"skill_file:{crel}"
                G.add_node(child_id, type="skill_file", path=crel, label=child.name)
                G.add_edge(node_id, child_id, edge_type="contains")
    return found


def collect_demos(G: nx.DiGraph) -> List[Path]:
    _log("Collecting demo nodes...")
    return _glob_add(G, "demo", ["demos/*.py"])


def collect_experiments(G: nx.DiGraph) -> List[Path]:
    _log("Collecting experiment nodes...")
    patterns = ["*_experiment*.py", "run_*.py", "backtest_*.py"]
    found: List[Path] = []
    for pat in patterns:
        for p in sorted(REPO_ROOT.glob(pat)):
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
                rel = _relative(p)
                node_id = f"experiment:{rel}"
                if node_id not in G:
                    G.add_node(node_id, type="experiment", path=rel, label=p.name)
                    found.append(p)
    return found


def collect_documents(G: nx.DiGraph) -> List[Path]:
    _log("Collecting document nodes...")
    found: List[Path] = []
    for pat in ["docs/**/*.md", "Documents/*.md"]:
        for p in sorted(REPO_ROOT.glob(pat)):
            if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
                rel = _relative(p)
                node_id = f"document:{rel}"
                if node_id not in G:
                    G.add_node(node_id, type="document", path=rel, label=p.name)
                    found.append(p)
    return found


def collect_cert_fixtures(G: nx.DiGraph) -> List[Path]:
    _log("Collecting cert_fixture nodes...")
    return _glob_add(G, "cert_fixture", ["**/fixtures/*.json"])


def collect_certpacks(G: nx.DiGraph) -> List[Path]:
    _log("Collecting certpack nodes...")
    found: List[Path] = []
    for p in sorted(REPO_ROOT.glob("**/certpacks/*/")):
        if p.is_dir() and not any(part in SKIP_DIRS for part in p.parts):
            rel = _relative(p)
            node_id = f"certpack:{rel}"
            if node_id not in G:
                G.add_node(node_id, type="certpack", path=rel, label=p.name)
                found.append(p)
    return found


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def _resolve_family_root(root_rel: str) -> Path:
    """Resolve a family_root_rel to an absolute path."""
    if root_rel.startswith(".."):
        return (REPO_ROOT / "qa_alphageometry_ptolemy" / root_rel).resolve()
    elif root_rel == ".":
        return (REPO_ROOT / "qa_alphageometry_ptolemy").resolve()
    else:
        return (REPO_ROOT / "qa_alphageometry_ptolemy" / root_rel).resolve()


def build_family_containment_edges(G: nx.DiGraph) -> None:
    """Link families to fixtures and certpacks by directory containment."""
    _log("Building family containment edges...")
    family_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "family"]
    for fam_node in family_nodes:
        fam_data = G.nodes[fam_node]
        root_rel = fam_data.get("family_root_rel", ".")
        is_shared_root = (root_rel == ".")

        if is_shared_root:
            # Shared root: match fixtures by doc_slug substring in path
            slug = fam_data.get("doc_slug", "")
            terms = _slug_match_terms(slug)
            if not terms:
                continue
            for node_id, ndata in G.nodes(data=True):
                if ndata.get("type") in ("cert_fixture", "certpack"):
                    node_path = ndata.get("path", "")
                    for term in terms:
                        if term in node_path:
                            G.add_edge(fam_node, node_id, edge_type="has_fixture")
                            break
        else:
            # Dedicated root: use directory containment
            fam_root_str = str(_resolve_family_root(root_rel))
            for node_id, ndata in G.nodes(data=True):
                if ndata.get("type") in ("cert_fixture", "certpack"):
                    node_path = str((REPO_ROOT / ndata.get("path", "")).resolve())
                    if node_path.startswith(fam_root_str):
                        G.add_edge(fam_node, node_id, edge_type="has_fixture")


def _slug_match_terms(doc_slug: str) -> List[str]:
    """Generate match terms from a doc_slug, from most to least specific.

    E.g. "31_math_compiler_stack" -> ["math_compiler_stack", "math_compiler", "compiler_stack"]
    """
    parts = doc_slug.split("_", 1)
    suffix = parts[1] if len(parts) > 1 else doc_slug
    if not suffix:
        return []
    terms = [suffix]
    # Also try dropping the last segment for partial matches
    segs = suffix.split("_")
    if len(segs) > 1:
        terms.append("_".join(segs[:-1]))   # e.g. "math_compiler"
        terms.append("_".join(segs[1:]))    # e.g. "compiler_stack"
    return terms


def build_validator_edges(G: nx.DiGraph) -> None:
    """Link validators to families/schemas via filename and import analysis."""
    _log("Building validator edges...")
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") != "validator":
            continue
        path = REPO_ROOT / ndata["path"]
        content = _safe_read(path)
        path_str = str(path)

        # Link validator -> family by directory proximity
        for fam_node, fam_data in [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "family"]:
            root_rel = fam_data.get("family_root_rel", ".")
            is_shared = (root_rel == ".")
            if is_shared:
                # For shared-root families, try progressively looser slug matching
                slug = fam_data.get("doc_slug", "")
                for term in _slug_match_terms(slug):
                    if term in path_str:
                        G.add_edge(node_id, fam_node, edge_type="validates")
                        break
                continue
            fam_root = _resolve_family_root(root_rel)
            if str(path.resolve()).startswith(str(fam_root)):
                G.add_edge(node_id, fam_node, edge_type="validates")

        # Link validator -> schema by uses_schema
        if content:
            for schema_node, schema_data in [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "schema"]:
                schema_name = schema_data.get("label", "")
                if schema_name and schema_name in content:
                    G.add_edge(node_id, schema_node, edge_type="uses_schema")


def build_fixture_schema_edges(G: nx.DiGraph) -> None:
    """Link fixtures to schemas via $schema or schema fields in JSON."""
    _log("Building fixture-schema edges...")
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") != "cert_fixture":
            continue
        path = REPO_ROOT / ndata["path"]
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        schema_ref = data.get("$schema") or data.get("schema")
        if not isinstance(schema_ref, str):
            continue
        # Match against schema nodes
        for sn, sd in G.nodes(data=True):
            if sd.get("type") == "schema":
                if sd.get("label", "") in schema_ref or sd.get("path", "") in schema_ref:
                    G.add_edge(node_id, sn, edge_type="uses_schema")


def build_document_edges(G: nx.DiGraph) -> None:
    """Link documents to families via [NN]_ prefix in filename slug AND content refs."""
    _log("Building document edges...")
    family_ids = {int(d["family_id"]): n
                  for n, d in G.nodes(data=True) if d.get("type") == "family"}

    # Phase 1: filename slug matching (e.g. 37_ebm_navigation_cert.md)
    doc_slug_re = re.compile(r"^(\d+)_")
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") != "document":
            continue
        m = doc_slug_re.match(ndata.get("label", ""))
        if not m:
            continue
        doc_num = int(m.group(1))
        fam_node = f"family:{doc_num}"
        if fam_node in G:
            G.add_edge(node_id, fam_node, edge_type="documents")

    # Phase 2: content-based family references in docs without slug matches
    bracket_re = re.compile(r"\[(\d+)\]")
    family_kw_re = re.compile(r"family\s*\[?(\d+)\]?", re.IGNORECASE)
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") != "document":
            continue
        # Skip docs already linked by slug
        if any(G.edges[node_id, t].get("edge_type") == "documents"
               for t in G.successors(node_id)
               if G.nodes.get(t, {}).get("type") == "family"):
            continue
        path = REPO_ROOT / ndata.get("path", "")
        content = _safe_read(path)
        if not content:
            continue
        # Only use bracket refs that co-occur with "family" nearby
        for m in family_kw_re.finditer(content):
            ref_id = int(m.group(1))
            if ref_id in family_ids:
                fam_node = family_ids[ref_id]
                if not G.has_edge(node_id, fam_node):
                    G.add_edge(node_id, fam_node, edge_type="documents")


def build_two_hop_schema_edges(G: nx.DiGraph) -> None:
    """Derive family → uses_schema edges via two-hop closure through fixtures.

    If family --has_fixture--> fixture --uses_schema--> schema,
    then add family --uses_schema--> schema (derived).
    """
    _log("Building two-hop schema closure edges...")
    added = 0
    for fam_node, fam_data in G.nodes(data=True):
        if fam_data.get("type") != "family":
            continue
        for fixture in G.successors(fam_node):
            if G.edges[fam_node, fixture].get("edge_type") != "has_fixture":
                continue
            for schema in G.successors(fixture):
                if G.edges[fixture, schema].get("edge_type") != "uses_schema":
                    continue
                if not G.has_edge(fam_node, schema):
                    G.add_edge(fam_node, schema, edge_type="uses_schema",
                               derived="two_hop")
                    added += 1
    _log(f"  Added {added} derived family→schema edges (fixture path)")


def build_validator_fanout_schema_edges(G: nx.DiGraph) -> None:
    """Derive family → uses_schema edges via validator fanout.

    If validator --validates--> family AND validator --uses_schema--> schema,
    then add family --uses_schema--> schema (derived).
    """
    _log("Building validator fanout schema edges...")
    added = 0
    for val_node, val_data in G.nodes(data=True):
        if val_data.get("type") != "validator":
            continue
        # Collect families and schemas this validator connects to
        families: List[str] = []
        schemas: List[str] = []
        for target in G.successors(val_node):
            edge = G.edges[val_node, target]
            if edge.get("edge_type") == "validates":
                families.append(target)
            elif edge.get("edge_type") == "uses_schema":
                schemas.append(target)
        # Cross-product: every family gets every schema
        for fam in families:
            for schema in schemas:
                if not G.has_edge(fam, schema):
                    G.add_edge(fam, schema, edge_type="uses_schema",
                               derived="validator_fanout")
                    added += 1
    _log(f"  Added {added} derived family→schema edges (validator fanout)")


def build_import_edges(G: nx.DiGraph) -> None:
    """Link scripts to modules via import/from statements."""
    _log("Building import edges...")
    import_re = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE)
    script_types = {"tool", "demo", "experiment", "validator"}
    script_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") in script_types]

    # Build module name -> node_id lookup for all Python nodes
    py_nodes: Dict[str, str] = {}
    for n, d in G.nodes(data=True):
        p = d.get("path", "")
        if p.endswith(".py"):
            mod_name = Path(p).stem
            py_nodes[mod_name] = n

    for node_id, ndata in script_nodes:
        path = REPO_ROOT / ndata["path"]
        content = _safe_read(path)
        if not content:
            continue
        for m in import_re.finditer(content):
            mod = m.group(1).split(".")[0]
            target = py_nodes.get(mod)
            if target and target != node_id:
                G.add_edge(node_id, target, edge_type="imports")


# ---------------------------------------------------------------------------
# Concept layer
# ---------------------------------------------------------------------------

def build_concept_layer(G: nx.DiGraph) -> None:
    """Add concept nodes and mentions/co_occurrence edges."""
    _log("Building concept layer...")
    structural_types = {
        "family", "validator", "schema", "tool", "skill", "demo",
        "experiment", "document", "cert_fixture", "certpack",
    }

    # Read content cache: node_id -> lowercase content
    content_cache: Dict[str, str] = {}
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") not in structural_types:
            continue
        path_str = ndata.get("path")
        if not path_str:
            continue
        path = REPO_ROOT / path_str
        if path.is_dir():
            # For directories (skills, certpacks), concatenate readable children
            parts = []
            for child in sorted(path.iterdir()):
                c = _safe_read(child)
                if c:
                    parts.append(c)
            content = "\n".join(parts).lower() if parts else ""
        else:
            c = _safe_read(path)
            content = c.lower() if c else ""
        if content:
            content_cache[node_id] = content

    # For family nodes without path, try to read from meta-validator label
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") == "family" and node_id not in content_cache:
            content_cache[node_id] = ndata.get("label", "").lower()

    # Add concept nodes
    for concept in CONCEPT_CLUSTERS:
        G.add_node(f"concept:{concept}", type="concept", label=concept)

    # Build mentions edges
    concept_members: Dict[str, Set[str]] = {c: set() for c in CONCEPT_CLUSTERS}

    for node_id, content in content_cache.items():
        for concept, keywords in CONCEPT_CLUSTERS.items():
            matches = sum(1 for kw in keywords if kw in content)
            if matches >= CONCEPT_THRESHOLD:
                G.add_edge(node_id, f"concept:{concept}", edge_type="mentions")
                concept_members[concept].add(node_id)

    # Build co_occurrence edges between concepts sharing >= 2 structural nodes
    concept_list = list(CONCEPT_CLUSTERS.keys())
    for i, c1 in enumerate(concept_list):
        for c2 in concept_list[i + 1:]:
            shared = concept_members[c1] & concept_members[c2]
            if len(shared) >= 2:
                G.add_edge(f"concept:{c1}", f"concept:{c2}",
                           edge_type="co_occurrence", shared_count=len(shared))


# ---------------------------------------------------------------------------
# Family cross-reference edges
# ---------------------------------------------------------------------------

def build_family_dependency_edges(G: nx.DiGraph) -> None:
    """Detect cross-references between families via validators, READMEs, and docs."""
    _log("Building family dependency edges...")
    family_ids: Dict[int, str] = {}
    family_roots: Dict[int, str] = {}
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") == "family":
            fid = int(ndata["family_id"])
            family_ids[fid] = node_id
            family_roots[fid] = ndata.get("family_root_rel", ".")

    # Pattern: explicit "family [NN]" or "family_NN" references (high confidence)
    fam_explicit_re = re.compile(r"family[_\s]*\[?(\d+)\]?", re.IGNORECASE)
    # Pattern: bare bracket [NN] — only counted if "family" appears within 40 chars
    bracket_re = re.compile(r"\[(\d+)\]")
    PROXIMITY = 40  # chars to search for "family" context around bare [NN]

    def _extract_family_refs(content: str) -> Set[int]:
        """Extract family ID references with guardrails against footnote noise."""
        refs: Set[int] = set()
        # Explicit "family [NN]" — always trusted
        for m in fam_explicit_re.finditer(content):
            ref_id = int(m.group(1))
            if ref_id in family_ids:
                refs.add(ref_id)
        # Bare [NN] — only if "family" appears nearby or file is in a family root
        content_lower = content.lower()
        for m in bracket_re.finditer(content):
            ref_id = int(m.group(1))
            if ref_id not in family_ids:
                continue
            start = max(0, m.start() - PROXIMITY)
            end = min(len(content_lower), m.end() + PROXIMITY)
            window = content_lower[start:end]
            if "family" in window or "qa_" in window or "cert" in window:
                refs.add(ref_id)
        return refs

    # Scan validators AND documents AND READMEs in family roots
    scan_types = {"validator", "document"}
    scan_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") in scan_types]

    # Also scan README.md in each family root
    readme_nodes: List[Tuple[Optional[int], str]] = []
    for fid, root_rel in family_roots.items():
        fam_root = _resolve_family_root(root_rel)
        readme = fam_root / "README.md"
        if readme.is_file():
            readme_nodes.append((fid, str(readme)))

    def _find_owner_family(node_id: str, path: Path) -> Optional[int]:
        """Determine which family a node belongs to."""
        # Check validates/documents edges
        for target in G.successors(node_id):
            if G.nodes[target].get("type") == "family":
                return int(G.nodes[target].get("family_id", 0))
        # Check path containment for dedicated-root families
        path_str = str(path.resolve())
        for fid in sorted(family_ids.keys()):
            if family_roots[fid] != ".":
                fam_root = _resolve_family_root(family_roots[fid])
                if path_str.startswith(str(fam_root)):
                    return fid
        return None

    for node_id, ndata in scan_nodes:
        path = REPO_ROOT / ndata["path"]
        content = _safe_read(path)
        if not content:
            continue
        owner_fam = _find_owner_family(node_id, path)
        if owner_fam is None:
            continue
        for ref_id in _extract_family_refs(content):
            if ref_id != owner_fam:
                src = family_ids[owner_fam]
                tgt = family_ids[ref_id]
                if not G.has_edge(src, tgt):
                    G.add_edge(src, tgt, edge_type="depends_on")

    # Scan READMEs in family roots
    for fid, readme_path in readme_nodes:
        content = _safe_read(Path(readme_path))
        if not content:
            continue
        for ref_id in _extract_family_refs(content):
            if ref_id != fid:
                src = family_ids[fid]
                tgt = family_ids[ref_id]
                if not G.has_edge(src, tgt):
                    G.add_edge(src, tgt, edge_type="depends_on")


# ---------------------------------------------------------------------------
# Tool containment
# ---------------------------------------------------------------------------

def build_tool_containment_edges(G: nx.DiGraph) -> None:
    """Explicit contains edges for tools/ directory."""
    _log("Building tool containment edges...")
    tools_dir = REPO_ROOT / "tools"
    if not tools_dir.is_dir():
        return
    container_id = "dir:tools"
    G.add_node(container_id, type="directory", path="tools", label="tools/")
    for node_id, ndata in G.nodes(data=True):
        if ndata.get("type") == "tool":
            G.add_edge(container_id, node_id, edge_type="contains")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Build JSON-serializable summary statistics."""
    node_type_counts: Dict[str, int] = collections.Counter()
    edge_type_counts: Dict[str, int] = collections.Counter()

    for _, ndata in G.nodes(data=True):
        node_type_counts[ndata.get("type", "unknown")] += 1
    for _, _, edata in G.edges(data=True):
        edge_type_counts[edata.get("edge_type", "unknown")] += 1

    # Top-10 connected nodes by degree
    degree_list = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    top_connected = [
        {"node": n, "degree": d, "type": G.nodes[n].get("type", "unknown")}
        for n, d in degree_list
    ]

    # --- Orphan detection ---
    orphan_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    orphans_by_type: Dict[str, List[str]] = collections.defaultdict(list)
    for n in orphan_nodes:
        ntype = G.nodes[n].get("type", "unknown")
        orphans_by_type[ntype].append(n)
    orphan_summary = {
        "total": len(orphan_nodes),
        "by_type": {k: len(v) for k, v in sorted(orphans_by_type.items())},
    }
    # Flag orphaned families specifically (red flag)
    orphan_families = orphans_by_type.get("family", [])
    if orphan_families:
        orphan_summary["orphan_families"] = orphan_families

    # --- Coverage ratios ---
    n_families = node_type_counts.get("family", 0)
    n_fixtures = node_type_counts.get("cert_fixture", 0)
    n_validates = edge_type_counts.get("validates", 0)
    n_has_fixture = edge_type_counts.get("has_fixture", 0)
    n_uses_schema = edge_type_counts.get("uses_schema", 0)
    n_documents = edge_type_counts.get("documents", 0)

    # Count distinct families with at least one validates edge
    families_with_validator = set()
    families_with_fixture = set()
    families_with_doc = set()
    for u, v, edata in G.edges(data=True):
        etype = edata.get("edge_type", "")
        if etype == "validates" and G.nodes.get(v, {}).get("type") == "family":
            families_with_validator.add(v)
        if etype == "has_fixture" and G.nodes.get(u, {}).get("type") == "family":
            families_with_fixture.add(u)
        if etype == "documents" and G.nodes.get(v, {}).get("type") == "family":
            families_with_doc.add(v)

    # Granular schema coverage by source type
    fixtures_with_schema_ref = 0
    validators_with_schema = set()
    families_with_schema = set()
    for u, v, edata in G.edges(data=True):
        if edata.get("edge_type") != "uses_schema":
            continue
        src_type = G.nodes.get(u, {}).get("type", "")
        if src_type == "cert_fixture":
            fixtures_with_schema_ref += 1
        elif src_type == "validator":
            validators_with_schema.add(u)
        elif src_type == "family":
            families_with_schema.add(u)

    coverage = {
        "families_with_validator": f"{len(families_with_validator)}/{n_families}",
        "families_with_fixture": f"{len(families_with_fixture)}/{n_families}",
        "families_with_doc": f"{len(families_with_doc)}/{n_families}",
        "families_with_schema": f"{len(families_with_schema)}/{n_families}",
        "validators_with_schema": f"{len(validators_with_schema)}/{node_type_counts.get('validator', 0)}",
        "fixtures_with_schema_ref": f"{fixtures_with_schema_ref}/{n_fixtures}",
    }
    # List unvalidated families
    all_family_nodes = {n for n, d in G.nodes(data=True) if d.get("type") == "family"}
    unvalidated = sorted(all_family_nodes - families_with_validator)
    if unvalidated:
        coverage["unvalidated_families"] = unvalidated
    undocumented = sorted(all_family_nodes - families_with_doc)
    if undocumented:
        coverage["undocumented_families"] = undocumented

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "top_10_connected": top_connected,
        "orphans": orphan_summary,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # Structural layer
    collect_families(G)
    collect_validators(G)
    collect_schemas(G)
    collect_tools(G)
    collect_skills(G)
    collect_demos(G)
    collect_experiments(G)
    collect_documents(G)
    collect_cert_fixtures(G)
    collect_certpacks(G)

    _log(f"Structural nodes: {G.number_of_nodes()}")

    # Edges (phase 1: direct links)
    build_family_containment_edges(G)
    build_validator_edges(G)
    build_fixture_schema_edges(G)
    build_document_edges(G)
    build_import_edges(G)
    build_tool_containment_edges(G)
    build_family_dependency_edges(G)

    # Derived edges (phase 2: closures + inline detection)
    build_two_hop_schema_edges(G)
    build_validator_fanout_schema_edges(G)
    collect_inline_validators(G)

    # Concept layer
    build_concept_layer(G)

    _log(f"Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    return G


# ---------------------------------------------------------------------------
# --check mode (CI-friendly health gate)
# ---------------------------------------------------------------------------

def check_graph_health(summary: Dict[str, Any]) -> List[str]:
    """Return a list of health check failures. Empty list = healthy."""
    failures: List[str] = []
    coverage = summary.get("coverage", {})
    orphans = summary.get("orphans", {})

    # 1. No orphaned families
    orphan_families = orphans.get("orphan_families", [])
    if orphan_families:
        failures.append(f"ORPHAN_FAMILIES: {orphan_families}")

    # 2. All families documented
    undoc = coverage.get("undocumented_families", [])
    if undoc:
        failures.append(f"UNDOCUMENTED_FAMILIES: {undoc}")

    # 3. All families validated (file-backed or inline)
    unval = coverage.get("unvalidated_families", [])
    if unval:
        failures.append(f"UNVALIDATED_FAMILIES: {unval}")

    # 4. Orphan count sanity (warn if > 50% of nodes are orphans)
    total_nodes = summary.get("total_nodes", 1)
    orphan_total = orphans.get("total", 0)
    if total_nodes > 0 and orphan_total / total_nodes > 0.5:
        failures.append(
            f"ORPHAN_RATIO_HIGH: {orphan_total}/{total_nodes} "
            f"({100*orphan_total/total_nodes:.0f}%)"
        )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build QA project knowledge graph (structural + concept layers)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="_forensics",
        help="Output directory (default: _forensics/)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Run health checks and exit nonzero on failure.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("=== QA Project Knowledge Graph Builder ===")
    G = build_graph()

    # Export GraphML
    graphml_path = output_dir / "project_knowledge_graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    _log(f"Wrote {graphml_path}")

    # Export summary JSON
    summary = build_summary(G)
    summary_path = output_dir / "project_knowledge_graph_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"Wrote {summary_path}")

    # Print summary to stdout
    print(json.dumps(summary, indent=2))

    # Health check mode
    if args.check:
        failures = check_graph_health(summary)
        if failures:
            _log("\n=== HEALTH CHECK FAILED ===")
            for f in failures:
                _log(f"  FAIL: {f}")
            sys.exit(1)
        else:
            _log("\n=== HEALTH CHECK PASSED ===")


if __name__ == "__main__":
    main()
