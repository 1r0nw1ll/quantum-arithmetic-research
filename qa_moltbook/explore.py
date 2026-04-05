#!/usr/bin/env python3
"""QA Moltbook — Interactive Certificate Family Explorer.

Browse, search, and validate the 186 QA certificate families.

Usage:
    python qa_moltbook/explore.py                  # list all families
    python qa_moltbook/explore.py --search chromo  # search by keyword
    python qa_moltbook/explore.py --show 130       # show family details
    python qa_moltbook/explore.py --validate 130   # validate a family
    python qa_moltbook/explore.py --stats          # ecosystem statistics
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PTOLEMY = REPO_ROOT / "qa_alphageometry_ptolemy"
META_VALIDATOR = PTOLEMY / "qa_meta_validator.py"
DOCS_FAMILIES = REPO_ROOT / "docs" / "families"


def load_families():
    """Extract family registry from qa_meta_validator.py FAMILY_SWEEPS."""
    families = []
    src = (PTOLEMY / "qa_meta_validator.py").read_text()
    in_sweeps = False
    buf = ""
    for line in src.splitlines():
        if "FAMILY_SWEEPS = [" in line:
            in_sweeps = True
            buf = "["
            continue
        if in_sweeps:
            buf += line + "\n"
            if line.strip() == "]":
                break

    # Parse tuples manually — they reference functions we can't import
    entries = []
    current = ""
    depth = 0
    for ch in buf:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                current += ch
                entries.append(current.strip())
                current = ""
                continue
        if depth > 0:
            current += ch

    for entry in entries:
        # Extract: (id, "label", func, "pass_desc", "doc_slug", "root", bool)
        # Use proper parsing that respects quoted strings with commas
        inner = entry.strip("()")
        parts = []
        cur = ""
        in_str = False
        str_char = None
        for ch in inner:
            if not in_str and ch in ('"', "'"):
                in_str = True
                str_char = ch
                cur += ch
            elif in_str and ch == str_char:
                in_str = False
                cur += ch
            elif not in_str and ch == ",":
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        if cur.strip():
            parts.append(cur.strip())

        if len(parts) < 6:
            continue
        try:
            fam_id = int(parts[0].strip())
        except ValueError:
            continue
        label = parts[1].strip().strip('"').strip("'")
        pass_desc = parts[3].strip().strip('"').strip("'")
        doc_slug = parts[4].strip().strip('"').strip("'")
        root_rel = parts[5].strip().strip('"').strip("'")

        # Check for doc file
        doc_path = DOCS_FAMILIES / f"{doc_slug}.md"
        has_doc = doc_path.exists()

        # Check for dedicated cert dir
        cert_dirs = list(PTOLEMY.glob(f"qa_*_cert_v*"))
        has_cert_dir = any(
            d.is_dir() and str(fam_id) in d.name
            for d in cert_dirs
        )

        families.append({
            "id": fam_id,
            "label": label,
            "pass_desc": pass_desc,
            "doc_slug": doc_slug,
            "root_rel": root_rel,
            "has_doc": has_doc,
            "has_cert_dir": has_cert_dir,
        })

    return sorted(families, key=lambda f: f["id"])


def cmd_list(families, args):
    """List all families."""
    print(f"QA Certificate Families ({len(families)} registered)\n")
    print(f"{'ID':>5s}  {'Label':<55s}  {'Doc':>3s}  {'Dir':>3s}")
    print("-" * 72)
    for f in families:
        doc = "Y" if f["has_doc"] else "-"
        cdir = "Y" if f["has_cert_dir"] else "-"
        print(f"[{f['id']:>3d}]  {f['label']:<55s}  {doc:>3s}  {cdir:>3s}")


def cmd_search(families, args):
    """Search families by keyword."""
    q = args.search.lower()
    hits = [f for f in families if q in f["label"].lower() or q in f["doc_slug"].lower()]
    if not hits:
        print(f"No families match '{args.search}'")
        return
    print(f"Families matching '{args.search}' ({len(hits)} results)\n")
    for f in hits:
        print(f"  [{f['id']:>3d}] {f['label']}")


def cmd_show(families, args):
    """Show details for one family."""
    fam = next((f for f in families if f["id"] == args.show), None)
    if not fam:
        print(f"Family [{args.show}] not found")
        return 1

    print(f"=== [{fam['id']}] {fam['label']} ===\n")
    print(f"  Validation: {fam['pass_desc']}")
    print(f"  Doc slug:   {fam['doc_slug']}")
    print(f"  Root:       {fam['root_rel']}")
    print(f"  Has doc:    {'Yes' if fam['has_doc'] else 'No'}")
    print(f"  Has dir:    {'Yes' if fam['has_cert_dir'] else 'No'}")

    # Show doc content if available
    doc_path = DOCS_FAMILIES / f"{fam['doc_slug']}.md"
    if doc_path.exists():
        text = doc_path.read_text()[:500]
        print(f"\n  --- Documentation (first 500 chars) ---")
        for line in text.splitlines():
            print(f"  {line}")
        if len(doc_path.read_text()) > 500:
            print(f"  ... ({len(doc_path.read_text())} chars total)")


def cmd_validate(families, args):
    """Run meta-validator for a specific family."""
    fam = next((f for f in families if f["id"] == args.validate), None)
    if not fam:
        print(f"Family [{args.validate}] not found")
        return 1

    print(f"Validating [{fam['id']}] {fam['label']}...")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(META_VALIDATOR)],
        cwd=str(PTOLEMY),
        capture_output=True,
        text=True,
        timeout=180,
    )
    elapsed = time.time() - t0

    # Find lines for this family
    for line in proc.stdout.splitlines():
        if f"[{fam['id']}]" in line:
            print(f"  {line.strip()}")

    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"\n  Overall: {status} ({elapsed:.1f}s)")
    return proc.returncode


def cmd_stats(families):
    """Show ecosystem statistics."""
    total = len(families)
    with_doc = sum(1 for f in families if f["has_doc"])
    with_dir = sum(1 for f in families if f["has_cert_dir"])
    id_range = f"[{families[0]['id']}]-[{families[-1]['id']}]" if families else "none"

    # Count domain categories from labels
    domains = {
        "math": ["pythagorean", "chromogeometry", "E8", "identities", "koenig", "fibonacci",
                 "spread", "conic", "quadrance", "eisenstein", "algebra"],
        "signal": ["EEG", "EMG", "audio", "cardiac", "seismic"],
        "geo": ["geodesi", "megalith", "celestial", "nav", "ellipsoid", "WGS", "bragg"],
        "graph": ["graph", "community", "modularity", "kernel"],
        "climate": ["climate", "ERA5", "teleconnect"],
        "finance": ["finance", "volatil", "coherence"],
        "infra": ["datastore", "ingest", "compiler", "pipeline", "agent", "guardrail",
                  "security", "mapping", "protocol", "spec"],
    }
    counts = {}
    for dname, keywords in domains.items():
        counts[dname] = sum(
            1 for f in families
            if any(k.lower() in f["label"].lower() for k in keywords)
        )

    print(f"QA Certificate Ecosystem\n")
    print(f"  Total families:  {total}")
    print(f"  ID range:        {id_range}")
    print(f"  With docs:       {with_doc}/{total} ({100*with_doc//total}%)")
    print(f"  With cert dirs:  {with_dir}/{total} ({100*with_dir//total}%)")
    print(f"\n  By domain:")
    for dname, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {dname:<12s} {count:3d}")
    uncategorized = total - sum(counts.values())
    if uncategorized > 0:
        print(f"    {'other':<12s} {uncategorized:3d}")


def main():
    parser = argparse.ArgumentParser(
        description="QA Moltbook — Certificate Family Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--search", "-s", help="Search families by keyword")
    parser.add_argument("--show", "-i", type=int, help="Show details for family ID")
    parser.add_argument("--validate", "-v", type=int, help="Validate family ID")
    parser.add_argument("--stats", action="store_true", help="Show ecosystem statistics")
    args = parser.parse_args()

    families = load_families()

    if args.search:
        cmd_search(families, args)
    elif args.show is not None:
        return cmd_show(families, args)
    elif args.validate is not None:
        return cmd_validate(families, args)
    elif args.stats:
        cmd_stats(families)
    else:
        cmd_list(families, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
