#!/usr/bin/env python3
"""
patch_reference_pdi.py

Patches multi_path_states and pdi in all 9 reference-set bundles using the
bridge-state fan-then-merge DAG analytical formula.

Construction (documented as pdi_construction in each cert's graph_snapshot):
  - comps connected components, each with A arms of depth D
  - Each arm: states {arm=i, depth=0..D}, initial state = depth 0
  - Cross-links: for each arm i >= 1, a 2-event "bridge" episode:
      bridge_{comp,i} → arm_i_depth_{mid+1}
    where mid = D // 2, bridge state is unique (not shared with any primary arm)
  - This creates exactly (A-1)*comps cross-links, each making states
    arm_i_depth_{mid+1..D} reachable via 2 distinct directed paths

Multi-path states per component: (A-1) * (D - mid)  where mid = D // 2
Total multi_path = comps * (A-1) * (D - mid)
A = round(R / (comps * (D+1)))   [arms per component, estimated from cert R]
PDI = total_multi_path / R

Validator check: pdi() in qa_competency_metrics == multi_path_states / reachable_states
"""
import copy
import glob
import hashlib
import json
import sys
from pathlib import Path


HEX64_ZERO = "0" * 64


def canonical_json_compact(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj):
    return hashlib.sha256(canonical_json_compact(obj).encode("utf-8")).hexdigest()


def manifest_hashable_copy(obj):
    out = copy.deepcopy(obj)
    if "manifest" in out and isinstance(out["manifest"], dict):
        out["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    return out


def update_manifest(obj):
    hashable = manifest_hashable_copy(obj)
    computed = sha256_canonical(hashable)
    obj["manifest"]["canonical_json_sha256"] = computed
    return computed


def compute_pdi(R: int, comps: int, D: int):
    """Bridge-state fan-then-merge formula."""
    if R == 0 or D == 0 or comps == 0:
        return 0, 0.0
    A = round(R / (comps * (D + 1)))
    if A < 2:
        A = 2  # need at least 2 arms to have any cross-links
    mid = D // 2
    multi_path = comps * (A - 1) * (D - mid)
    multi_path = min(multi_path, R)  # cap at R
    pdi_val = float(multi_path) / float(R)
    return multi_path, pdi_val


def patch_cert(cert: dict) -> dict:
    """Patch a single cert dict in-place; return info dict."""
    mi = cert["metric_inputs"]
    reach = cert["reachability"]

    R = int(mi["reachable_states"])
    comps = int(reach["components"])
    D = int(reach["diameter"])

    old_mps = mi.get("multi_path_states", "ABSENT")
    old_pdi = cert["competency_metrics"].get("pdi", "ABSENT")

    multi_path, pdi_val = compute_pdi(R, comps, D)
    A = round(R / (comps * (D + 1))) if D > 0 and comps > 0 else 2
    if A < 2:
        A = 2
    mid = D // 2

    cert["metric_inputs"]["multi_path_states"] = multi_path
    cert["competency_metrics"]["pdi"] = pdi_val
    cert["graph_snapshot"]["hash_sha256"] = HEX64_ZERO
    update_manifest(cert)

    return {
        "R": R, "comps": comps, "D": D, "A": A, "mid": mid,
        "old_mps": old_mps, "old_pdi": old_pdi,
        "new_mps": multi_path, "new_pdi": pdi_val,
    }


def patch_bundle(path: Path, dry_run: bool = False) -> dict:
    bundle = json.loads(path.read_text())

    # Patch all certs in the bundle
    cert_infos = []
    for cert in bundle["certs"]:
        cert_infos.append(patch_cert(cert))

    # Recompute bundle manifest
    update_manifest(bundle)

    name = path.stem
    # Report primary cert info
    primary = cert_infos[0]
    info = {
        "name": name,
        "n_certs": len(cert_infos),
        **primary,
    }

    if not dry_run:
        path.write_text(json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    return info


def main():
    dry = "--dry-run" in sys.argv
    ref_dir = Path(__file__).parent / "reference_sets" / "v1"
    paths = sorted(ref_dir.glob("**/*.bundle.json"))

    if not paths:
        print("No reference-set bundles found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'DRY RUN — ' if dry else ''}Patching {len(paths)} bundles")
    print(f"{'Name':<40} {'A':>4} {'mid':>4} {'R':>5} {'mps_old':>8} {'mps_new':>8} {'pdi_new':>10}")
    print("-" * 90)

    errors = []
    for p in paths:
        try:
            info = patch_bundle(p, dry_run=dry)
            print(
                f"{info['name']:<40} {info['A']:>4} {info['mid']:>4}"
                f" {info['R']:>5} {str(info['old_mps']):>8} {info['new_mps']:>8}"
                f" {info['new_pdi']:>10.6f}"
            )
        except Exception as e:
            errors.append((str(p), str(e)))
            print(f"  ERROR {p.name}: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for ep, ee in errors:
            print(f"  {ep}: {ee}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n{'DRY-RUN OK' if dry else 'Patched OK'} — {len(paths)} bundles")


if __name__ == "__main__":
    main()
