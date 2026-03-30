#!/usr/bin/env python3
"""
acquire_chbmit_scale.py — Smart CHB-MIT acquisition for full-scale QA analysis

Downloads only what the scaling analysis needs:
  1. Summary file for every patient (tiny — ~5KB each)
  2. Parse seizure annotations from summary
  3. Download only EDF files that contain seizures
  4. Download 2 interictal EDF files per patient (for baseline windows)

This avoids downloading the full ~6 GB dataset.
Estimated download: ~1.5–2.5 GB for 23 patients.

Usage:
    python acquire_chbmit_scale.py              # dry-run: show plan
    python acquire_chbmit_scale.py --download   # execute downloads
    python acquire_chbmit_scale.py --patient chb02 chb03  # specific patients
    python acquire_chbmit_scale.py --download --patient chb02

Progress is checkpointed: re-running skips already-downloaded files.
"""

import sys
import re
import time
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError

BASE_URL    = "https://physionet.org/files/chbmit/1.0.0"
CHBMIT_ROOT = Path(__file__).parent / "archive/phase_artifacts/phase2_data/eeg/chbmit"

# All 23 CHB-MIT patients
ALL_PATIENTS = [f"chb{i:02d}" for i in range(1, 24)]

# Approximate EDF file size (1-hour file at 256 Hz, 23 channels, 16-bit)
# = 23 channels × 256 samples/s × 3600 s × 2 bytes ≈ 42 MB
EDF_SIZE_MB_APPROX = 42.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wget(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download url → dest. Returns True on success. Skips if dest exists."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    [EXISTS] {dest.name}")
        return True
    if dry_run:
        print(f"    [DRY-RUN] would download: {url}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"    → {dest.name} ... ", end="", flush=True)
    try:
        urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"done ({size_mb:.1f} MB)")
        return True
    except (URLError, HTTPError, OSError) as e:
        print(f"FAILED: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def _parse_summary(summary_path: Path) -> dict:
    """
    Parse *-summary.txt → {
        'seizure_files': [fname, ...],    # EDF files with ≥1 seizure
        'total_seizures': int,
        'all_files': [fname, ...],        # all EDF files listed
    }
    """
    seizure_files = []
    all_files     = []
    current_file  = None
    n_seizures    = 0
    total         = 0

    with open(summary_path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m:
                current_file  = m.group(1).lower()
                n_seizures    = 0
                all_files.append(current_file)
                continue
            m = re.match(r"Number of Seizures in File:\s+(\d+)", line, re.I)
            if m and current_file:
                n_seizures = int(m.group(1))
                if n_seizures > 0:
                    seizure_files.append(current_file)
                total += n_seizures

    return {
        "seizure_files":  seizure_files,
        "total_seizures": total,
        "all_files":      all_files,
    }


def _pick_interictal_files(summary_info: dict, n: int = 2) -> list[str]:
    """
    Pick n EDF files that contain no seizures, spread across the recording.
    Prefer files from the middle of the recording (avoid artefact-heavy start/end).
    """
    no_seizure = [f for f in summary_info["all_files"]
                  if f not in summary_info["seizure_files"]]
    if not no_seizure:
        return []
    # Spread: take evenly spaced from the no-seizure pool
    step = max(1, len(no_seizure) // (n + 1))
    picks = [no_seizure[i * step] for i in range(1, n + 1) if i * step < len(no_seizure)]
    return picks[:n]


# ── Per-patient acquisition plan ──────────────────────────────────────────────

def build_plan(patient_id: str, dry_run: bool = False) -> dict | None:
    """
    Download summary, parse it, build acquisition plan.
    Returns plan dict or None if patient not available.
    """
    patient_dir  = CHBMIT_ROOT / patient_id
    summary_name = f"{patient_id}-summary.txt"
    summary_url  = f"{BASE_URL}/{patient_id}/{summary_name}"
    summary_path = patient_dir / summary_name

    print(f"\n  [{patient_id}] Fetching summary...")
    ok = _wget(summary_url, summary_path, dry_run=dry_run)
    if not ok or not summary_path.exists():
        print(f"  [{patient_id}] Summary unavailable — skipping")
        return None

    info = _parse_summary(summary_path)
    if not info["seizure_files"]:
        print(f"  [{patient_id}] No seizures in summary — skipping")
        return None

    interictal = _pick_interictal_files(info, n=2)
    files_needed = list(dict.fromkeys(info["seizure_files"] + interictal))

    est_mb = len(files_needed) * EDF_SIZE_MB_APPROX
    print(f"  [{patient_id}] {info['total_seizures']} seizures across "
          f"{len(info['seizure_files'])} files; "
          f"{len(interictal)} interictal; "
          f"est. {est_mb:.0f} MB")

    return {
        "patient_id":     patient_id,
        "patient_dir":    patient_dir,
        "files_needed":   files_needed,
        "seizure_files":  info["seizure_files"],
        "interictal":     interictal,
        "total_seizures": info["total_seizures"],
        "est_mb":         est_mb,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Acquire CHB-MIT data for QA scaling study")
    parser.add_argument("--download", action="store_true",
                        help="Execute downloads (default: dry-run)")
    parser.add_argument("--patient", nargs="+", metavar="chbNN",
                        help="Specific patients to process (default: all 23)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip patients with existing summary+EDFs")
    args = parser.parse_args()

    dry_run  = not args.download
    patients = args.patient if args.patient else ALL_PATIENTS

    if dry_run:
        print("DRY-RUN MODE — pass --download to execute")
        print()

    print("=" * 68)
    print("CHB-MIT Smart Acquisition — seizure EDF files only")
    print(f"Target root: {CHBMIT_ROOT}")
    print(f"Patients:    {patients}")
    print("=" * 68)

    plans  = []
    failed = []

    # Phase 1: fetch all summaries + build plans
    print("\nPhase 1: fetch summaries and build download plan")
    for pid in patients:
        plan = build_plan(pid, dry_run=dry_run)
        if plan:
            plans.append(plan)
        else:
            failed.append(pid)

    if not plans:
        print("\nNo downloadable patients found.")
        return

    # Summary table
    total_files = sum(len(p["files_needed"]) for p in plans)
    total_mb    = sum(p["est_mb"] for p in plans)
    print(f"\n  Download plan: {len(plans)} patients, "
          f"{total_files} EDF files, ~{total_mb:.0f} MB")

    if dry_run:
        print("\n  Per-patient plan:")
        for p in plans:
            existing = sum(
                1 for f in p["files_needed"]
                if (p["patient_dir"] / f).exists()
            )
            print(f"    {p['patient_id']}: {len(p['seizure_files'])} seizure files + "
                  f"{len(p['interictal'])} interictal, "
                  f"~{p['est_mb']:.0f} MB "
                  f"({existing}/{len(p['files_needed'])} already downloaded)")
        print()
        print("  Run with --download to execute.")
        return

    # Phase 2: download EDF files
    print("\nPhase 2: downloading EDF files")
    results = []
    t_start = time.time()

    for plan in plans:
        pid = plan["patient_id"]
        downloaded, skipped, errors = 0, 0, 0

        for fname in plan["files_needed"]:
            url  = f"{BASE_URL}/{pid}/{fname}"
            dest = plan["patient_dir"] / fname
            if dest.exists() and dest.stat().st_size > 1_000_000:
                skipped += 1
                print(f"    [EXISTS] {pid}/{fname}")
                continue
            ok = _wget(url, dest, dry_run=False)
            if ok:
                downloaded += 1
            else:
                errors += 1

        results.append({
            "patient": pid,
            "downloaded": downloaded,
            "skipped": skipped,
            "errors": errors,
        })

    elapsed = time.time() - t_start

    # Final report
    print()
    print("=" * 68)
    print("ACQUISITION COMPLETE")
    print(f"  Elapsed: {elapsed:.0f}s")
    print()
    print(f"  {'Patient':<8}  {'Downloaded':>10}  {'Skipped':>8}  {'Errors':>7}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*7}")
    for r in results:
        print(f"  {r['patient']:<8}  {r['downloaded']:>10}  {r['skipped']:>8}  {r['errors']:>7}")

    n_ready = sum(1 for r in results if r["errors"] == 0)
    print()
    print(f"  Patients ready for analysis: {n_ready}/{len(results)}")
    print()
    print("  Run the scaling analysis:")
    print("    python eeg_chbmit_scale.py")

    # Save acquisition manifest
    manifest = {
        "patients_planned": [p["patient_id"] for p in plans],
        "patients_failed":  failed,
        "results":          results,
        "total_elapsed_s":  elapsed,
    }
    manifest_path = CHBMIT_ROOT / "acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
