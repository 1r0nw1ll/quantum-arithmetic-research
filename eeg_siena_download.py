#!/usr/bin/env python3
"""Download Siena Scalp EEG Database from PhysioNet using urllib (no wget needed).

Run from signal_experiments/ root:
  python3 eeg_siena_download.py

Data lands at: archive/phase_artifacts/phase2_data/eeg/siena/
"""
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
DEST = Path(__file__).parent / "archive/phase_artifacts/phase2_data/eeg/siena"

PATIENTS = [
    "PN00", "PN01", "PN03", "PN05", "PN06", "PN07",
    "PN09", "PN10", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17",
]


def fetch_patient_files(patient):
    """Get list of EDF + TXT files for a patient from PhysioNet index."""
    url = f"{BASE}/{patient}/"
    with urllib.request.urlopen(url, timeout=30) as r:
        html = r.read().decode()
    files = re.findall(r'href="([^"]+\.(?:edf|txt))"', html)
    return [f for f in files if "/" not in f]   # local filenames only


def download_file(url, dest_path, label="", retries=3, timeout=600):
    dest_path = Path(dest_path)
    if dest_path.exists():
        remote_size = None
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=30
            ) as r:
                remote_size = int(r.headers.get("Content-Length", 0))
        except Exception:
            pass
        if remote_size and dest_path.stat().st_size == remote_size:
            print(f"  skip {label} (already complete)")
            return
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        print(f"  downloading {label} (attempt {attempt}) ...", end="", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as fh:
                chunk = 65536
                total = 0
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
                    total += len(buf)
            tmp.rename(dest_path)
            mb = total / 1e6
            print(f" {mb:.1f} MB")
            return
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f" ERROR: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for patient in PATIENTS:
        print(f"\n=== {patient} ===")
        pat_dir = DEST / patient
        pat_dir.mkdir(exist_ok=True)
        try:
            files = fetch_patient_files(patient)
        except Exception as e:
            print(f"  index fetch failed: {e}")
            continue
        for fname in sorted(files):
            url  = f"{BASE}/{patient}/{fname}"
            dest = pat_dir / fname
            download_file(url, dest, label=fname)

    print("\n--- Summary ---")
    for patient in PATIENTS:
        pat_dir = DEST / patient
        edfs = list(pat_dir.glob("*.edf")) if pat_dir.exists() else []
        txts = list(pat_dir.glob("*.txt")) if pat_dir.exists() else []
        tmps = list(pat_dir.glob("*.tmp")) if pat_dir.exists() else []
        sz   = sum(f.stat().st_size for f in edfs) / 1e6
        print(f"  {patient}: {len(edfs)} edfs, {len(txts)} txt, {len(tmps)} tmp, {sz:.0f} MB")


if __name__ == "__main__":
    main()
