#!/usr/bin/env python3
"""Remove stalled .tmp files and re-download missing EDFs for held-out patients."""
import re
import time
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

BASE_URL    = "https://physionet.org/files/chbmit/1.0.0"
CHBMIT_ROOT = Path(__file__).parent / "archive/phase_artifacts/phase2_data/eeg/chbmit"

HELD_OUT = [
    "chb08","chb09","chb10","chb12","chb13","chb14",
    "chb15","chb16","chb17","chb19","chb21","chb22","chb23",
]

def clean_tmp(pdir):
    for tmp in pdir.glob("*.tmp"):
        print(f"  Removing stalled: {tmp.name}")
        tmp.unlink()

def parse_summary_files(summary_path):
    seizure_files, all_files = [], []
    cur = None
    with open(summary_path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.I)
            if m:
                cur = m.group(1).lower(); all_files.append(cur); continue
            m = re.match(r"Number of Seizures in File:\s+(\d+)", line, re.I)
            if m and cur and int(m.group(1)) > 0:
                seizure_files.append(cur)
    return seizure_files, all_files

def download_patient(pid):
    pdir = CHBMIT_ROOT / pid
    pdir.mkdir(parents=True, exist_ok=True)
    clean_tmp(pdir)

    summary_path = pdir / f"{pid}-summary.txt"
    if not summary_path.exists():
        url = f"{BASE_URL}/{pid}/{pid}-summary.txt"
        print(f"  [{pid}] Downloading summary...")
        try:
            urlretrieve(url, summary_path)
        except Exception as e:
            print(f"  [{pid}] Summary failed: {e}"); return False

    seizure_files, all_files = parse_summary_files(summary_path)
    if not seizure_files:
        print(f"  [{pid}] No seizures"); return False

    no_sei = [f for f in all_files if f not in seizure_files]
    step = max(1, len(no_sei) // 3)
    interictal = [no_sei[i] for i in range(step, len(no_sei), step)][:2]
    needed = list(dict.fromkeys(seizure_files + interictal))

    print(f"  [{pid}] {len(needed)} files needed")
    for fname in needed:
        dest = pdir / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"    [EXISTS] {fname}"); continue
        url = f"{BASE_URL}/{pid}/{fname}"
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        print(f"    {fname} ... ", end="", flush=True)
        try:
            urlretrieve(url, tmp)
            tmp.rename(dest)
            print(f"done ({dest.stat().st_size/1e6:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")
            if tmp.exists(): tmp.unlink()
    return True

for pid in HELD_OUT:
    pdir = CHBMIT_ROOT / pid
    if not pdir.exists():
        download_patient(pid)
    else:
        stalls = list(pdir.glob("*.tmp"))
        if stalls:
            print(f"\n[{pid}] Cleaning {len(stalls)} stalled file(s) and resuming")
            download_patient(pid)
        else:
            # Check for missing EDFs
            if not (pdir / f"{pid}-summary.txt").exists():
                download_patient(pid)
            else:
                seizure_files, all_files = parse_summary_files(pdir / f"{pid}-summary.txt")
                no_sei = [f for f in all_files if f not in seizure_files]
                step = max(1, len(no_sei) // 3)
                interictal = [no_sei[i] for i in range(step, len(no_sei), step)][:2]
                needed = list(dict.fromkeys(seizure_files + interictal))
                missing = [f for f in needed if not (pdir / f).exists() or (pdir / f).stat().st_size < 1_000_000]
                if missing:
                    print(f"\n[{pid}] {len(missing)} missing files, downloading...")
                    download_patient(pid)
                else:
                    print(f"[{pid}] complete ({len(needed)} files)")

print("\nDone. Run: python3 eeg_replication_heldout.py")
