#!/usr/bin/env python3
"""
LaCie migration — runs via Python to bypass macOS TCC shell restrictions.
Moves large data dirs to /Volumes/lacie, leaves symlinks in place.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

LACIE = Path("/Volumes/lacie/signal_experiments_offload")
REPO  = Path("/Users/player3/signal_experiments")
HOME  = Path("/Users/player3")

def log(msg): print(msg, flush=True)

def check_lacie():
    test = LACIE.parent / ".write_test"
    try:
        LACIE.mkdir(parents=True, exist_ok=True)
        test.touch(); test.unlink()
        log(f"✓ LaCie writable at {LACIE}")
    except Exception as e:
        log(f"✗ LaCie not writable: {e}"); sys.exit(1)

def move_and_link(src: Path, dst: Path):
    """rsync src/ → dst/, remove source, leave symlink at src."""
    if src.is_symlink():
        log(f"  SKIP (already symlink): {src}"); return
    if not src.exists():
        log(f"  SKIP (not found): {src}"); return
    log(f"  Moving: {src}")
    log(f"       → {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["rsync", "-a", "--remove-source-files",
         str(src) + "/", str(dst) + "/"],
        check=True
    )
    # Remove leftover empty dirs
    for dirpath, dirnames, filenames in os.walk(str(src), topdown=False):
        try: os.rmdir(dirpath)
        except OSError: pass
    if src.exists() and not src.is_symlink():
        shutil.rmtree(str(src), ignore_errors=True)
    src.symlink_to(dst)
    log(f"  ✓ Symlink: {src.name} → {dst}")

def move_file_and_link(src: Path, dst_dir: Path):
    """rsync single file, remove source, leave symlink."""
    if src.is_symlink():
        log(f"  SKIP (already symlink): {src}"); return
    if not src.exists():
        log(f"  SKIP (not found): {src}"); return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    log(f"  Moving file: {src.name} → {dst_dir}")
    subprocess.run(["rsync", "-a", str(src), str(dst_dir) + "/"], check=True)
    src.unlink()
    src.symlink_to(dst)
    log(f"  ✓ Symlink: {src.name} → {dst}")

def move_only(src: Path, dst: Path):
    """Move without leaving a symlink (Downloads — no code references)."""
    if not src.exists():
        log(f"  SKIP (not found): {src}"); return
    dst.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Moving (no symlink): {src.name} → {dst}")
    if src.is_dir():
        subprocess.run(
            ["rsync", "-a", "--remove-source-files",
             str(src) + "/", str(dst) + "/"], check=True)
        shutil.rmtree(str(src), ignore_errors=True)
    else:
        subprocess.run(["rsync", "-a", str(src), str(dst)], check=True)
        src.unlink()
    log(f"  ✓ Moved")

# ─────────────────────────────────────────────────────────────────────────────

log("=" * 60)
log("LaCie Migration — Python runner")
log("=" * 60)
check_lacie()

log("\n=== TIER 1: Corpus data ===")
move_and_link(REPO/"corpus/pepe_pose",          LACIE/"corpus/pepe_pose")
move_and_link(REPO/"corpus/cmu_mocap_zhou2019", LACIE/"corpus/cmu_mocap_zhou2019")
move_and_link(REPO/"corpus/modelnet40",         LACIE/"corpus/modelnet40")
move_and_link(REPO/"corpus/cmu_mocap_asfamc",   LACIE/"corpus/cmu_mocap_asfamc")

log("\n=== TIER 1: Experiment caches ===")
move_and_link(
    REPO/"experiments/qa_ml/cache_pepe_ch4_pose1_table_4_2_finetuned_full13",
    LACIE/"experiments/qa_ml/cache_pepe_ch4_pose1_table_4_2_finetuned_full13")
move_and_link(
    REPO/"experiments/qa_ml/pepe_ch2_rot3_rebuilt",
    LACIE/"experiments/qa_ml/pepe_ch2_rot3_rebuilt")
move_and_link(
    REPO/"experiments/qa_ml/cache_full_psp",
    LACIE/"experiments/qa_ml/cache_full_psp")

log("\n=== TIER 1: QA lab data ===")
move_and_link(REPO/"qa_lab/data/rruff_raman",         LACIE/"qa_lab/data/rruff_raman")
move_and_link(REPO/"qa_lab/data/rruff_zips",          LACIE/"qa_lab/data/rruff_zips")
move_and_link(REPO/"qa_lab/data/houston2013_raw",     LACIE/"qa_lab/data/houston2013_raw")
move_and_link(REPO/"qa_lab/data/cifar-10-batches-py", LACIE/"qa_lab/data/cifar-10-batches-py")

log("\n=== TIER 1: Results DB ===")
move_file_and_link(
    REPO/"results/qa_exact_orbit_theorem_demo_2026_06_09.db",
    LACIE/"results")

log("\n=== TIER 2: EEG archive (approved) ===")
move_and_link(
    REPO/"archive/phase_artifacts/phase2_data/eeg",
    LACIE/"archive/phase_artifacts/phase2_data/eeg")

log("\n=== TIER 3: Python envs ===")
move_and_link(REPO/".venv",                             LACIE/"venv/signal_experiments_venv")
move_and_link(REPO/"experiments/qa_ml/gptq_awq_env",   LACIE/"experiments/qa_ml/gptq_awq_env")

log("\n=== TIER 3: ~/.cache/torch ===")
move_and_link(HOME/".cache/torch", LACIE/"home_cache/torch")

log("\n=== TIER 3: Downloads (no symlinks) ===")
move_only(HOME/"Downloads/Wolfram Player 14.3",    LACIE/"home_downloads/Wolfram_Player_14.3")
move_only(HOME/"Downloads/Ring35_Dataset_Txt 2",   LACIE/"home_downloads/Ring35_Dataset_Txt_dup")
move_only(HOME/"Downloads/Claude.dmg",             LACIE/"home_downloads/Claude.dmg")
move_only(HOME/"Downloads/Codex.dmg",              LACIE/"home_downloads/Codex.dmg")
move_only(HOME/"Downloads/Ring35_Dataset_Txt.zip", LACIE/"home_downloads/Ring35_Dataset_Txt.zip")

log("\n=== Verifying symlinks ===")
checks = [
    REPO/"corpus/pepe_pose",
    REPO/"corpus/cmu_mocap_zhou2019",
    REPO/"experiments/qa_ml/pepe_ch2_rot3_rebuilt",
    REPO/"experiments/qa_ml/cache_pepe_ch4_pose1_table_4_2_finetuned_full13",
    REPO/"qa_lab/data/rruff_raman",
    REPO/"archive/phase_artifacts/phase2_data/eeg",
    REPO/".venv",
    HOME/".cache/torch",
]
ok = 0
for p in checks:
    if p.is_symlink():
        log(f"  ✓ {p.relative_to(p.parent.parent) if p.is_relative_to(REPO) else p} → {os.readlink(p)}")
        ok += 1
    else:
        log(f"  ✗ MISSING SYMLINK: {p}")

log(f"\n  {ok}/{len(checks)} symlinks verified")

log("\n=== Disk usage after migration ===")
subprocess.run(["df", "-h", "/"])
subprocess.run(["df", "-h", "/Volumes/lacie"])

log("\n=== DONE ===")
